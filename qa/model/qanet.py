import os
import shutil
import time
import math
import json
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from qa.util.file import FileUtil
from ..util.interpolation import interpolation
from ..util.tensor import tensor_from_var_2d_list
from ..util.metric import convert_tokens, evaluate_by_dict


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


def PosEncoder(x,
               min_timescale=1.0,
               max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)


def get_timing_signal(length,
                      channels,
                      min_timescale=1.0,
                      max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class Initialized_Conv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 relu=False,
                 bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=groups,
                             bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return nn.functional.relu(self.out(x))
        else:
            return self.out(x)


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):

    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList(
            [Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList(
            [Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            # gate = F.sigmoid(self.gate[i](x)) # deprecated
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(nn.Module):

    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head)
                for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Embedding(nn.Module):

    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(
            cemb_dim, d_model, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(
            wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):

    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(
            d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(
            1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model*2, 1)
        self.w2 = Initialized_Conv1d(d_model*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANetTokenizer:

    def __init__(self):
        self.tokenizer = spacy.blank("en")

    def prepare(self, text):
        return text.replace("''", '" ').replace("``", '" ')

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        return [token.text for token in tokens if token.text != " "]

    def to_index(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} not found!".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


class QANet(nn.Module):

    def __init__(self,
                 word_mat,
                 char_mat,
                 c_max_len,
                 q_max_len,
                 d_model,
                 train_cemb=False,
                 pad=0,
                 dropout=0.1,
                 num_head=1
                 ):
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_mat,
                                                         freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_mat)
        self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4,
                                    d_model=d_model,
                                    num_head=num_head,
                                    k=7,
                                    dropout=dropout)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(
            conv_num=2,
            d_model=d_model,
            num_head=num_head,
            k=5,
            dropout=dropout) for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) *
                 self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) *
                 self.PAD != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)


class QANetTrainer(object):

    def __init__(self,
                 device,
                 model,
                 loss,
                 optimizer,
                 use_scheduler,
                 scheduler,
                 use_grad_clip,
                 grad_clip,
                 use_ema,
                 ema,
                 use_kd,
                 interpolation,
                 truncation_max_len,
                 temperature,
                 alpha,
                 train_data_loader,
                 dev_data_loader,
                 dev_eval_dict,
                 epochs,
                 save_dir,
                 save_freq,
                 save_prefix,
                 resume,
                 print_freq,
                 evaluate_num_batches=500,
                 use_early_stop=False,
                 early_stop=10,
                 debug=False,
                 debug_num_examples=2,
                 ):
        self.device = device

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema = ema
        self.use_kd = use_kd
        self.interpolation = interpolation
        self.truncation_max_len = truncation_max_len
        self.temperature = temperature
        self.alpha = alpha

        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.dev_eval_dict = dev_eval_dict

        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_prefix = save_prefix + "_" if save_prefix != "" else ""
        self.resume = resume
        self.print_freq = print_freq
        self.evaluate_num_batches = evaluate_num_batches
        self.use_early_stop = use_early_stop
        self.early_stop = early_stop

        self.debug = debug
        self.debug_num_examples = debug_num_examples

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if resume:
            self._resume_checkpoint(resume)
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    def train(self):
        start = datetime.now()

        patience = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.use_early_stop:
                if result["f1"] < self.best_f1 and result["em"] < self.best_em:
                    patience += 1
                    if patience > self.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            is_best = False
            if result["f1"] > self.best_f1:
                is_best = True
            if result["f1"] == self.best_f1 and result["em"] > self.best_em:
                is_best = True
            self.best_f1 = max(self.best_f1, result["f1"])
            self.best_em = max(self.best_em, result["em"])

            if epoch % self.save_freq == 0:
                self._save_checkpoint(
                    epoch, result["f1"], result["em"], is_best)

        duration = datetime.now() - start
        print("Training Time: {}".format(duration))

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        global_loss = 0.0
        last_step = self.step - 1
        last_time = time.time()

        for batch_idx, batch in enumerate(self.train_data_loader):
            (context_wids,
             context_cids,
             question_wids,
             question_cids,
             y1,
             y2,
             y1s,
             y2s,
             id,
             tss,  # teacher start scores
             tes,  # teacher end scores
             tass,  # teacher all start scores
             taes,  # teacher all end scores
             answerable) = batch

            batch_num, question_len = question_wids.size()
            _, context_len = context_wids.size()
            context_wids = context_wids.to(self.device)
            context_cids = context_cids.to(self.device)
            question_wids = question_wids.to(self.device)
            question_cids = question_cids.to(self.device)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            id = id.to(self.device)
            answerable = answerable.to(self.device)

            self.model.zero_grad()
            p1, p2 = self.model(
                context_wids,
                context_cids,
                question_wids,
                question_cids
            )

            # Knowledge distillation & Interpolation
            if self.use_kd:
                tss_ts = p1.clone()
                tes_ts = p2.clone()
                for i, v in enumerate(tss):
                    for j, w in enumerate(tss[i]):
                        if tss[i][j] != 0:
                            tss_ts[i][j] = tss[i][j]
                        if tes[i][j] != 0:
                            tes_ts[i][j] = tes[i][j]

                mask_val = (-1e30)
                student_start_logits = p1.tolist()
                for i, v in enumerate(student_start_logits):
                    for j, w in enumerate(student_start_logits[i]):
                        if student_start_logits[i][j] <= mask_val:
                            student_start_logits[i] = student_start_logits[i][:j]
                            student_start_logits[i] = interpolation(
                                student_start_logits[i],
                                len(tass[i]),
                                kind=self.interpolation)
                            break

                student_end_logits = p2.tolist()
                for i, v in enumerate(student_end_logits):
                    for j, w in enumerate(student_end_logits[i]):
                        if student_end_logits[i][j] <= mask_val:
                            student_end_logits[i] = student_end_logits[i][:j]
                            student_end_logits[i] = interpolation(
                                student_end_logits[i],
                                len(taes[i]),
                                kind=self.interpolation)
                            break

                trunc_len = self.truncation_max_len  # 512
                student_start_logits_ts = tensor_from_var_2d_list(
                    student_start_logits,
                    padding=mask_val,
                    max_len=trunc_len,
                    requires_grad=True)
                student_start_logits_ts.to(self.device)
                student_end_logits_ts = tensor_from_var_2d_list(
                    student_end_logits,
                    padding=mask_val,
                    max_len=trunc_len,
                    requires_grad=True)
                student_end_logits_ts.to(self.device)
                teacher_start_logits_ts = tensor_from_var_2d_list(
                    tass,
                    padding=mask_val,
                    max_len=trunc_len,
                    requires_grad=False)
                teacher_start_logits_ts.to(self.device)
                teacher_end_logits_ts = tensor_from_var_2d_list(
                    taes,
                    padding=mask_val,
                    max_len=trunc_len,
                    requires_grad=False)
                teacher_end_logits_ts.to(self.device)

                # Interpolation Loss
                mseloss1 = F.mse_loss(
                    student_start_logits_ts, teacher_start_logits_ts)
                mseloss2 = F.mse_loss(
                    student_end_logits_ts, teacher_end_logits_ts)
                mseloss = (mseloss1 + mseloss2)

                # C-Hard Loss (Cross-Entropy)
                loss1 = self.loss(p1, y1)
                loss2 = self.loss(p2, y2)
                C_hard = (loss1 + loss2)

                # Knowledge Distillation Loss
                T = self.temperature
                klloss1 = T * T * F.kl_div(
                    F.log_softmax(p1/T, dim=1),
                    F.softmax(tss_ts/T, dim=1),
                    reduction="batchmean")
                klloss2 = T * T * F.kl_div(
                    F.log_softmax(p2/T, dim=1),
                    F.softmax(tes_ts/T, dim=1),
                    reduction="batchmean")
                C_soft = (klloss1 + klloss2)

                # Balancer
                alpha = self.alpha
                loss = (1. - alpha) * C_hard + alpha * C_soft

                loss.backward()
                global_loss += loss.item()

            else:
                loss1 = self.loss(p1, y1)
                loss2 = self.loss(p2, y2)
                loss = torch.mean(loss1 + loss2)
                loss.backward()
                global_loss += loss.item()

            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_clip)

            # Update model
            self.optimizer.step()

            # Update learning rate
            if self.use_scheduler:
                self.scheduler.step()

            # Use EMA
            if self.use_ema and self.ema is not None:
                self.ema(self.model, self.step)

            # Training information
            if self.step % self.print_freq == self.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_data_loader.batch_size * \
                    step_num / used_time
                batch_loss = global_loss / step_num
                s_tmp = "step: {}/{}\tepoch: {}\tlr: {}\tloss: {}\tspeed: {} ex/s"
                print(s_tmp.format(
                    batch_idx,
                    len(self.train_data_loader),
                    epoch,
                    self.scheduler.get_last_lr(),  # warning: get_lr()
                    batch_loss,
                    speed))
                global_loss = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.debug and batch_idx >= self.debug_num_examples:
                break

        metrics = self._evaluate_epoch(
            self.dev_eval_dict, self.dev_data_loader)
        print("dev_em: {}\tdev_f1: {}".format(
            metrics["exact_match"], metrics["f1"]))

        result = {}
        result["em"] = metrics["exact_match"]
        result["f1"] = metrics["f1"]
        return result

    def _evaluate_epoch(self, eval_dict, data_loader):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        self.model.eval()
        answer_dict = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                (context_wids,
                 context_cids,
                 question_wids,
                 question_cids,
                 y1,
                 y2,
                 y1s,
                 y2s,
                 id,
                 tss,  # teacher start scores
                 tes,  # teacher end scores
                 tass,  # teacher all start scores
                 taes,  # teacher all end scores
                 answerable) = batch
                context_wids = context_wids.to(self.device)
                context_cids = context_cids.to(self.device)
                question_wids = question_wids.to(self.device)
                question_cids = question_cids.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                answerable = answerable.to(self.device)

                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids)

                p1 = F.softmax(p1, dim=1)
                p2 = F.softmax(p2, dim=1)
                outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                for j in range(outer.size()[0]):
                    outer[j] = torch.triu(outer[j])

                a1, _ = torch.max(outer, dim=2)
                a2, _ = torch.max(outer, dim=1)
                ymin = torch.argmax(a1, dim=1)
                ymax = torch.argmax(a2, dim=1)
                answer_dict_, _ = convert_tokens(
                    eval_dict, id.tolist(), ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)

                if((batch_idx + 1) == self.evaluate_num_batches):
                    break

                if self.debug and batch_idx >= self.debug_num_examples:
                    break

        metrics = evaluate_by_dict(eval_dict, answer_dict)
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))

    def _save_checkpoint(self, epoch, f1, em, is_best):
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'step': self.step + 1,
            'start_time': self.start_time}
        filename = os.path.join(
            self.save_dir,
            self.save_prefix +
            'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(
                epoch, f1, em))
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(self.save_dir, 'model_best.pth.tar'))
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        return filename


class QANetEvaluator(object):

    def __init__(self,
                 device,
                 model,
                 use_ema,
                 dev_data_loader,
                 dev_eval_dict,
                 ema,
                 resume,
                 evaluation_results_file,
                 evaluation_answers_file,
                 evaluation_dev_eval_dict_file,
                 evaluation_predictions_file):
        self.device = device
        self.model = model
        self.dev_data_loader = dev_data_loader
        self.dev_eval_dict = dev_eval_dict
        self.use_ema = use_ema
        self.ema = ema
        self.resume = resume
        self.evaluation_results_file = evaluation_results_file
        self.evaluation_answers_file = evaluation_answers_file
        self.evaluation_dev_eval_dict_file = evaluation_dev_eval_dict_file
        self.evaluation_predictions_file = evaluation_predictions_file

    def evaluate(self):
        start_inference = datetime.now()

        checkpoint = torch.load(self.resume)
        self.model.load_state_dict(checkpoint["state_dict"])

        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        answer_dict = {}
        prediction_dict = {}
        data_loader_len = len(self.dev_data_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dev_data_loader):
                (context_wids,
                 context_cids,
                 question_wids,
                 question_cids,
                 _,  # y1
                 _,  # y2
                 _,  # y1s
                 _,  # y2s
                 id,
                 _,  # tss: teacher start scores
                 _,  # tes: teacher end scores
                 _,  # tass: teacher all start scores
                 _,  # taes: teacher all end scores
                 _  # answerable
                 ) = batch

                context_wids = context_wids.to(self.device)
                context_cids = context_cids.to(self.device)
                question_wids = question_wids.to(self.device)
                question_cids = question_cids.to(self.device)

                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids
                )

                p1 = F.softmax(p1, dim=1)
                p2 = F.softmax(p2, dim=1)
                outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                for j in range(outer.size()[0]):
                    outer[j] = torch.triu(outer[j])

                a1, _ = torch.max(outer, dim=2)
                a2, _ = torch.max(outer, dim=1)
                ymin = torch.argmax(a1, dim=1)
                ymax = torch.argmax(a2, dim=1)

                answer_dict_, prediction_dict_ = convert_tokens(
                    self.dev_eval_dict,
                    id.tolist(),
                    ymin.tolist(),
                    ymax.tolist()
                )
                for k in answer_dict_:
                    answer_dict[k] = answer_dict_[k]
                for k in prediction_dict_:
                    prediction_dict[k] = prediction_dict_[k]

        duration_inference = datetime.now() - start_inference
        start_evaluation = datetime.now()

        metrics = evaluate_by_dict(self.dev_eval_dict, answer_dict)

        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)

        duration_evaluation = datetime.now() - start_evaluation

        metrics["inference_time"] = str(duration_inference)
        metrics["evaluation_time"] = str(duration_evaluation)
        metrics["data_loader_len"] = data_loader_len
        print(json.dumps(metrics, sort_keys=False, indent=4))

        FileUtil(self.evaluation_results_file).save_json(metrics)
        FileUtil(self.evaluation_answers_file).save_json(answer_dict)
        FileUtil(self.evaluation_dev_eval_dict_file).save_json(
            self.dev_eval_dict)
        FileUtil(self.evaluation_predictions_file).save_json(prediction_dict)
