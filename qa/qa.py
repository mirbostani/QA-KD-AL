import os
import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from distutils.util import strtobool
from .util.file import FileUtil
from .dataset.dataset import Dataset
from .dataset.squad import SQuAD
from .model.qanet import QANet
from .model.qanet import QANetTokenizer
from .model.qanet import QANetTrainer
from .model.qanet import QANetEvaluator
from .model.bert import BertX
from .model.bert import BertTokenizerX
from .module.ema import EMA


class QuestionAnswering:
    def __init__(self):
        self.config = self.parse_arguments()
        self.summary()

        # GPU
        self.config.use_cuda &= torch.cuda.is_available()
        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        # Teacher Model
        self.teacher_model = None
        self.teacher_tokenizer = None
        if self.config.train and self.config.use_kd:
            if self.config.teacher == "bert":
                self.teacher_model = BertX(
                    device=self.device,
                    teacher_model_or_path=self.config.teacher_model_or_path,
                )
                self.teacher_tokenizer = BertTokenizerX(
                    teacher_tokenizer_or_path=self.config.teacher_tokenizer_or_path,
                )
            else:
                raise Exception("Invalid teacher model")

        # Student Model
        self.student_tokenizer = None
        if self.config.student == "qanet":
            self.student_tokenizer = QANetTokenizer()
        else:
            raise Exception("Invalid student model")

        # Preprocess Dataset
        self.dataset = Dataset(
            use_cuda=self.config.use_cuda,
            device=self.device,
            student_tokenizer=self.student_tokenizer,
            use_kd=self.config.use_kd,
            teacher_tokenizer=self.teacher_tokenizer,
            teacher_model=self.teacher_model,
            teacher_batch_size=self.config.teacher_batch_size,
            truncation_max_len=self.config.truncation_max_len,
            train_file=self.config.train_file,
            dev_file=self.config.dev_file,
            glove_word_file=self.config.glove_word_file,
            glove_word_size=self.config.glove_word_size,
            glove_word_dim=self.config.glove_word_dim,
            use_pretrained_char=self.config.use_pretrained_char,
            glove_char_file=self.config.glove_char_file,
            glove_char_size=self.config.glove_char_size,
            glove_char_dim=self.config.glove_char_dim,
            train_examples_file=self.config.train_examples_file,
            train_meta_file=self.config.train_meta_file,
            train_eval_file=self.config.train_eval_file,
            dev_examples_file=self.config.dev_examples_file,
            dev_meta_file=self.config.dev_meta_file,
            dev_eval_file=self.config.dev_eval_file,
            word_emb_file=self.config.word_emb_file,
            word_dict_file=self.config.word_dict_file,
            char_emb_file=self.config.char_emb_file,
            char_dict_file=self.config.char_dict_file,
            context_limit=self.config.context_limit,
            question_limit=self.config.question_limit,
            answer_limit=self.config.answer_limit,
            char_limit=self.config.char_limit,
            debug=self.config.debug,
            debug_num_examples=self.config.debug_num_examples,
        )
        if self.config.train and not self.config.use_processed_data:
            self.dataset.train_process()
        if self.config.evaluate:
            self.dataset.eval_process()

        if self.config.student == "qanet":
            if self.config.train:
                self.qanet_train()
            elif self.config.evaluate:
                self.qanet_evaluate()
            else:
                raise Exception("Invalid operation")
        else:
            raise Exception("Invalid student model")

    def qanet_train(self):
        # Load Dataset
        self.wv_tensor = torch.tensor(
            np.array(object=FileUtil(self.config.word_emb_file).load(),
                     dtype=np.float32),
            dtype=torch.float32
        )
        self.cv_tensor = torch.tensor(
            np.array(object=FileUtil(self.config.char_emb_file).load(),
                     dtype=np.float32),
            dtype=torch.float32
        )
        self.wv_word2idx = FileUtil(self.config.word_dict_file).load()

        self.train_data_loader = SQuAD(
            examples_file=self.config.train_examples_file
        ).get_loader(
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_examples
        )

        self.dev_eval_dict = FileUtil(self.config.dev_eval_file).load()
        self.dev_data_loader = SQuAD(
            examples_file=self.config.dev_examples_file
        ).get_loader(
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_examples
        )

        # QANet Model
        self.student_model = QANet(
            word_mat=self.wv_tensor,
            char_mat=self.cv_tensor,
            c_max_len=self.config.context_limit,
            q_max_len=self.config.question_limit,
            d_model=self.config.qanet_hidden_size,
            train_cemb=(not self.config.use_pretrained_char),
            num_head=self.config.qanet_attention_heads,
            pad=self.wv_word2idx["<PAD>"]
        )
        self.student_model.summary()
        if torch.cuda.device_count() > 1 and self.config.multi_gpu:
            self.student_model = nn.DataParallel(self.student_model)
        self.student_model.to(self.device)

        # Optimizer
        self.student_parameters = filter(lambda p: p.requires_grad,
                                         self.student_model.parameters())
        self.student_optimizer = optim.Adam(
            params=self.student_parameters,
            lr=self.config.lr,
            betas=(self.config.lr_beta1,
                   self.config.lr_beta2),
            eps=1e-8,
            weight_decay=3e-7
        )

        # Scheduler
        cr = 1.0 / math.log(self.config.lr_warm_up_num)
        self.student_scheduler = optim.lr_scheduler.LambdaLR(
            self.student_optimizer,
            lr_lambda=lambda ee: cr * math.log(ee + 1)
            if ee < self.config.lr_warm_up_num else 1)

        # Exponential Moving Average
        self.ema = EMA(self.config.ema_decay)
        if self.config.use_ema:
            for name, param in self.student_model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

        # Loss Function
        self.student_loss = nn.CrossEntropyLoss()

        # Initialize the Student Model Trainer
        trainer = QANetTrainer(
            device=self.device,
            model=self.student_model,
            loss=self.student_loss,
            optimizer=self.student_optimizer,
            use_scheduler=self.config.use_scheduler,
            scheduler=self.student_scheduler,
            use_grad_clip=self.config.use_grad_clip,
            grad_clip=self.config.grad_clip,
            use_ema=self.config.use_ema,
            ema=self.ema,
            use_kd=self.config.use_kd,
            interpolation=self.config.interpolation,
            truncation_max_len=self.config.truncation_max_len,
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            train_data_loader=self.train_data_loader,
            dev_data_loader=self.dev_data_loader,
            dev_eval_dict=self.dev_eval_dict,
            epochs=self.config.epochs,
            save_dir=self.config.save_dir,
            save_freq=self.config.save_freq,
            save_prefix=self.config.save_prefix,
            resume=self.config.resume,
            print_freq=self.config.print_freq,
            use_early_stop=self.config.use_early_stop,
            early_stop=self.config.early_stop,
            debug=self.config.debug,
            debug_num_examples=self.config.debug_num_examples,
        )

        # Train the Student Model
        trainer.train()

    def qanet_evaluate(self):
        # Load Dataset
        self.wv_tensor = torch.tensor(
            np.array(object=FileUtil(self.config.word_emb_file).load(),
                     dtype=np.float32),
            dtype=torch.float32
        )
        self.cv_tensor = torch.tensor(
            np.array(object=FileUtil(self.config.char_emb_file).load(),
                     dtype=np.float32),
            dtype=torch.float32
        )
        self.wv_word2idx = FileUtil(self.config.word_dict_file).load()

        self.dev_eval_dict = FileUtil(self.config.dev_eval_file).load()
        self.dev_data_loader = SQuAD(
            examples_file=self.config.dev_examples_file
        ).get_loader(
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_examples
        )

        # QANet Model
        self.student_model = QANet(
            word_mat=self.wv_tensor,
            char_mat=self.cv_tensor,
            c_max_len=self.config.context_limit,
            q_max_len=self.config.question_limit,
            d_model=self.config.qanet_hidden_size,
            train_cemb=False,
            num_head=self.config.qanet_attention_heads,
            pad=self.wv_word2idx["<PAD>"]
        )
        self.student_model.summary()
        if torch.cuda.device_count() > 1 and self.config.multi_gpu:
            self.student_model = nn.DataParallel(self.student_model)
        self.student_model.to(self.device)

        # Exponential Moving Average
        self.ema = EMA(self.config.ema_decay)
        if self.config.use_ema:
            for name, param in self.student_model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

        # Load Checkpoint
        evaluator = QANetEvaluator(
            device=self.device,
            model=self.student_model,
            use_ema=self.config.use_ema,
            ema=self.ema,
            dev_data_loader=self.dev_data_loader,
            dev_eval_dict=self.dev_eval_dict,
            resume=self.config.resume,
            evaluation_results_file=self.config.evaluation_results_file,
            evaluation_answers_file=self.config.evaluation_answers_file,
            evaluation_dev_eval_dict_file=self.config.evaluation_dev_eval_dict_file,
            evaluation_predictions_file=self.config.evaluation_predictions_file,
        )
        evaluator.evaluate()

    def parse_arguments(self):

        # Working Directories
        data_dir = os.path.join(os.getcwd(), "data")
        data_squad_dir = os.path.join(data_dir, "squad")
        data_glove_dir = os.path.join(data_dir, "glove")
        processed_dir = os.path.join(os.getcwd(), "processed")
        processed_data_dir = os.path.join(processed_dir, "data")
        processed_save_dir = os.path.join(processed_dir, "checkpoints")
        processed_eval_dir = os.path.join(processed_dir, "evaluation")
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(processed_eval_dir, exist_ok=True)

        # Arguments Parser
        parser = argparse.ArgumentParser(
            description="Improving Question Answering Performance \
                Using Knowledge Distillation and Active Learning",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Dataset: SQuAD v1.1
        parser.add_argument(
            "--use_processed_data",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Use already processed dataset"
        )
        parser.add_argument(
            "--train_file",
            type=str,
            default=os.path.relpath(os.path.join(
                data_squad_dir, "train-v1.1.json")),
            help="Path to the train dataset file"
        )
        parser.add_argument(
            "--train_examples_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "train-examples.pkl")),
            help="Path to the train dataset examples file"
        )
        parser.add_argument(
            "--train_meta_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "train-meta.pkl")),
            help="Path to the train dataset meta file"
        )
        parser.add_argument(
            "--train_eval_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "train-eval.pkl")),
            help="Path to the train dataset evaluation file"
        )
        parser.add_argument(
            "--dev_file",
            type=str,
            default=os.path.relpath(os.path.join(
                data_squad_dir, "dev-v1.1.json")),
            help="Path to the dev dataset file"
        )
        parser.add_argument(
            "--dev_examples_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "dev-examples.pkl")),
            help="Path to the dev dataset examples file"
        )
        parser.add_argument(
            "--dev_meta_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "dev-meta.pkl")),
            help="Path to the dev dataset meta file"
        )
        parser.add_argument(
            "--dev_eval_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "dev-eval.pkl")),
            help="Path to the dev dataset evaluation file"
        )

        # Embedding: Word
        parser.add_argument(
            "--glove_word_file",
            type=str,
            default=os.path.relpath(os.path.join(
                data_glove_dir, "glove.840B.300d.txt")),
            help="Path to the word embedding file"
        )
        parser.add_argument(
            "--glove_word_size",
            type=int,
            default=int(2.2e6),
            help="Corpus size for GloVe word embedding"
        )
        parser.add_argument(
            "--glove_word_dim",
            type=int,
            default=300,
            help="Dimension size for GloVe word embedding"
        )
        parser.add_argument(
            "--word_emb_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "word_emb.pkl")),
            help="Path to the word embedding matrix file"
        )
        parser.add_argument(
            "--word_dict_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "word_dict.pkl")),
            help="Path to the word embedding dictionary file"
        )

        # Processed Data
        parser.add_argument(
            "--processed_data_dir",
            type=str,
            default=os.path.relpath(processed_data_dir),
            help="Processed data directory"
        )

        # Embedding: Character
        parser.add_argument(
            "--use_pretrained_char",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Use pretrained character embedding"
        )
        parser.add_argument(
            "--glove_char_file",
            type=str,
            default=os.path.relpath(os.path.join(
                data_glove_dir, "glove.840B.300d-char.txt")),
            help="Path to the GloVe character embedding file"
        )
        parser.add_argument(
            "--glove_char_size",
            type=int,
            default=94,
            help="Corpus size for GloVe character embedding"
        )
        parser.add_argument(
            "--glove_char_dim",
            type=int,
            default=64,
            help="Dimension size for GloVe character embedding"
        )
        parser.add_argument(
            "--char_emb_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "char_emb.pkl")),
            help="Path to the character embedding matrix file"
        )
        parser.add_argument(
            "--char_dict_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_data_dir, "char_dict.pkl")),
            help="Path to the character embedding dictionary file"
        )

        # Student Model
        parser.add_argument(
            "--student",
            type=str,
            default="qanet",
            choices=["qanet"],
            help="Base model to be used as the student model"
        )
        parser.add_argument(
            "--context_limit",
            type=int,
            default=512,
            help="Maximum context token number"
        )
        parser.add_argument(
            "--question_limit",
            type=int,
            default=50,
            help="Maximum question token number"
        )
        parser.add_argument(
            "--answer_limit",
            type=int,
            default=30,
            help="Maximum answer token number"
        )
        parser.add_argument(
            "--char_limit",
            type=int,
            default=16,
            help="Maximum number of characters in a word"
        )
        parser.add_argument(
            "--qanet_hidden_size",
            type=int,
            default=128,
            help="Model hidden size"
        )
        parser.add_argument(
            "--qanet_attention_heads",
            type=int,
            default=8,
            help="Number of attention heads"
        )

        # Teacher Model
        parser.add_argument(
            "--use_kd",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Use knowledge distillation"
        )
        parser.add_argument(
            "--teacher",
            type=str,
            default="bert",
            choices=["bert"],
            help="Model to be used as the teacher model"
        )
        parser.add_argument(
            "--teacher_model_or_path",
            type=str,
            default="bert-large-uncased-whole-word-masking-finetuned-squad",
            help="Teacher model's name or the path to a trained model directory"
        )
        parser.add_argument(
            "--teacher_tokenizer_or_path",
            type=str,
            default="bert-large-uncased-whole-word-masking-finetuned-squad",
            help="Teacher tokenizer's name or the path to a trained model's \
                tokenizer"
        )
        parser.add_argument(
            "--teacher_batch_size",
            type=int,
            default=32,
            help="Teacher model's batch size"
        )
        parser.add_argument(
            "--truncation_max_len",
            type=int,
            default=512,
            help="Maximum number of tokens after which truncation is to occur"
        )
        parser.add_argument(
            "--interpolation",
            type=str,
            default="linear",
            choices=["linear", "cubic", "quadratic"],
            help="Interpolation between the student and the teacher model"
        )
        parser.add_argument(
            "--temperature",
            type=int,
            default=10,
            help="Knowledge distillation temperature value"
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.7,
            help="Knowledge distillation alpha value"
        )

        # Train & Evaluation
        parser.add_argument(
            "--train",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Train the student model"
        )
        parser.add_argument(
            "--evaluate",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Evaluate the student model"
        )
        parser.add_argument(
            "--use_cuda",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Enable CUDA"
        )
        parser.add_argument(
            "--multi_gpu",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Use multiple GPUs if available"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Number of batches"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Number of epochs"
        )
        parser.add_argument(
            "--shuffle_examples",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Shuffle dataset examples"
        )
        parser.add_argument(
            "--use_early_stop",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Early stop the training"
        )
        parser.add_argument(
            "--early_stop",
            type=int,
            default=10,
            help="Number of checkpoints to trigger early stop"
        )
        parser.add_argument(
            "--evaluate_num_batches",
            type=int,
            default=500,
            help="Number of batches for evaluation"
        )
        parser.add_argument(
            "--evaluation_results_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_eval_dir, "eval_results.json")),
            help="Path to the model evaluation results file"
        )
        parser.add_argument(
            "--evaluation_answers_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_eval_dir, "eval_answers.json")),
            help="Path to the model evaluation answers file"
        )
        parser.add_argument(
            "--evaluation_dev_eval_dict_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_eval_dir, "eval_dev_eval_dict.json")),
            help="Path to the model evaluation dev dict file"
        )
        parser.add_argument(
            "--evaluation_predictions_file",
            type=str,
            default=os.path.relpath(os.path.join(
                processed_eval_dir, "eval_predictions.json")),
            help="Path to the model evaluation predictions file"
        )

        # Exponential Moving Average
        parser.add_argument(
            "--use_ema",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Use exponential moving average"
        )
        parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.9999,
            help="Exponential moving average decay"
        )

        # Optimizer
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Optimizer learning rate"
        )
        parser.add_argument(
            "--lr_warm_up_num",
            type=int,
            default=1000,
            help="Number of warm-up steps for learning rate"
        )
        parser.add_argument(
            "--lr_beta1",
            type=float,
            default=0.8,
            help="Learning rate beta 1"
        )
        parser.add_argument(
            "--lr_beta2",
            type=float,
            default=0.999,
            help="Learning rate beta 2"
        )

        # Scheduler
        parser.add_argument(
            "--use_scheduler",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Use learning rate scheduler"
        )

        # Gradient Clipping
        parser.add_argument(
            "--use_grad_clip",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Use gradient clip"
        )
        parser.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient clipping value"
        )

        # Checkpoints
        parser.add_argument(
            "--resume",
            type=str,
            default="",
            help="Path to a checkpoint"
        )
        parser.add_argument(
            "--save_dir",
            type=str,
            default=os.path.relpath(processed_save_dir),
            help="Saved checkpoints directory path"
        )
        parser.add_argument(
            "--save_freq",
            type=int,
            default=1,
            help="Save checkpoint frequency"
        )
        parser.add_argument(
            "--save_prefix",
            type=str,
            default="",
            help="Checkpoint name prefix"
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=10,
            help="Print frequency"
        )

        # Debug
        parser.add_argument(
            "--debug",
            type=lambda v: bool(strtobool(str(v))),
            default=False,
            help="Enable debug mode"
        )
        parser.add_argument(
            "--debug_num_examples",
            type=int,
            default=2,
            help="Number of examples to be processed when debug mode is enabled"
        )

        # Parse Arguments
        args = parser.parse_args()

        # Modify Processed Data Files
        if args.processed_data_dir != os.path.relpath(processed_data_dir):
            params = (
                "train_examples_file",
                "train_meta_file",
                "train_eval_file",
                "dev_examples_file",
                "dev_meta_file",
                "dev_eval_file",
                "word_emb_file",
                "word_dict_file",
                "char_emb_file",
                "char_dict_file",
            )
            for k, v in enumerate(params):
                args.__dict__[v] = os.path.relpath(os.path.join(
                    args.processed_data_dir,
                    os.path.basename(args.__dict__[v])))

        if args.evaluate:
            args.train = False

        return args

    def summary(self):
        print(json.dumps(vars(self.config), sort_keys=False, indent=4))
        return self
