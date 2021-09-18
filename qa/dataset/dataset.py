import os
import re
import json
import random
import psutil
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from qa.util.file import FileUtil


class Dataset:

    def __init__(self,
                 use_cuda,
                 device,
                 student_tokenizer,
                 use_kd,
                 teacher_tokenizer,
                 teacher_model,
                 teacher_batch_size,
                 truncation_max_len,
                 train_file,
                 dev_file,
                 glove_word_file,
                 glove_word_size,
                 glove_word_dim,
                 use_pretrained_char,
                 glove_char_file,
                 glove_char_size,
                 glove_char_dim,
                 train_examples_file,
                 train_meta_file,
                 train_eval_file,
                 dev_examples_file,
                 dev_meta_file,
                 dev_eval_file,
                 word_emb_file,
                 word_dict_file,
                 char_emb_file,
                 char_dict_file,
                 context_limit,
                 question_limit,
                 answer_limit,
                 char_limit,
                 debug=False,
                 debug_num_examples=2,
                 ):
        self.use_cuda = use_cuda
        self.device = device

        # Student
        self.student_tokenizer = student_tokenizer

        # Teacher
        self.use_kd = use_kd
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_batch_size = teacher_batch_size
        self.truncation_max_len = truncation_max_len

        # Limitations
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.answer_limit = answer_limit
        self.char_limit = char_limit

        # Input
        self.train_file = train_file
        self.dev_file = dev_file
        self.glove_word_file = glove_word_file
        self.glove_word_size = glove_word_size
        self.glove_word_dim = glove_word_dim
        self.use_pretrained_char = use_pretrained_char
        self.glove_char_file = glove_char_file
        self.glove_char_size = glove_char_size
        self.glove_char_dim = glove_char_dim

        # Output
        self.train_examples_file = train_examples_file
        self.train_meta_file = train_meta_file
        self.train_eval_file = train_eval_file
        self.dev_examples_file = dev_examples_file
        self.dev_meta_file = dev_meta_file
        self.dev_eval_file = dev_eval_file
        self.word_emb_file = word_emb_file
        self.word_dict_file = word_dict_file
        self.char_emb_file = char_emb_file
        self.char_dict_file = char_dict_file

        # Debug
        self.debug = debug
        self.debug_num_examples = debug_num_examples

    def train_process(self):
        word_counter = Counter()
        char_counter = Counter()

        train_examples, train_meta, train_eval = self.get_examples(
            filename=self.train_file,
            word_counter=word_counter,
            char_counter=char_counter,
            description="train")

        dev_examples, dev_meta, dev_eval = self.get_examples(
            filename=self.dev_file,
            word_counter=word_counter,
            char_counter=char_counter,
            description="dev")

        if self.use_kd:
            (train_start_scores,  # teacher-student aligned start scores
             train_end_scores,  # teacher-student aligned end scores
             train_all_start_scores,  # teacher's all context start scores
             train_all_end_scores  # teacher's all context end scores
             ) = self.get_teacher_scores(train_examples, description="train")

            (dev_start_scores,
             dev_end_scores,
             dev_all_start_scores,
             dev_all_end_scores
             ) = self.get_teacher_scores(dev_examples, description="dev")

            for i, v in enumerate(train_examples):
                train_examples[i]["teacher_start_scores"] = train_start_scores[i]
                train_examples[i]["teacher_end_scores"] = train_end_scores[i]
                train_examples[i]["teacher_all_start_scores"] = train_all_start_scores[i]
                train_examples[i]["teacher_all_end_scores"] = train_all_end_scores[i]

            for i, v in enumerate(dev_examples):
                dev_examples[i]["teacher_start_scores"] = dev_start_scores[i]
                dev_examples[i]["teacher_end_scores"] = dev_end_scores[i]
                dev_examples[i]["teacher_all_start_scores"] = dev_all_start_scores[i]
                dev_examples[i]["teacher_all_end_scores"] = dev_all_end_scores[i]

        word_emb_file = self.glove_word_file
        word_emb_size = self.glove_word_size
        word_emb_dim = self.glove_word_dim
        char_emb_file = self.glove_char_file if self.use_pretrained_char else None
        char_emb_size = self.glove_char_size if self.use_pretrained_char else None
        char_emb_dim = self.glove_word_dim if self.use_pretrained_char else self.glove_char_dim

        # Words
        word_emb_mat, word2idx_dict = self.get_embedding(
            counter=word_counter,
            emb_file=word_emb_file,
            emb_size=word_emb_size,
            emb_dim=word_emb_dim,
            description="word"
        )

        # Characters
        char_emb_mat, char2idx_dict = self.get_embedding(
            counter=char_counter,
            emb_file=char_emb_file,
            emb_size=char_emb_size,
            emb_dim=char_emb_dim,
            description="char"
        )

        train_examples, train_meta = self.build_features(
            examples=train_examples,
            meta=train_meta,
            word2idx_dict=word2idx_dict,
            char2idx_dict=char2idx_dict,
            description="train"
        )
        dev_examples, dev_meta = self.build_features(
            examples=dev_examples,
            meta=dev_meta,
            word2idx_dict=word2idx_dict,
            char2idx_dict=char2idx_dict,
            description="dev"
        )

        FileUtil(self.word_emb_file).save(word_emb_mat)
        FileUtil(self.char_emb_file).save(char_emb_mat)
        FileUtil(self.word_dict_file).save(word2idx_dict)
        FileUtil(self.char_dict_file).save(char2idx_dict)
        FileUtil(self.train_examples_file).save(train_examples)
        FileUtil(self.train_meta_file).save(train_meta)
        FileUtil(self.train_eval_file).save(train_eval)
        FileUtil(self.dev_examples_file).save(dev_examples)
        FileUtil(self.dev_meta_file).save(dev_meta)
        FileUtil(self.dev_eval_file).save(dev_eval)

    def eval_process(self):
        word_counter = Counter()
        char_counter = Counter()

        dev_examples, dev_meta, dev_eval = self.get_examples(
            filename=self.dev_file,
            word_counter=word_counter,
            char_counter=char_counter,
            description="dev")

        for i, v in enumerate(dev_examples):
            dev_examples[i]["teacher_start_scores"] = []
            dev_examples[i]["teacher_end_scores"] = []
            dev_examples[i]["teacher_all_start_scores"] = []
            dev_examples[i]["teacher_all_end_scores"] = []

        word2idx_dict = FileUtil(self.word_dict_file).load()
        char2idx_dict = FileUtil(self.char_dict_file).load()

        dev_examples, dev_meta = self.build_features(
            examples=dev_examples,
            meta=dev_meta,
            word2idx_dict=word2idx_dict,
            char2idx_dict=char2idx_dict,
            description="dev"
        )

        FileUtil(self.dev_examples_file).save(dev_examples)
        FileUtil(self.dev_meta_file).save(dev_meta)
        FileUtil(self.dev_eval_file).save(dev_eval)

    def get_examples(self,
                     filename: str,
                     word_counter: Counter,
                     char_counter: Counter,
                     description: str = None):
        if description:
            print("Getting examples: {}".format(description))

        examples = []
        meta = {}
        eval_examples = {}

        with open(filename, "r") as f:
            source = json.load(f)
            meta["version"] = version = source["version"]
            meta["num_q"] = 0
            meta["num_q_answerable"] = 0
            meta["num_qa_answerable"] = 0
            meta["num_q_noanswer"] = 0

            for article in tqdm(source["data"]):
                for para in article["paragraphs"]:

                    # Tokenize Context
                    context = self.student_tokenizer.prepare(para["context"])
                    context_tokens = self.student_tokenizer.tokenize(context)
                    context_chars = [list(token) for token in context_tokens]
                    spans = self.student_tokenizer.to_index(
                        context, context_tokens)
                    for token in context_tokens:
                        word_counter[token] += len(para["qas"])
                        for char in token:
                            char_counter[char] += len(para["qas"])

                    for qa in para["qas"]:
                        meta["num_q"] += 1

                        # Tokenize Question
                        question = self.student_tokenizer.prepare(
                            qa["question"])
                        question_tokens = self.student_tokenizer.tokenize(
                            question)
                        question_chars = [list(token)
                                          for token in question_tokens]
                        for token in question_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1

                        y1s, y2s = [], []
                        answer_texts = []
                        answers = qa["answers"]
                        answerable = 1
                        if version == "v2.0" and qa["is_impossible"] is True:
                            answers = qa["plausible_answers"]
                            answerable = 0
                        meta["num_q_answerable"] += answerable
                        if len(answers) == 0:
                            meta["num_q_noanswer"] += 1
                            continue

                        for answer in answers:
                            answer_text = answer["text"]
                            answer_start = answer['answer_start']
                            answer_end = answer_start + len(answer_text)
                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or
                                        answer_start >= span[1]):
                                    answer_span.append(idx)
                            y1, y2 = answer_span[0], answer_span[-1]
                            y1s.append(y1)
                            y2s.append(y2)
                            meta["num_qa_answerable"] += answerable

                        example = {
                            "context_tokens": context_tokens,
                            "context_chars": context_chars,
                            "question_tokens": question_tokens,
                            "question_chars": question_chars,
                            "y1s": y1s,
                            "y2s": y2s,
                            "id": meta["num_q"],
                            "context": context,
                            "spans": spans,
                            "question": question,
                            "answers": answer_texts,
                            "answerable": answerable,
                            "uuid": qa["id"]
                        }
                        examples.append(example)

                        eval_examples[str(meta["num_q"])] = {
                            "context": context,
                            "spans": spans,
                            "question": question,
                            "answers": answer_texts,
                            "answerable": answerable,
                            "uuid": qa["id"]
                        }

                        if (self.debug and meta["num_q"]
                                >= self.debug_num_examples):
                            return examples, meta, eval_examples

            random.shuffle(examples)

        return examples, meta, eval_examples

    def get_teacher_scores(self,
                           examples,
                           description: str = None,
                           debug: bool = False):
        if description:
            print("Getting teacher scores: {}".format(description))

        all_sep_index = []
        all_last_sep_index = []
        all_token_ids = []
        all_segment_ids = []
        all_student_tokens = []

        for i, v in tqdm(enumerate(examples), total=len(examples)):
            context = examples[i]["context"]
            question = examples[i]["question"]
            # answers = examples[i]["answers"] # labels
            student_tokens = examples[i]["context_tokens"]
            all_student_tokens.append(student_tokens)

            # Token indices sequence length is longer than the specified
            # maximum sequence length for this model (585 > 512). Running this
            # sequence through the model will result in indexing errors
            # 115 samples of total 87599 have been truncated.
            token_ids = self.teacher_tokenizer.tokenizer.encode(
                question,
                context,
                max_length=self.truncation_max_len,
                truncation=True)

            sep_index = token_ids.index(
                self.teacher_tokenizer.tokenizer.sep_token_id)
            all_sep_index.append(sep_index)

            last_sep_index = (token_ids.index(
                self.teacher_tokenizer.tokenizer.sep_token_id, sep_index+1)
                if self.teacher_tokenizer.tokenizer.sep_token_id in token_ids[sep_index+1:]
                else -1)
            all_last_sep_index.append(last_sep_index)

            num_seg_question = sep_index + 1
            num_seg_context = len(token_ids) - num_seg_question
            segment_ids = [0] * num_seg_question + [1] * num_seg_context

            assert len(token_ids) == len(segment_ids)

            all_token_ids.append(token_ids)
            all_segment_ids.append(segment_ids)

        # Padding teacher tokens
        all_token_ids_len = []
        for i, token_ids in enumerate(all_token_ids):
            all_token_ids_len.append(len(token_ids))
        max_length = max(all_token_ids_len)
        for i, v in tqdm(enumerate(all_token_ids), total=len(all_token_ids)):
            while len(all_token_ids[i]) < max_length:
                all_token_ids[i].append(0)  # default token id padding
                all_segment_ids[i].append(0)  # default segment id padding

        # Get teacher start/end scores
        all_start_scores = []  # teacher start scores
        all_end_scores = []  # teacher end scores
        all_tokens = []  # teacher tokens
        all_answers = []  # teacher answers

        last_i = 0
        t = tqdm(enumerate(all_token_ids), total=len(
            all_token_ids), desc=self.get_memory_stat())
        for i, v in t:
            # Compute when teacher batch size is reached
            if (i+1) % self.teacher_batch_size != 0 and i != len(all_token_ids) - 1:
                continue

            t.set_description(self.get_memory_stat())

            # Extract batch examples from `last_i` included to `i+1` excluded.
            # Next batch examples' start index will be `last_i=i+1`.
            batch_token_ids = all_token_ids[last_i:i+1]
            batch_segment_ids = all_segment_ids[last_i:i+1]

            bs = len(batch_token_ids)
            if (bs == 0):
                continue

            batch_token_ids_ts = torch.tensor(batch_token_ids).to(self.device)
            batch_segment_ids_ts = torch.tensor(
                batch_segment_ids).to(self.device)

            # Tensor size is (teacher_batch_size, max_length)
            with torch.no_grad():
                output = self.teacher_model.model(
                    batch_token_ids_ts,
                    token_type_ids=batch_segment_ids_ts)
                batch_start_scores_ts = output.start_logits
                batch_end_scores_ts = output.end_logits

            for j in range(0, bs):
                # sep_index holds the first [SEP] index and last_sep_index holds
                # the last [SEP] index. They are used to extract scores and tokens
                # of the context. That's because output of the BERT model tokens
                # is as follows. BERT model only supports maximum 512 tokens; so
                # there might be a truncation cutting the middle of the context.
                # [CLS]<question>[SEP]<context>[SEP][PAD]...[PAD]
                sep_index = all_sep_index[last_i+j]
                last_sep_index = all_last_sep_index[last_i+j]
                last_sep_index = last_sep_index if last_sep_index != - \
                    1 else len(batch_token_ids[j])

                all_start_scores.append(
                    batch_start_scores_ts[j][sep_index+1:last_sep_index].tolist())
                all_end_scores.append(
                    batch_end_scores_ts[j][sep_index+1:last_sep_index].tolist())

                start_score_ts, start_index_ts = torch.max(
                    batch_start_scores_ts[j][sep_index+1:last_sep_index], 0)
                end_score_ts, end_index_ts = torch.max(
                    batch_end_scores_ts[j][sep_index+1:last_sep_index], 0)

                tokens = self.teacher_tokenizer.tokenizer.convert_ids_to_tokens(
                    batch_token_ids[j][sep_index+1:last_sep_index])
                all_tokens.append(tokens)

                answer = tokens[start_index_ts.item()]
                for k in range(start_index_ts.item() + 1, end_index_ts.item() + 1):
                    if tokens[k][0:2] == '##':
                        answer += tokens[k][2:]
                    else:
                        answer += ' ' + tokens[k]
                all_answers.append(answer)

            # Next batch examples' start index
            last_i = i+1

        # Extract BERT start/end scores based on the student's tokens.
        # all_start_scores -> teacher model's start scores
        # all_end_scores -> teacher model's end scores
        # all_tokens -> teacher model's context tokens
        # all_answers -> teacher model's answers based on max start/end scores
        def_val = 0
        all_extracted_start_scores = []
        all_extracted_end_scores = []

        for i, d0 in tqdm(enumerate(all_student_tokens), total=len(all_student_tokens)):
            start_scores = [def_val] * len(all_student_tokens[i])
            end_scores = [def_val] * len(all_student_tokens[i])
            _j = 0  # skips student tokens
            _k = 0  # skips teacher tokens
            for j, d1 in enumerate(all_student_tokens[i]):
                if _j > 0:
                    _j -= 1
                    continue
                student_token = all_student_tokens[i][j].lower()
                # next_student_token = all_student_tokens[i][j+1].lower() if j+1 < len(all_student_tokens[i]) else None

                # j iterates over student tokens until _k reaches all_tokens[i]
                # which is a way of implementing trunc_max_len limitation. Do not
                # use trunc_max_len directly to limit j progress because we extract
                # <context> section of BERT model input, so <context> might be
                # available inside all_tokens[i] completely or truncated from the
                # right side. Here we use all_tokens[i] length to put scores at
                # the beginning of the start_scores and end_scores. The rest of
                # the are def_val.
                for k in range(_k, len(all_tokens[i])):
                    teacher_token = all_tokens[i][k]
                    # next_teacher_token = all_tokens[i][k+1] if k+1 < len(all_tokens[i]) else None

                    if teacher_token[0:2] == '##':
                        teacher_token = teacher_token[2:]

                    # Eliminate non-ASCII characters
                    student_token = self.teacher_tokenizer.to_ascii(
                        student_token, debug=debug).lower()
                    teacher_token = teacher_token.lower()

                    # Skip current student token when it is a white spaces or an
                    # escaped character
                    # Student tokenizer preserves all white spaces while teacher
                    # tokenizer eliminates them.
                    # student       teacher
                    # hello         hello
                    # \n            world
                    # world
                    # ---------------------
                    # hello         hello
                    # \t\n          world
                    # world
                    # ---------------------
                    # \u3000
                    if len(white_spaces := re.findall(r'[\s]+', student_token)) > 0:
                        if debug:
                            print('{:>6} WSP {:20}{:20}\twhite_spaces={}'.format(
                                i, student_token, teacher_token, white_spaces))
                        break

                    # Student tokens: ['◌', 'ʰ⟩']
                    # Teacher tokens: ['[UNK]', '⟩']
                    # ------------------------------
                    # Student tokens: ['◌', '˭⟩'] -> teacher_tokenizer -> ['[unk]⟩', '⟩']
                    # Teacher tokens: ['[UNK]', '⟩']
                    # 14400:14500
                    if k-1 >= 0 and j-1 >= 0 and ord(all_student_tokens[i][j-1][-1]) == 9676 and all_tokens[i][k-1].lower() == '[unk]':
                        if debug:
                            print('{:>6} UN1 {:20}{:20}'.format(
                                i, student_token, teacher_token))
                        if student_token.find("[unk]") == 0:
                            student_token = student_token[len("[unk]"):]
                        else:
                            student_token = student_token[1:]
                        if debug:
                            print('{:>6} UN2 {:20}{:20}'.format(
                                i, student_token, teacher_token))

                    # Teacher token and student token are equal
                    #
                    # student       teacher
                    # indicate      indicate
                    if student_token == teacher_token:
                        start_scores[j] = all_start_scores[i][k]
                        end_scores[j] = all_end_scores[i][k]
                        _k = k + 1
                        if debug:
                            print('{:>6}     {:20}{:20}'.format(
                                i, student_token, teacher_token))
                        break

                    # student       teacher
                    # can           cannot      teacher contains student
                    # not
                    # ----------------------
                    # was           wasn        teacher contains student
                    # n't           '
                    #               t
                    # ----------------------
                    # £             £1          teacher contains student
                    # 1.05          .
                    # in            05
                    #               in
                    # ----------------------
                    # sir           sir
                    # jonathan      jonathan
                    # i             iv          teacher contains student
                    # ve            ##e
                    # ----------------------
                    # Accomodation  acc         student constains teacher
                    #               ##omo
                    #               ##dation
                    elif (ts := (teacher_token.find(student_token) == 0)) or (st := (student_token.find(teacher_token) == 0)):
                        start_scores[j] = all_start_scores[i][k]
                        end_scores[j] = all_end_scores[i][k]

                        def find_skip_steps():
                            # m = 10 # DO NOT PUT LIMIT HERE
                            s_len = len(all_student_tokens[i])
                            # if j+m < s_len: s_len = j+m
                            t_len = len(all_tokens[i])
                            # if k+m < t_len: t_len = k+m
                            s_str = ""
                            t_start_index = -1
                            for s in range(j, s_len):
                                s_str += self.teacher_tokenizer.to_ascii(
                                    all_student_tokens[i][s]).lower()
                                t_str = ""
                                for t in range(k, t_len):
                                    t_str += all_tokens[i][t][2:].lower(
                                    ) if all_tokens[i][t][0:2] == '##' else all_tokens[i][t].lower()
                                    if debug:
                                        print("{:>6}     {} == {}".format(
                                            i, s_str, t_str))
                                    if s_str == t_str:
                                        return s-j, t-k, s_str, t_str
                                    # Skip to the next student tokens batch because
                                    # teacher tokens batch is of a trucated kind
                                    # and is reached the end.
                                    # e.g. daphnisinasongcontest...writers. == da
                                    # 20500:20520
                                    # 82500:82550
                                    t_start_index = s_str.find(t_str)

                            if t_start_index == 0:
                                if debug:
                                    print("{:>6} JMP {} == {}".format(
                                        i, s_str, t_str))
                                return s_len-1-j, t_len-1-k, s_str, t_str

                            raise Exception(
                                "Failed to find skip steps s and t.")

                        s_step, t_step, s_str, t_str = find_skip_steps()
                        _j = s_step
                        _k = k + 1 + t_step

                        lbl = ""
                        if ts:
                            lbl = "T_s"
                        elif st:
                            lbl = "S_t"
                        if debug:
                            print('{:>6} {} {:20}{:20}\ts_step={}\tt_step={}'.format(
                                i, lbl, student_token, teacher_token, s_step, t_step))

                        break

                    # Otherwise
                    else:
                        if debug:
                            print('{:>6} ERR {:20}{:20}'.format(
                                i, student_token, teacher_token))
                        # TODO You can skip `i` if there is an error
                        raise Exception()

            all_extracted_start_scores.append(start_scores)
            all_extracted_end_scores.append(end_scores)

        assert len(all_extracted_start_scores) == len(all_extracted_end_scores)
        assert len(all_start_scores) == len(all_end_scores)
        assert len(examples) == len(all_extracted_start_scores)
        assert len(examples) == len(all_extracted_end_scores)
        assert len(examples) == len(all_start_scores)
        assert len(examples) == len(all_end_scores)

        return all_extracted_start_scores, all_extracted_end_scores, all_start_scores, all_end_scores

    def get_embedding(self,
                      counter: Counter,
                      emb_file: str = None,
                      emb_size: int = None,
                      emb_dim: int = None,
                      limit: int = -1,
                      specials=["<PAD>", "<OOV>", "<SOS>", "<EOS>"],
                      description: str = None):
        if description:
            print("Getting embedding: {}".format(description))

        embedding_dict = {}
        filtered_elements = [k for k, v in counter.items() if v > limit]
        if emb_file is not None:
            assert emb_size is not None
            assert emb_dim is not None
            with open(emb_file, "r", encoding="utf-8") as fh:
                for line in tqdm(fh, total=emb_size):
                    array = line.split()
                    word = "".join(array[0:-emb_dim])
                    vector = list(map(float, array[-emb_dim:]))
                    if word in counter and counter[word] > limit:
                        embedding_dict[word] = vector
        else:
            assert emb_dim is not None
            for token in filtered_elements:
                embedding_dict[token] = [
                    np.random.normal(scale=0.1) for _ in range(emb_dim)]

        token2idx_dict = {token: idx
                          for idx, token
                          in enumerate(embedding_dict.keys(), len(specials))}
        for i in range(len(specials)):
            token2idx_dict[specials[i]] = i
            embedding_dict[specials[i]] = [0. for _ in range(emb_dim)]
        idx2emb_dict = {idx: embedding_dict[token]
                        for token, idx in token2idx_dict.items()}
        emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
        return emb_mat, token2idx_dict

    def word2wid(self, word, word2idx_dict):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return word2idx_dict["<OOV>"]

    def char2cid(self, char, char2idx_dict):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return char2idx_dict["<OOV>"]

    def filter_func(self, example):
        return (len(example["context_tokens"]) > self.context_limit or
                len(example["question_tokens"]) > self.question_limit or
                (example["y2s"][0] - example["y1s"][0]) > self.answer_limit)

    def build_features(self,
                       examples,
                       meta,
                       word2idx_dict,
                       char2idx_dict,
                       description: str = None):
        if description:
            print("Building features: {}".format(description))

        total = 0
        total_ = 0
        examples_with_features = []
        for example in tqdm(examples):
            total_ += 1
            if self.filter_func(example):
                continue
            total += 1

            context_wids = np.ones(
                [self.context_limit], dtype=np.int32) * word2idx_dict["<PAD>"]
            context_cids = np.ones(
                [self.context_limit, self.char_limit], dtype=np.int32) * char2idx_dict["<PAD>"]
            question_wids = np.ones(
                [self.question_limit], dtype=np.int32) * word2idx_dict["<PAD>"]
            question_cids = np.ones(
                [self.question_limit, self.char_limit], dtype=np.int32) * char2idx_dict["<PAD>"]
            y1 = np.zeros([self.context_limit], dtype=np.float32)
            y2 = np.zeros([self.context_limit], dtype=np.float32)

            for i, token in enumerate(example["context_tokens"]):
                context_wids[i] = self.word2wid(token, word2idx_dict)

            for i, token in enumerate(example["question_tokens"]):
                question_wids[i] = self.word2wid(token, word2idx_dict)

            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == self.char_limit:
                        break
                    context_cids[i, j] = self.char2cid(char, char2idx_dict)

            for i, token in enumerate(example["question_chars"]):
                for j, char in enumerate(token):
                    if j == self.char_limit:
                        break
                    question_cids[i, j] = self.char2cid(char, char2idx_dict)

            start, end = example["y1s"][-1], example["y2s"][-1]
            y1[start], y2[end] = 1.0, 1.0

            example["context_wids"] = context_wids
            example["context_cids"] = context_cids
            example["question_wids"] = question_wids
            example["question_cids"] = question_cids
            example["y1"] = start
            example["y2"] = end

            # Free loading memory
            example["spans"] = None
            example["context_tokens"] = None
            example["context_chars"] = None
            example["question_tokens"] = None
            example["question_chars"] = None
            examples_with_features.append(example)

        meta["num_q_filtered"] = total
        return examples_with_features, meta

    def get_memory_stat(self):
        """
        Get memory statistics
        @static
        """
        ram = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        if self.use_cuda:
            vram = torch.cuda.max_memory_allocated('cuda') / 1024 ** 2
            return "VRAM:{:.0f}|RAM:{:.0f}".format(vram, ram)
        else:
            return "RAM:{:.0f}".format(ram)
