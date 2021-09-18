from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


class BertTokenizerX:

    def __init__(self,
                 teacher_tokenizer_or_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            teacher_tokenizer_or_path)

    def to_ascii(self, s, debug: bool = False):
        if not self.is_ascii(s):
            _s = ''.join(
                [v[2:] if v[0:2] == '##' else v for v in self.tokenizer.tokenize(s)])
            if debug:
                print('{:>6} 2En {} -> {}'.format("", s, _s))
            return _s
        return s

    def is_ascii(self, s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

class BertX:

    def __init__(self,
                 device,
                 teacher_model_or_path):
        self.device = device
        self.model = BertForQuestionAnswering.from_pretrained(
            teacher_model_or_path)
        self.model.to(self.device)
