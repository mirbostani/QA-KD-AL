import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..util.file import FileUtil


class SQuAD(Dataset):

    def __init__(self, examples_file):
        self.examples = FileUtil(examples_file).load()
        self.num = len(self.examples)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return (self.examples[idx]["context_wids"],
                self.examples[idx]["context_cids"],
                self.examples[idx]["question_wids"],
                self.examples[idx]["question_cids"],
                self.examples[idx]["y1"],
                self.examples[idx]["y2"],
                self.examples[idx]["y1s"],
                self.examples[idx]["y2s"],
                self.examples[idx]["id"],
                (self.examples[idx]["teacher_start_scores"]
                 if "teacher_start_scores" in self.examples[idx] else []),
                (self.examples[idx]["teacher_end_scores"]
                 if "teacher_end_scores" in self.examples[idx] else []),
                (self.examples[idx]["teacher_all_start_scores"]
                 if 'teacher_all_start_scores' in self.examples[idx] else []),
                (self.examples[idx]["teacher_all_end_scores"]
                 if 'teacher_all_end_scores' in self.examples[idx] else []),
                self.examples[idx]["answerable"])

    def collate(self, data):
        (Cwid,
         Ccid,
         Qwid,
         Qcid,
         y1,
         y2,
         y1s,
         y2s,
         id,
         tss,
         tes,
         tass,
         taes,
         answerable) = zip(*data)

        Cwid = torch.tensor(Cwid).long()
        Ccid = torch.tensor(Ccid).long()
        Qwid = torch.tensor(Qwid).long()
        Qcid = torch.tensor(Qcid).long()
        y1 = torch.from_numpy(np.array(y1)).long()
        y2 = torch.from_numpy(np.array(y2)).long()
        id = torch.from_numpy(np.array(id)).long()
        answerable = torch.tensor(answerable).long()

        return (Cwid,
                Ccid,
                Qwid,
                Qcid,
                y1,
                y2,
                y1s,
                y2s,
                id,
                tss,
                tes,
                tass,
                taes,
                answerable)

    def get_loader(self,
                   batch_size: int = 8,
                   shuffle: bool = True,
                   num_workers: int = 0):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate
        )
