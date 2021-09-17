import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SentimentDataset(Dataset):
    def __init__(self, path, model_path):
        super(SentimentDataset, self).__init__()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_data = open(path).readlines()
        self.label = []
        self.sentence = []
        for line in input_data:
            line = line.strip().split('\t')
            self.label.append(int(line[1]))
            self.sentence.append(tokenizer.encode(line[0]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return torch.tensor(self.sentence[item]), self.label[item]
