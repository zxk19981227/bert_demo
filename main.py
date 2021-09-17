import argparse
import random

import numpy as np
import torch
from numpy import mean
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SentimentDataset
from model import Bert


def collate_fn(batch):
    ids, labels = [], []
    for id, label in batch:
        ids.append(id)
        labels.append(label)
    return pad_sequence(ids, batch_first=True), torch.tensor(labels)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(epoch, model: Bert, optim: torch.optim, loader: torch.utils.data.DataLoader, method: str, device: str):
    losses = []
    correct_num = 0
    total_train = 0
    for sentences, label in tqdm(loader):
        predictions = model(sentences, device)
        label = label.view(-1).to(device)
        loss = cross_entropy(predictions, label.view(-1))
        if method == 'train':
            loss.backward()
            optim.step()
            optim.zero_grad()
        predict = torch.argmax(predictions, -1)
        correct_num += (predict == label).float().sum().item()
        total_train += predict.shape[0]
        losses.append(loss.item())
    mean_loss = mean(losses)
    accuracy = correct_num / total_train
    print(f"\nepoch {epoch} {method} loss f{mean_loss:.2f} accuracy:{accuracy * 100:.2f}%")
    return mean_loss, accuracy


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_path", default='./pytorch_model', help='the path where pretrained model is saved')
    arg_parser.add_argument('--lr', default=1e-3, type=float,
                            help='the learning rate use to train the model ,default is 1e-3')
    arg_parser.add_argument('--device', default='cpu', help='the device used to run the programme, default is cpu')
    arg_parser.add_argument('--epoch_num', default=5, help='how many times that the dataset need to run')
    args = arg_parser.parse_args()
    setup_seed(1)
    batch_size = 8
    max_epoch = args.epoch_num
    # set random seed to make the result could be reimplemented
    train_dataloader = DataLoader(SentimentDataset('./train.tsv', args.model_path), batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(SentimentDataset('./valid.tsv', args.model_path), batch_size=batch_size,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(SentimentDataset('./test.tsv', args.model_path), batch_size=batch_size,
                                 collate_fn=collate_fn)
    device = args.device
    model = Bert(args.model_path).to(device)
    best_valid_accuracy = 0
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for i in range(max_epoch):
        model.train()
        run(i, model, optim, train_dataloader, 'train', device)
        model.eval()
        # close the random dropout to make evaluation stable
        with torch.no_grad():
            loss, accuracy = run(i, model, optim, valid_dataloader, 'valid', device)
            if accuracy > best_valid_accuracy:
                best_valid_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model.pkl')
    model.load_state_dict(torch.load('best_model.pkl'))
    with torch.no_grad():
        run(0, model, optim, test_dataloader, 'test', device)


if __name__ == "__main__":
    main()
