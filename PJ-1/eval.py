import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from dataloader import SSTDataSet
from LSTM import LSTMModel
from utils import custom_collate
from sklearn.metrics import accuracy_score
import torch.nn as nn


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


# 评估函数
def evaluate(rnn, iterator, criteon):
    avg_loss = []
    avg_acc = []
    rnn.eval()  # 进入测试模式

    with torch.no_grad():
        for batch in iterator:
            pred = rnn(batch.text).squeeze()  # [batch, 1] -> [batch]

            loss = criteon(pred, batch.label)
            acc = binary_acc(pred, batch.label).item()

            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc


if __name__ == '__main__':

    train_set = SSTDataSet('data/train.tsv')
    dev_set = SSTDataSet('data/dev.tsv')

    # Step 2: Building vocab
    vocab = {'<pad>': 0, '<unk>': 1}
    for ins in train_set:
        for word in ins['raw_words']:
            if word not in vocab:
                vocab[word] = len(vocab)

    for ins in dev_set:
        for word in ins['raw_words']:
            if word not in vocab:
                vocab[word] = len(vocab)

    train_set.convert_word_to_ids(vocab)
    dev_set.convert_word_to_ids(vocab)

    # Step 3: Building model
    model = LSTMModel(vocab_size=len(vocab), embed_dim=100, hidden_size=200, num_labels=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('weight/wordavg-model.pt'))

    model.to(device)

    sampler = SequentialSampler(data_source=dev_set)
    dev_data_iter = DataLoader(dev_set, batch_size=32, sampler=sampler, num_workers=4,
                               collate_fn=custom_collate)
    loss_fct = nn.CrossEntropyLoss()
    total_pred, total_target = [], []
    model.eval()
    for dev_batch in dev_data_iter:
        x, y = dev_batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss = loss_fct(logits.view(-1, 2), y.view(-1))
            pred = torch.argmax(logits, dim=-1)
            pred = pred.detach().cpu().numpy().tolist()
            target = y.to('cpu').numpy().tolist()
            total_pred.extend(pred)
            total_target.extend(target)
    acc = accuracy_score(total_target, total_pred)
    print("acc: {}".format(acc))
