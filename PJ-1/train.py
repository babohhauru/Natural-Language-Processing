import torch
from torch.optim import Adam
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from dataloader import SSTDataSet
from LSTM import LSTMModel
from utils import custom_collate
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score
import torch.nn as nn


def train():
    # Step 1: Loading data
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
    # Step 4: Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    sampler = RandomSampler(data_source=train_set)
    train_data_iter = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4, collate_fn=custom_collate)
    loss_fct = nn.CrossEntropyLoss()
    num_epochs = 50
    tr_loss = 0.0
    global_step = 0
    eval_every = 100
    for i_epoch in trange(num_epochs, desc="Epoch"):
        epoch_iterator = tqdm(train_data_iter, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fct(logits.view(-1, 2), y.view(-1))
            pred = torch.argmax(logits, dim=-1)
            # loss, pred = model(x.to(device), y.to(device))
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % eval_every == 0:
                # evaluate
                total_pred, total_target = [], []
                model.eval()
                sampler = SequentialSampler(data_source=dev_set)
                dev_data_iter = DataLoader(dev_set, batch_size=32, sampler=sampler, num_workers=4,
                                           collate_fn=custom_collate)
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
                print("step: {} | loss: {} | acc: {}".format(global_step, tr_loss / eval_every, acc))
                tr_loss = 0
    torch.save(model.state_dict(), 'model/wordavg-model.pt')


def run():
    train()


if __name__ == '__main__':
    run()
