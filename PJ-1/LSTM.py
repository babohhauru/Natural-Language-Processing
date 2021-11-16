import torch
import torch.nn as nn
import numpy as np


def init_embedding(input_embedding, seed=1337):
    """initiate weights in embedding layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear(input_linear, seed=1337):
    """initiate weights in linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)
    # nn.init.uniform(input_linear.bias, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_labels=2, dropout=0.5, num_layers=1):
        super(LSTMModel, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)

        self.out = nn.Linear(hidden_size * 2, num_labels)

        init_linear(self.out)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: batch_size x seq_len
        :param y: batch_size
        :return:
            loss: scale
            pred: batch_size
        """
        batch_size, seq_len = x.size()
        x = self.dropout(self.embed(x))
        rnn_out, _ = self.lstm(x)
        rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_size)
        rnn_out = torch.cat([rnn_out[:, -1, 0, :], rnn_out[:, 0, 1, :]], dim=-1)
        # rnn_out = rnn_out[:, -1, :]  # take the last hidden
        rnn_out = self.dropout(rnn_out)
        logits = self.out(rnn_out)
        # loss = self.loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        # pred = torch.argmax(logits, dim=-1)

        return logits
