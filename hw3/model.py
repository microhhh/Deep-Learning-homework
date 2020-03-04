import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import math
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from LSTM_LN import LayerNormLSTM,MultiLayerLSTM

class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninput, nhid, nlayers, tie_weights=False, dropout=0, layer_norm=False):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, ninput)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninput, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninput, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.rnn_type = rnn_type

        self.decoder = nn.Linear(nhid, nvoc)

        if tie_weights:
            if nhid != ninput:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def forward(self, input, h):
        embeddings = self.drop(self.encoder(input))
        output, hidden = self.rnn(embeddings, h)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


class Attention(nn.Module):
    def __init__(self, nhid):
        super(Attention, self).__init__()
        self.nhid = nhid
        self.linear_hid = nn.Linear(nhid, nhid)
        self.linear_input = nn.Linear(nhid, nhid)
        self.w = nn.Parameter(torch.randn(1, 1, nhid))
        stdv = 1. / math.sqrt(self.w.size(2))
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, input, hid):
        """
        :param input: [batch, nhid]
        :param hid: [timestep, batch, nhid]
        :return: energy: timestep, batch, 1
        """
        # print(input.size(), hid.size())
        timestep = hid.size(0)
        batch = hid.size(1)
        feature = hid.size(2)
        hid = self.linear_hid(hid.view(timestep * batch, feature)).view(timestep, batch, feature)
        input = self.linear_input(input).view(1, batch, feature)
        total = hid + input
        total = torch.tanh(total)
        energy = total * self.w
        energy = torch.sum(energy, dim=2)
        energy = F.softmax(energy, dim=0)
        return energy.view(energy.size(0), energy.size(1), 1)


class LSTMAtt(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninput, nhid, nlayers):
        super(LSTMAtt, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embedding = nn.Embedding(nvoc, ninput)

        self.rnn_encoder = nn.LSTM(ninput, nhid, nlayers, bidirectional=False)
        self.rnn_decoder = nn.LSTM(nhid + ninput, nhid, nlayers, bidirectional=False)

        self.attention = Attention(nhid)
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type

    def init_weights(self):
        init_uniform = 0.1
        self.embedding.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, h):
        embeddings = self.drop(self.embedding(input))

        hid_enc = self.init_hidden(input.size(1))
        hid_dec = self.init_hidden(input.size(1))
        output, hid_enc = self.rnn_encoder(embeddings, hid_enc)

        timestep = input.size(0)
        predicts = []
        for i in range(0, timestep):
            att_hid = hid_dec[0][-1:]
            energy = self.attention(att_hid, output)
            state = torch.sum(energy * hid_enc[0][-1:], 0)
            pred, hid_dec = self.rnn_decoder(torch.cat((state, embeddings[i]), dim=1).view(1, state.size(0), -1),
                                             hid_dec)
            predicts.append(pred.view(-1, pred.size(2)))

        output = torch.stack(predicts, dim=0)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        hidden = (torch.stack((hid_enc[0], hid_dec[0]), 0), torch.stack((hid_enc[1], hid_dec[1]), 0))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


class LSTM_LN_Att(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninput, nhid, nlayers):
        super(LSTM_LN_Att, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embedding = nn.Embedding(nvoc, ninput)
        print(self.embedding)

        self.rnn_encoder = MultiLayerLSTM(ninput, LayerNormLSTM, tuple([nhid for _ in range(nlayers)]))
        self.rnn_decoder = MultiLayerLSTM(nhid + ninput, LayerNormLSTM, tuple([nhid for _ in range(nlayers)]))

        self.attention = Attention(nhid)

        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.embedding.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, h):
        embeddings = self.drop(self.embedding(input))

        hid_enc = self.init_hidden(input.size(1), "en")
        hid_dec = self.init_hidden(input.size(1), "de")
        output, hid_enc = self.rnn_encoder(embeddings, hid_enc)

        timestep = input.size(0)
        predicts = []
        for i in range(0, timestep):

            att_hid = hid_dec[-1][0]
            energy = self.attention(att_hid, output)
            state = torch.sum(energy * hid_enc[-1][0], 0)
            pred, hid_dec = self.rnn_decoder(torch.cat((state, embeddings[i]), dim=1).view(1, state.size(0), -1),
                                            hid_dec)
            predicts.append(pred.view(-1, pred.size(2)))

        output = torch.stack(predicts, dim=0)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        hidden = (hid_enc, hid_dec)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, type):
        if type == "en":
            return self.rnn_encoder.create_hiddens(bsz)
        else:
            return self.rnn_decoder.create_hiddens(bsz)