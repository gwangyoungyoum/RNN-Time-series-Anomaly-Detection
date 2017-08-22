import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        x = emb.clone()
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        x = output.clone()
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class RNNModel_timeDiff(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel_timeDiff, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.drop(self.encoder(input))
        emb = self.drop(self.encoder(input.contiguous().view(input.size(0)*input.size(1),1)))
        output, hidden = self.rnn(emb.view(input.size(1),input.size(0),self.nhid), hidden)
        output = self.drop(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #decoded = decoded + x
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class CNNModel_protocol(nn.Module):

    def __init__(self, ntoken, ninp, input_length, kernel_sizes, kernel_nums, reductionRatio = 4):
        super(CNNModel_protocol,self).__init__()

        self.kernel_sizes=kernel_sizes
        blockNum=0
        self.encoder = nn.Embedding(ntoken, ninp)
        self.norms1 = nn.ModuleList([nn.BatchNorm2d(1) for kernel_size in kernel_sizes])
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_nums[blockNum], (kernel_size, ninp), padding=(i+1, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        blockNum=0
        self.norms21 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum]) for kernel_size in kernel_sizes])
        self.convs21 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] // reductionRatio, (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms22 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs22 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum] // reductionRatio, (kernel_size, 1), padding=(i + 1, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms23 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs23 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum], (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.convs24 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] * 2, (1, 1), padding=(0, 0), stride=2) for i, kernel_size in enumerate(kernel_sizes)])

        blockNum=1
        self.norms31 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum]) for kernel_size in kernel_sizes])
        self.convs31 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] // reductionRatio, (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms32 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs32 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum] // reductionRatio, (kernel_size, 1), padding=(i + 1, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms33 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs33 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum], (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.convs34 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] * 2, (1, 1), padding=(0, 0), stride=2) for i, kernel_size in enumerate(kernel_sizes)])

        blockNum=2
        self.norms41 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum]) for kernel_size in kernel_sizes])
        self.convs41 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] // reductionRatio, (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms42 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs42 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum] // reductionRatio, (kernel_size, 1), padding=(i + 1, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms43 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs43 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum], (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.convs44 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] * 2, (1, 1), padding=(0, 0), stride=2) for i, kernel_size in enumerate(kernel_sizes)])


        blockNum=3
        self.norms51 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum]) for kernel_size in kernel_sizes])
        self.convs51 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum] // reductionRatio, (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms52 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs52 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum] // reductionRatio, (kernel_size, 1), padding=(i + 1, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.norms53 = nn.ModuleList([nn.BatchNorm2d(kernel_nums[blockNum] // reductionRatio) for kernel_size in kernel_sizes])
        self.convs53 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum] // reductionRatio, kernel_nums[blockNum], (1, 1), padding=(0, 0)) for i, kernel_size in enumerate(kernel_sizes)])
        self.convs54 = nn.ModuleList([nn.Conv2d(kernel_nums[blockNum], kernel_nums[blockNum], (1, 1), padding=(0, 0), stride=2) for i, kernel_size in enumerate(kernel_sizes)])


        self.decoder = nn.Linear(len(kernel_sizes) * (kernel_nums[blockNum]), ntoken)


    def forward(self, input):

        '''
        # Trial 5
        emb = self.drop(self.encoder(input)) #(batch_size,#words,wordVec)
        x = emb.unsqueeze(1)
        x = [conv(x) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms1[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_).squeeze(3) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(x_, 2) for x_ in x]  # [(N,Co), ...]*len(Ks)

        x = [x_.unsqueeze(3) for x_ in x]
        x = [self.convs2[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms2[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_).squeeze(3) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(x_, x_.size(2)).squeeze(2) for x_ in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        decoded = self.decoder(x) # (batch_size, ntoken)
        '''

        emb = self.encoder(input) #(batch_size,#words,wordVec)
        x = emb.unsqueeze(1)
        x = [norm(x) for norm in self.norms1]  # [(N,Co,W), ...]*len(Ks)
        #x = [x]*len(self.kernel_sizes)
        x = [self.convs1[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_).squeeze(3) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(x_, 2) for x_ in x]  # [(N,Co), ...]*len(Ks)
        x = [x_.unsqueeze(3) for x_ in x]

        residue = [x_.clone() for x_ in x]
        x = [self.norms21[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs21[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms22[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs22[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms23[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x]  # [(N,Co,W), ...]*len(Ks)
        x = [self.convs23[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [x_ + residue[i] for i, x_ in enumerate(x)]

        x = [self.convs24[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)

        residue = [x_.clone() for x_ in x]
        x = [self.norms31[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs31[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms32[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs32[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms33[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x]  # [(N,Co,W), ...]*len(Ks)
        x = [self.convs33[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [x_ + residue[i] for i, x_ in enumerate(x)]

        x = [self.convs34[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)

        residue = [x_.clone() for x_ in x]
        x = [self.norms41[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs41[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms42[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs42[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms43[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x]  # [(N,Co,W), ...]*len(Ks)
        x = [self.convs43[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [x_ + residue[i] for i, x_ in enumerate(x)]


        x = [self.convs44[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)



        residue = [x_.clone() for x_ in x]
        x = [self.norms51[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs51[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms52[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x] #[(N,Co,W), ...]*len(Ks)
        x = [self.convs52[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [self.norms53[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [F.relu(x_) for x_ in x]  # [(N,Co,W), ...]*len(Ks)
        x = [self.convs53[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        x = [x_ + residue[i] for i, x_ in enumerate(x)]


        x = [self.convs54[i](x_) for i, x_ in enumerate(x)]  # [(N,Co,W), ...]*len(Ks)
        


        x = [x_.squeeze(3) for x_ in x]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(x_, x_.size(2)).squeeze(2) for x_ in x]  # [(N,Co), ...]*len(Ks)
#        x = [F.avg_pool1d(x_, x_.size(2)).squeeze(2) for x_ in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        decoded = self.decoder(x) # (batch_size, ntoken)

        return decoded

