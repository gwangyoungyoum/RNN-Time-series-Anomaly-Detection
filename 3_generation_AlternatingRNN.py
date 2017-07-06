import argparse
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import data
from model import model
import shutil
parser = argparse.ArgumentParser(description='PyTorch RNN Language + Time Difference Prediction Model on Kohyoung Dataset')
parser.add_argument('--data', type=str, default='ky',
                    help='type of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=bool, default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
if args.data == 'ptb':
    corpus = data.Corpus('./dataset/ptb/')
elif args.data == 'ky':
    corpus = data.Corpus_ky_timeDifference('./dataset/ky/')





def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 1
train_data_protocol = batchify(corpus.train_protocol, args.batch_size)
train_data_timeDiff = batchify(corpus.train_timeDiff, args.batch_size)
val_data_protocol = batchify(corpus.valid_protocol, eval_batch_size)
val_data_timeDiff = batchify(corpus.valid_timeDiff, eval_batch_size)
test_data_protocol = batchify(corpus.test_protocol, eval_batch_size)
test_data_timeDiff = batchify(corpus.test_timeDiff, eval_batch_size)

losses_train=[]
losses_train_protocol=[]
losses_train_timeDiff=[]
losses_valid=[]
losses_valid_protocol=[]
losses_valid_timeDiff=[]
losses_test=[]
losses_test_protocol=[]
losses_test_timeDiff=[]


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model1_for_protocol = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
model2_for_timeDiff = model.RNNModel_timeDiff(args.model, 1, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model1_for_protocol.cuda()
    model2_for_timeDiff.cuda()

model1_optimizer = optim.Adam(model1_for_protocol.parameters(), lr= args.lr)
model2_optimizer = optim.Adam(model2_for_timeDiff.parameters(), lr= 0.1)

criterion_for_protocol = nn.CrossEntropyLoss()
criterion_for_timeDiff = nn.MSELoss()
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def generate_output(data_source_protocol, data_source_timeDiff):
    # Turn on evaluation mode which disables dropout.
    model1_for_protocol.eval()
    model2_for_timeDiff.eval()
    hidden = model1_for_protocol.init_hidden(eval_batch_size)
    outputs1_protocols = []
    outputs2_timeDiffs = []
    file = open('output_generated.txt','w')
    for i in range(data_source_protocol.size(0)-1):

        output1_protocol, hidden = model1_for_protocol.forward(Variable(data_source_protocol[i].unsqueeze(0),volatile=True), hidden)
        output2_timeDiff, hidden = model2_for_timeDiff.forward(Variable(data_source_timeDiff[i].unsqueeze(0).t(),volatile=True), hidden)

        val, protocol_idx = output1_protocol.topk(1)

        output1_protocol_word = corpus.dictionary.idx2word[protocol_idx.data.cpu()[0][0][0]]
        target1_protocol_word = corpus.dictionary.idx2word[data_source_protocol[i+1].cpu()[0]]
        output2_timeDiff_value = output2_timeDiff.data.cpu()[0][0][0]
        target2_timeDiff_value = data_source_timeDiff[i+1].cpu()[0]

        outputs1_protocols.append(output1_protocol_word)
        outputs2_timeDiffs.append(output2_timeDiff_value)

        #print i
        print 'output protocol {:>20} | target protocol {:>20}   |   output timeDiff {:9.4f} | target timeDiff {:9.4f}'\
           .format(output1_protocol_word,target1_protocol_word,output2_timeDiff_value,target2_timeDiff_value)
        file.write('output protocol {:>20} | target protocol {:>20}   |   output timeDiff {:9.4f} | target timeDiff {:9.4f}'\
            .format(output1_protocol_word,target1_protocol_word,output2_timeDiff_value,target2_timeDiff_value)+'\n')
    file.close()

    return outputs1_protocols, outputs2_timeDiffs


# Loop over epochs.
lr = args.lr
best_val_loss = None



print("=> loading checkpoint ")
checkpoint = torch.load('./save/ky/checkpoint.pth.tar')
args.start_epoch = checkpoint['epoch']
best_loss = checkpoint['best_loss']
model1_for_protocol.load_state_dict(checkpoint['model1_state_dict'])
model1_optimizer.load_state_dict((checkpoint['model1_optimizer']))
model2_for_timeDiff.load_state_dict(checkpoint['model2_state_dict'])
model2_optimizer.load_state_dict((checkpoint['model2_optimizer']))
del checkpoint
print("=> loaded checkpoint")


# At any point you can hit Ctrl + C to break out of training early.

try:

   outputs1_protocol, outputs2_timeDiff = generate_output(val_data_protocol, val_data_timeDiff)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


from matplotlib import pyplot as plt
plt.figure()
plot1=plt.plot(val_data_timeDiff.cpu().numpy(),'black',label='target')
plot2=plt.plot(outputs2_timeDiff,'r',label='output')
plt.legend()
plt.show()