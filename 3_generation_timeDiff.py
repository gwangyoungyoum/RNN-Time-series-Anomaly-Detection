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

parser = argparse.ArgumentParser(description='PyTorch RNN Language Model on PenTreeBank/Kohyoung Dataset')
parser.add_argument('--data', type=str, default='nyc_taxi',
                    help='type of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=bool, default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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
elif args.data == 'nyc_taxi':
    corpus = data.Corpus_nyc_taxi('./dataset/nyc_taxi/')



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
train_data_timeDiff = batchify(corpus.train_timeDiff, args.batch_size)
val_data_timeDiff = batchify(corpus.valid_timeDiff, eval_batch_size)
test_data_timeDiff = batchify(corpus.test_timeDiff, eval_batch_size)

losses_train=[]
losses_train_timeDiff=[]
losses_valid=[]
losses_valid_timeDiff=[]
losses_test=[]
losses_test_timeDiff=[]


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model2_for_timeDiff = model.RNNModel_timeDiff(args.model, 1, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
print(list(model2_for_timeDiff.parameters()))
if args.cuda:
    model2_for_timeDiff.cuda()

model2_optimizer = optim.Adam(model2_for_timeDiff.parameters(), lr= 0.001)

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


def generate_output(data_source_timeDiff):
    # Turn on evaluation mode which disables dropout.
    model2_for_timeDiff.eval()
    hidden = model2_for_timeDiff.init_hidden(eval_batch_size)
    outputs2_timeDiffs = []
    file = open('output_generated.txt','w')
    for i in range(8000):

        if i>5500:
            output2_timeDiff, hidden = model2_for_timeDiff.forward(output2_timeDiff[0], hidden)
        else:
            output2_timeDiff, hidden = model2_for_timeDiff.forward(Variable(data_source_timeDiff[i].unsqueeze(0).t(), volatile=True), hidden)

        #print data_source_timeDiff[i].unsqueeze(0).t()
        #print output2_timeDiff

        output2_timeDiff_value = output2_timeDiff.data.cpu()[0][0][0]
        target2_timeDiff_value = data_source_timeDiff[i+1].cpu()[0]

        outputs2_timeDiffs.append(output2_timeDiff_value)
        print( 'output timeDiff {:9.4f} | target timeDiff {:9.4f}'\
           .format(output2_timeDiff_value,target2_timeDiff_value))
        file.write('output timeDiff {:9.4f} | target timeDiff {:9.4f}'\
            .format(output2_timeDiff_value,target2_timeDiff_value)+'\n')
    file.close()

    return outputs2_timeDiffs


# Loop over epochs.
lr = args.lr
best_val_loss = None



print("=> loading checkpoint ")
checkpoint = torch.load('./save/'+args.data+'/checkpoint.pth.tar')
args.start_epoch = checkpoint['epoch']
best_loss = checkpoint['best_loss']
model2_for_timeDiff.load_state_dict(checkpoint['model2_state_dict'])
model2_optimizer.load_state_dict((checkpoint['model2_optimizer']))
del checkpoint
print("=> loaded checkpoint")


# At any point you can hit Ctrl + C to break out of training early.

try:

   outputs2_timeDiff = generate_output(val_data_timeDiff)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


from matplotlib import pyplot as plt
plt.figure()
plot1=plt.plot(val_data_timeDiff.cpu().numpy(),'.b',label='target')
plot2=plt.plot(outputs2_timeDiff,'--r',label='output')
plt.legend()
plt.show()