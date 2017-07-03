import argparse
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
from model import model
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch RNN Language Model on PenTreeBank/Kohyoung Dataset')
parser.add_argument('--data', type=str, default='ky',
                    help='type of the dataset')
parser.add_argument('--model', type=str, default='RNN_TANH',
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
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
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
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
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

eval_batch_size = 20
train_data_timeDiff = batchify(corpus.train_timeDiff, args.batch_size)
val_data_timeDiff = batchify(corpus.valid_timeDiff, eval_batch_size)
test_data_timeDiff = batchify(corpus.test_timeDiff, eval_batch_size)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model2_for_timeDiff = model.RNNModel_timeDiff(args.model, 1, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model2_for_timeDiff.cuda()

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


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model2_for_timeDiff.eval()
    total_loss = 0
    hidden = model2_for_timeDiff.init_hidden(eval_batch_size)
    for nbatch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data2_timeDiff, targets2_timeDiff = get_batch(data_source, i, evaluation=True)
        outputs2_timeDiffs = []
        for j in range(data2_timeDiff.size(0)):
            output2_timeDiff, hidden = model2_for_timeDiff.forward(data2_timeDiff[j].unsqueeze(0).t(), hidden)
            outputs2_timeDiffs.append(output2_timeDiff)

        outputs2_timeDiffs = torch.cat(outputs2_timeDiffs, 0)
        loss2_for_timeDiff = criterion_for_timeDiff(outputs2_timeDiffs.view(-1, 1), targets2_timeDiff)
        hidden = repackage_hidden(hidden)
        total_loss+= loss2_for_timeDiff.data

    return total_loss[0] / nbatch


def train():
    # Turn on training mode which enables dropout.
    model2_for_timeDiff.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model2_for_timeDiff.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data_timeDiff.size(0) - 1, args.bptt)):
        data2_timeDiff, targets2_timeDiff = get_batch(train_data_timeDiff, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model2_for_timeDiff.zero_grad()
        outputs2_timeDiffs=[]
        for j in range(data2_timeDiff.size(0)):
            output2_timeDiff, hidden = model2_for_timeDiff.forward(data2_timeDiff[j].unsqueeze(0).t(), hidden)
            outputs2_timeDiffs.append(output2_timeDiff)
        outputs2_timeDiffs = torch.cat(outputs2_timeDiffs,0)

        loss2_for_timeDiff = criterion_for_timeDiff(outputs2_timeDiffs.view(-1, 1), targets2_timeDiff)
        loss = loss2_for_timeDiff
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model2_for_timeDiff.parameters(), args.clip)

        for p in model2_for_timeDiff.parameters():
            p.data.add_(-lr, p.grad.data)


        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.4f} | '
                    'loss {:5.2f} '.format(
                epoch, batch, len(train_data_timeDiff) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data_timeDiff)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),
                                           val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open('./save/'+ args.data +'/'+ args.save, 'wb') as f:
                torch.save(model2_for_timeDiff, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('./save/'+ args.data +'/'+ args.save, 'rb') as f:
    model2_for_timeDiff = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data_timeDiff)
print('=' * 89)
print('| End of training | test loss {:5.24} '.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)