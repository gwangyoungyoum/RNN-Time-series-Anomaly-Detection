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

eval_batch_size = 20
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
model2_optimizer = optim.Adam(model2_for_timeDiff.parameters(), lr= 0.0001)

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

def get_batch_timeDiff(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)

    data = Variable(source[i:i+seq_len], volatile=evaluation)
    q_data = data.unsqueeze(1).floor()
    r_data = data.unsqueeze(1) - q_data
    concat_data = torch.cat((q_data,r_data),1)
    target = Variable(source[i+1:i+1+seq_len])
    q_target = target.unsqueeze(1).floor()
    r_target = target.unsqueeze(1) - q_target
    concat_target = torch.cat((q_target, r_target), 1)

    return concat_data, concat_target.view(-1)


def evaluate(data_source_protocol, data_source_timeDiff):
    # Turn on evaluation mode which disables dropout.
    model1_for_protocol.eval()
    model2_for_timeDiff.eval()
    total_loss = 0
    loss_protocol = 0
    loss_timeDiff = 0
    ntokens = len(corpus.dictionary)
    hidden = model1_for_protocol.init_hidden(eval_batch_size)
    nbatch=0
    for nbatch, i in enumerate(range(0, data_source_protocol.size(0) - 1, args.bptt)):
        data1_protocol, targets1_protocol = get_batch(data_source_protocol, i, True)
        data2_timeDiff, targets2_timeDiff = get_batch(data_source_timeDiff, i, True)


        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model1_for_protocol.zero_grad()
        model2_for_timeDiff.zero_grad()
        outputs1_protocols=[]
        outputs2_timeDiffs=[]
        for j in range(data1_protocol.size(0)):
            output1_protocol, hidden = model1_for_protocol.forward(data1_protocol[j].unsqueeze(0), hidden)
            output2_timeDiff, hidden = model2_for_timeDiff.forward(data2_timeDiff[j].unsqueeze(0).t(), hidden)
            outputs1_protocols.append(output1_protocol)
            outputs2_timeDiffs.append(output2_timeDiff)
        outputs1_protocols = torch.cat(outputs1_protocols,0)
        outputs2_timeDiffs = torch.cat(outputs2_timeDiffs,0)

        loss1_for_protocol = criterion_for_protocol(outputs1_protocols.view(-1, ntokens), targets1_protocol)
        loss2_for_timeDiff = criterion_for_timeDiff(outputs2_timeDiffs.view(-1, 1), targets2_timeDiff)
        loss = loss1_for_protocol + loss2_for_timeDiff

        total_loss += loss.data
        loss_protocol += loss1_for_protocol.data
        loss_timeDiff += loss2_for_timeDiff.data

        losses_valid.append(loss.data[0])
        losses_valid_protocol.append(loss1_for_protocol.data[0])
        losses_valid_timeDiff.append(loss2_for_timeDiff.data[0])

    return total_loss[0] / nbatch,  loss_protocol[0] / nbatch, loss_timeDiff[0] / nbatch

def train(source_data_protocol=train_data_protocol, source_data_timeDiff=train_data_timeDiff):
    # Turn on training mode which enables dropout.
    model1_for_protocol.train()
    total_loss = 0
    loss_protocol = 0
    loss_timeDiff = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model1_for_protocol.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data_protocol.size(0) - 1, args.bptt)):
        data1_protocol, targets1_protocol = get_batch(source_data_protocol, i)
        data2_timeDiff, targets2_timeDiff = get_batch(source_data_timeDiff, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model1_for_protocol.zero_grad()
        model2_for_timeDiff.zero_grad()
        outputs1_protocols=[]
        outputs2_timeDiffs=[]
        for j in range(data1_protocol.size(0)):
            output1_protocol, hidden = model1_for_protocol.forward(data1_protocol[j].unsqueeze(0), hidden)
            output2_timeDiff, hidden = model2_for_timeDiff.forward(data2_timeDiff[j].unsqueeze(0).t(), hidden)
            outputs1_protocols.append(output1_protocol)
            outputs2_timeDiffs.append(output2_timeDiff)
        outputs1_protocols = torch.cat(outputs1_protocols,0)
        outputs2_timeDiffs = torch.cat(outputs2_timeDiffs,0)

        loss1_for_protocol = criterion_for_protocol(outputs1_protocols.view(-1, ntokens), targets1_protocol)
        loss2_for_timeDiff = criterion_for_timeDiff(outputs2_timeDiffs.view(-1, 1), targets2_timeDiff)

        for i in range(len(targets2_timeDiff)):
            pass
            #print '{:2.6f} {:2.6f} {:2.6f}'.format(outputs2_timeDiffs.view(-1, 1).data[i][0],targets2_timeDiff.data[i],outputs2_timeDiffs.view(-1, 1).data[i][0]-targets2_timeDiff.data[i])
        loss = loss1_for_protocol + loss2_for_timeDiff
        #loss = loss1_for_protocol
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model1_for_protocol.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm(model2_for_timeDiff.parameters(), args.clip)

        model1_optimizer.step()
        model2_optimizer.step()
        '''
        # simple SGD
        for p in model1_for_protocol.parameters():
            p.data.add_(-lr, p.grad.data)
        for p in model2_for_timeDiff.parameters():
            p.data.add_(-lr, p.grad.data)
        '''

        total_loss += loss.data
        loss_protocol += loss1_for_protocol.data
        loss_timeDiff += loss2_for_timeDiff.data

        losses_train.append(loss.data[0])
        losses_train_protocol.append(loss1_for_protocol.data[0])
        losses_train_timeDiff.append(loss2_for_timeDiff.data[0])

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_total = total_loss[0] / args.log_interval
            cur_loss_protocol = loss_protocol[0] / args.log_interval
            cur_loss_timeDiff = loss_timeDiff[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:2.3f} | ms/batch {:5.4f} | '
                    'total loss {:5.2f} (protocol= {:5.2f}, timeDiff= {:5.2f}) | ppl {:8.4f}'.format(
                epoch, batch, len(source_data_protocol) // args.bptt,  model1_optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss_total,
                              cur_loss_protocol, cur_loss_timeDiff, math.exp(min(cur_loss_protocol,700))))
            total_loss = 0
            loss_protocol = 0
            loss_timeDiff = 0
            start_time = time.time()

def save_checkpoint(state, is_best, filename='./save/ky/checkpoint.pth.tar'):
    print("=> saving checkpoint ..")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './save/ky/model_best.pth.tar')
    print('=> checkpoint saved.')


# Loop over epochs.
lr = args.lr
best_val_loss = None


if args.resume or args.pretrained:
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
    pass
else:
    print("=> Start training from scratch")
# At any point you can hit Ctrl + C to break out of training early.
if not args.pretrained:
    try:
        for epoch in range(1, 400+1):
            epoch_start_time = time.time()
            train()
            val_loss_total, val_loss_protocol, val_loss_timeDiff = evaluate(val_data_protocol, val_data_timeDiff)
            print('-' * 89)
            print('| end of epoch {:2d} | time: {:5.2f}s | valid total loss {:5.4f} (protocol= {:5.4f}, timeDiff= {:5.2f}) | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss_total, val_loss_protocol, val_loss_timeDiff , math.exp(val_loss_protocol)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            is_best = val_loss_total > best_val_loss
            best_val_loss = max(val_loss_total, best_val_loss)
            model_dictionary = {'epoch': epoch + 1,
                                'best_loss': best_val_loss,
                                'model1_state_dict': model1_for_protocol.state_dict(),
                                'model1_optimizer': model1_optimizer.state_dict(),
                                'model2_state_dict': model2_for_timeDiff.state_dict(),
                                'model2_optimizer': model2_optimizer.state_dict(),
                                }
            save_checkpoint(model_dictionary, is_best)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

if not args.pretrained:

    if args.resume:
        option='a'
    else:
        option='w'

    with open('./result/ky/losses_train.txt', option) as file_save_loss:
        for loss in losses_train:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_train_protocol.txt', option) as file_save_loss:
        for loss in losses_train_protocol:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_train_timeDiff.txt', option) as file_save_loss:
        for loss in losses_train_timeDiff:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_valid.txt', option) as file_save_loss:
        for loss in losses_valid:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_valid_protocol.txt', option) as file_save_loss:
        for loss in losses_valid_protocol:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_valid_timeDiff.txt', option) as file_save_loss:
        for loss in losses_valid_timeDiff:
            file_save_loss.write(str(loss) + '\n')

# Run on test data.
test_loss_total, test_loss_protocol, test_loss_timeDiff = evaluate(test_data_protocol, test_data_timeDiff)
print('=' * 89)
print('| End of training | test total loss  {:5.4f} (protocol= {:5.4f}, timeDiff= {:5.4f}) | '
'test ppl {:8.4f}'.format(test_loss_total, test_loss_protocol, test_loss_timeDiff, math.exp(test_loss_protocol)))
print('=' * 89)


