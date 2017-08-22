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
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch RNN Language + Time Difference Prediction Model on Kohyoung Dataset')
parser.add_argument('--data', type=str, default='ky',
                    help='type of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=10,
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
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=64,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
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
    #corpus = data.Corpus_ky_timeDifference('./dataset/ky/')
    corpus = data.Corpus_ky('./dataset/ky/')


def batchify_CNN(data, bsz, input_seq_len=100, target_len=1, shuffle=True):

    pair_len = input_seq_len + target_len
    pair_num = data.size(0) - pair_len + 1

    pairs = torch.LongTensor(pair_num,pair_len).fill_(0)
    for i in range(pair_num):
        pairs[i]=data[i:pair_len+i]

    if shuffle:
        pairs = pairs.numpy()
        np.random.shuffle(pairs)
        pairs = torch.LongTensor(pairs)

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = pairs .size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    pairs = pairs .narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    pairs = pairs .view(nbatch,bsz, -1)


    if args.cuda:
        pairs = pairs.cuda()
    return pairs

eval_batch_size = args.batch_size
#train_data_protocol = batchify_CNN(corpus.train_protocol, args.batch_size,input_seq_len=args.bptt)
#val_data_protocol = batchify_CNN(corpus.valid_protocol, eval_batch_size,input_seq_len=args.bptt)
#test_data_protocol = batchify_CNN(corpus.test_protocol, eval_batch_size,input_seq_len=args.bptt)

train_data_protocol = batchify_CNN(corpus.train, args.batch_size,input_seq_len=args.bptt)
val_data_protocol = batchify_CNN(corpus.valid, eval_batch_size,input_seq_len=args.bptt)
test_data_protocol = batchify_CNN(corpus.test, eval_batch_size,input_seq_len=args.bptt)

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
model1_for_protocol = model.CNNModel_protocol(ntoken=ntokens, ninp=args.emsize, input_length=args.bptt, kernel_sizes= [3,5,7], kernel_nums=[64, 128, 256,512])
if args.cuda:
    model1_for_protocol.cuda()

model1_optimizer = optim.Adam(model1_for_protocol.parameters(), lr= args.lr)

criterion_for_protocol = nn.CrossEntropyLoss()
###############################################################################
# Training code
###############################################################################

def get_batch_protocol(source, i, evaluation=False):
    input_seq_len = args.bptt
    data = Variable(source[i,:,:input_seq_len], volatile=evaluation)
    target = Variable(source[i,:,input_seq_len:],volatile=evaluation)
    return data, target

def evaluate(data_source_protocol):
    # Turn on evaluation mode which disables dropout.
    model1_for_protocol.eval()
    total_loss = 0
    loss_protocol = 0
    ntokens = len(corpus.dictionary)
    nbatch=0
    for nbatch, i in enumerate(range(data_source_protocol.size(0))):
        data1_protocol, target1_protocol = get_batch_protocol(data_source_protocol, i, True)
        model1_for_protocol.zero_grad()
        model1_optimizer.zero_grad()

        output1_for_protocol = model1_for_protocol.forward(data1_protocol)
        loss1_for_protocol = criterion_for_protocol(output1_for_protocol, target1_protocol.contiguous().view(-1))
        loss = loss1_for_protocol
        total_loss += loss.data
        loss_protocol += loss1_for_protocol.data

        losses_valid.append(loss.data[0])
        losses_valid_protocol.append(loss1_for_protocol.data[0])

    return total_loss[0] / nbatch,  loss_protocol[0] / nbatch

def train(source_data_protocol=train_data_protocol):
    # Turn on training mode which enables dropout.
    model1_for_protocol.train()
    total_loss = 0
    loss_protocol = 0

    start_time = time.time()
    for batch, i in enumerate(range(source_data_protocol.size(0))):
        data1_protocol, targets1_protocol = get_batch_protocol(source_data_protocol, i)
        model1_for_protocol.zero_grad()
        model1_optimizer.zero_grad()

        output1_for_protocol = model1_for_protocol.forward(data1_protocol)

        loss1_for_protocol = criterion_for_protocol(output1_for_protocol, targets1_protocol.contiguous().view(-1))
        loss = loss1_for_protocol #+ loss2_for_timeDiff
        #loss = loss1_for_protocol
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model1_for_protocol.parameters(), args.clip)

        model1_optimizer.step()
        '''
        # simple SGD
        for p in model1_for_protocol.parameters():
            p.data.add_(-lr, p.grad.data)
        for p in model2_for_timeDiff.parameters():
            p.data.add_(-lr, p.grad.data)
        '''

        total_loss += loss.data
        loss_protocol += loss1_for_protocol.data

        losses_train.append(loss.data[0])
        losses_train_protocol.append(loss1_for_protocol.data[0])

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_total = total_loss[0] / args.log_interval
            cur_loss_protocol = loss_protocol[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:2.3f} | ms/batch {:5.4f} | '
                    'total loss {:5.2f} (protocol= {:5.2f}, timeDiff= 00) | ppl {:8.4f}'.format(
                epoch, batch, len(source_data_protocol),  model1_optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss_total,
                              cur_loss_protocol,  math.exp(min(cur_loss_protocol,700))))
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
            val_loss_total, val_loss_protocol= evaluate(val_data_protocol)
            print('-' * 89)
            print('| end of epoch {:2d} | time: {:5.2f}s | valid total loss {:5.4f} (protocol= {:5.4f}, timeDiff= 00) | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss_total, val_loss_protocol, math.exp(val_loss_protocol)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            is_best = val_loss_total > best_val_loss
            best_val_loss = max(val_loss_total, best_val_loss)
            model_dictionary = {'epoch': epoch + 1,
                                'best_loss': best_val_loss,
                                'model1_state_dict': model1_for_protocol.state_dict(),
                                'model1_optimizer': model1_optimizer.state_dict(),
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
    with open('./result/ky/losses_valid.txt', option) as file_save_loss:
        for loss in losses_valid:
            file_save_loss.write(str(loss) + '\n')
    with open('./result/ky/losses_valid_protocol.txt', option) as file_save_loss:
        for loss in losses_valid_protocol:
            file_save_loss.write(str(loss) + '\n')


# Run on test data.
test_loss_total, test_loss_protocol= evaluate(test_data_protocol)
print('=' * 89)
print('| End of training | test total loss  {:5.4f} (protocol= {:5.4f}, timeDiff= 00) | '
'test ppl {:8.4f}'.format(test_loss_total, test_loss_protocol, math.exp(test_loss_protocol)))
print('=' * 89)


