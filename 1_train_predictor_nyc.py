import argparse
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import preprocess_data
from model import model
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
import shutil

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='nyc_taxi',
                    help='type of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of encoded features by the linear layer')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio')
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
TimeseriesData = preprocess_data.DataLoad(args.data)




def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data['seqData'].size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    dataset = {}
    for key in ['seqData','timeOfDay','dayOfWeek']:
        dataset[key] = data[key].narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        dataset[key] = dataset[key].view(bsz, -1).t().contiguous().unsqueeze(2) # data: [ seq_length * batch_size * 1 ]

    batched_data = torch.cat([dataset['seqData'],dataset['timeOfDay'],dataset['dayOfWeek']],dim=2)
    # batched_data: [ seq_length * batch_size * feature_size ] , feature_size = 3
    if args.cuda:
        batched_data = batched_data.cuda()
    return batched_data


train_dataset = batchify(TimeseriesData.trainData, args.batch_size)
test_dataset = batchify(TimeseriesData.testData, args.eval_batch_size)
gen_dataset = batchify(TimeseriesData.testData, 1)



###############################################################################
# Build the model
###############################################################################

model = model.RNNPredictor(rnn_type = args.model, enc_inp_size=3, rnn_inp_size = args.emsize, rnn_hid_size = args.nhid,
                           dec_out_size=3, nlayers = args.nlayers, dropout = args.dropout, tie_weights= args.tied)
print(list(model.parameters()))
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr= args.lr)
criterion = nn.MSELoss()
###############################################################################
# Training code
###############################################################################



def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation) # [ seq_len * batch_size * feature_size ]
    target = Variable(source[i+1:i+1+seq_len]) # [ (seq_len x batch_size x feature_size) ]
    return data, target

def generate_output(args,epoch, model, gen_dataset, startPoint=500, endPoint=3500):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(1)
    outSeq = []
    for i in range(endPoint):

        if i>startPoint:
            out, hidden = model.forward(out, hidden)
        else:
            out, hidden = model.forward(Variable(gen_dataset[i].unsqueeze(0), volatile=True), hidden)


        outValue = out.data.cpu()[0][0][0]

        outSeq.append(outValue)

    target= preprocess_data.reconstruct(gen_dataset.cpu()[:, 0, 0].numpy(),
                                         TimeseriesData.trainData['seqData_mean'],
                                         TimeseriesData.trainData['seqData_std'])

    outSeq = preprocess_data.reconstruct(np.array(outSeq),
                                          TimeseriesData.trainData['seqData_mean'],
                                          TimeseriesData.trainData['seqData_std'])

    plt.figure(figsize=(15,5))
    plot1 = plt.plot(target,
                     label='Target',color='black',marker='.',linestyle='--', markersize=1,linewidth=0.5)
    plot2 = plt.plot(range(startPoint), outSeq[:startPoint],
                     label='1-step prediction', color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)

    plot3 = plt.plot(range(startPoint, endPoint, 1), outSeq[startPoint:],
                     label='Multi-step prediction', color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
    plt.xlim([1500, endPoint])
    plt.xlabel('Index',fontsize=15)
    plt.ylabel('Value',fontsize=15)

    plt.title('Time-series Prediction on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.text(1520, 32000, 'Epoch: '+str(epoch),fontsize=15)
    plt.savefig('result/nyc_taxi/fig_epoch'+str(epoch)+'.png')
    #plt.show()
    plt.close()

    return outSeq

def evaluate(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(args.eval_batch_size)
    for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):

        inputSeq, targetSeq = get_batch(test_dataset, i, evaluation=True)
        # outVal = inputSeq[0].unsqueeze(0)
        # outVals = []
        # for i in range(inputSeq.size(0)):
        #     outVal, hidden = model.forward(outVal, hidden)
        #     outVals.append(outVal)
        # outSeq = torch.cat(outVals, dim=0)

        outSeq, hidden = model.forward(inputSeq, hidden)

        loss = criterion(outSeq.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
        hidden = model.repackage_hidden(hidden)
        total_loss+= loss.data

    return total_loss[0] / nbatch

def train(args, model, train_dataset):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_dataset.size(0) - 1, args.bptt)):
        inputSeq, targetSeq = get_batch(train_dataset, i)
        # inputSeq: [ seq_len * batch_size * feature_size ]
        # targetSeq: [ seq_len * batch_size * feature_size ]

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.repackage_hidden(hidden)
        optimizer.zero_grad()
        USE_TEACHER_FORCING =  random.random() < args.teacher_forcing_ratio
        if USE_TEACHER_FORCING:
            outSeq, hidden = model.forward(inputSeq, hidden)
        else:
            outVal = inputSeq[0].unsqueeze(0)
            outVals=[]
            for i in range(inputSeq.size(0)):
                outVal, hidden = model.forward(outVal, hidden)
                outVals.append(outVal)
            outSeq = torch.cat(outVals,dim=0)

        #print('outSeq:',outSeq.size())

        #print('targetSeq:', targetSeq.size())

        loss = criterion(outSeq.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                  'loss {:5.2f} '.format(
                epoch, batch, len(train_dataset) // args.bptt,
                              elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
teacher_forcing_ratio = 1

if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = torch.load('./save/'+args.data+'/checkpoint.pth.tar')
    args.start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    print("=> loaded checkpoint")
    pass
else:
    print("=> Start training from scratch")
# At any point you can hit Ctrl + C to break out of training early.
save_interval=10
best_val_loss=0
if not args.pretrained:
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args,model,train_dataset)
            val_loss = evaluate(args,model,test_dataset)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),
                                               val_loss))
            print('-' * 89)

            if epoch%save_interval==0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss > best_val_loss
                best_val_loss = max(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch + 1,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    }
                model.save_checkpoint(args,model_dictionary, is_best)
            generate_output(args,epoch,model,gen_dataset,startPoint=2000)



    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


