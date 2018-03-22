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
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
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
TimeseriesData = preprocess_data.DataLoad(args.data)





def batchify(data, bsz,data_type=args.data):
    if data_type == 'nyc_taxi':
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
    elif data_type == 'ecg':
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data['seqData1'].size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        dataset = {}
        for key in ['seqData1', 'seqData2']:
            dataset[key] = data[key].narrow(0, 0, nbatch * bsz)
            # Evenly divide the data across the bsz batches.
            dataset[key] = dataset[key].view(bsz, -1).t().contiguous().unsqueeze(
                2)  # data: [ seq_length * batch_size * 1 ]

        batched_data = torch.cat([dataset['seqData1'], dataset['seqData2']], dim=2)
        # batched_data: [ seq_length * batch_size * feature_size ] , feature_size = 2

    if args.cuda:
        batched_data = batched_data.cuda()

    return batched_data


train_dataset = batchify(TimeseriesData.trainData, 1)[:10000]
test_dataset = batchify(TimeseriesData.testData, 1)


###############################################################################
# Build the model
###############################################################################

model = model.RNNPredictor(rnn_type = args.model, enc_inp_size=3, rnn_inp_size = args.emsize, rnn_hid_size = args.nhid,
                           dec_out_size=3,
                           nlayers = args.nlayers,)

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr= 0.0001)
criterion = nn.MSELoss()
###############################################################################
# Training code
###############################################################################


def fit_norm_distribution_param(args, model, train_dataset, endPoint=10000):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    pasthidden = model.init_hidden(1)
    predictions = []
    organized = []
    errors = []
    #out = Variable(test_dataset[0].unsqueeze(0))
    for t in range(endPoint):
        out, hidden = model.forward(Variable(train_dataset[t].unsqueeze(0), volatile=True), pasthidden)
        predictions.append([])
        organized.append([])
        errors.append([])
        predictions[t].append(out.data.cpu()[0][0][0])
        pasthidden = model.repackage_hidden(hidden)
        for prediction_step in range(1,args.prediction_window_size):
            out, hidden = model.forward(out, hidden)
            predictions[t].append(out.data.cpu()[0][0][0])

        if t >= args.prediction_window_size:
            for step in range(args.prediction_window_size):
                organized[t].append(predictions[step+t-args.prediction_window_size][args.prediction_window_size-1-step])
            errors[t] = torch.FloatTensor(organized[t]) - train_dataset[t][0][0]
            if args.cuda:
                errors[t] = errors[t].cuda()
            errors[t] = errors[t].unsqueeze(0)

    errors_tensor = torch.cat(errors[args.prediction_window_size:],dim=0)
    mean = errors_tensor.mean(dim=0)
    cov = errors_tensor.t().mm(errors_tensor)/errors_tensor.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
    # cov: positive-semidefinite and symmetric.

    return mean, cov

def anomalyScore(args,model,test_dataset,mean,cov,endPoint=10000):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    pasthidden = model.init_hidden(1)
    predictions = []
    organized = []
    errors = []
    # out = Variable(test_dataset[0].unsqueeze(0))
    for t in range(endPoint):
        out, hidden = model.forward(Variable(test_dataset[t].unsqueeze(0), volatile=True), pasthidden)
        predictions.append([])
        organized.append([])
        errors.append([])
        predictions[t].append(out.data.cpu()[0][0][0])
        pasthidden = model.repackage_hidden(hidden)
        for prediction_step in range(1, args.prediction_window_size):
            out, hidden = model.forward(out, hidden)
            predictions[t].append(out.data.cpu()[0][0][0])

        if t >= args.prediction_window_size:
            for step in range(args.prediction_window_size):
                organized[t].append(
                    predictions[step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
            organized[t] =torch.FloatTensor(organized[t]).unsqueeze(0)
            errors[t] = organized[t] - test_dataset[t][0][0]
            if args.cuda:
                errors[t] = errors[t].cuda()
        else:
            organized[t] = torch.zeros(1,args.prediction_window_size)
            errors[t] = torch.zeros(1,args.prediction_window_size)
            if args.cuda:
                errors[t] = errors[t].cuda()

    scores = []
    for error in errors:
        mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]
        mult2 = torch.inverse(cov) # [ prediction_window_size * prediction_window_size ]
        mult3 = mult1.t() # [ prediction_window_size * 1 ]
        score = torch.mm(mult1,torch.mm(mult2,mult3))
        scores.append(score[0][0])
    return scores, organized, errors


# Loop over epochs.
best_val_loss = None



print("=> loading checkpoint ")
checkpoint = torch.load('./save/'+args.data+'/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
del checkpoint
print("=> loaded checkpoint")

# At any point you can hit Ctrl + C t
endPoint=3500
try:

   mean, cov = fit_norm_distribution_param(args, model, train_dataset, endPoint)
   scores, sorted_predictions,sorted_errors = anomalyScore(args, model, test_dataset, mean, cov, endPoint)

   sorted_predictions = torch.cat(sorted_predictions, dim=0)
   sorted_errors = torch.cat(sorted_errors,dim=0)

   scores = np.array(scores)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
target= preprocess_data.reconstruct(test_dataset.cpu()[:, 0, 0].numpy(),
                                     TimeseriesData.trainData['seqData_mean'],
                                     TimeseriesData.trainData['seqData_std'])
sorted_predictions_mean = preprocess_data.reconstruct(sorted_predictions.mean(dim=1).numpy(),
                                     TimeseriesData.trainData['seqData_mean'],
                                     TimeseriesData.trainData['seqData_std'])

sorted_predictions_1step = preprocess_data.reconstruct(sorted_predictions[:,-1].numpy(),
                                     TimeseriesData.trainData['seqData_mean'],
                                     TimeseriesData.trainData['seqData_std'])

sorted_predictions_Nstep = preprocess_data.reconstruct(sorted_predictions[:,0].numpy(),
                                     TimeseriesData.trainData['seqData_mean'],
                                     TimeseriesData.trainData['seqData_std'])
sorted_errors_mean = sorted_errors.mean(dim=1).cpu().numpy()
sorted_errors = sorted_errors.abs()
#sorted_errors_mean = sorted_errors_mean**0.5
sorted_errors_mean *=TimeseriesData.trainData['seqData_std']

fig, ax1 = plt.subplots(figsize=(15,5))
ax1.plot(target,label='Target', color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)

ax1.plot(sorted_predictions_mean,label='Mean predictions', color='purple', marker='.', linestyle='--', markersize=1, linewidth=0.5)
ax1.plot(sorted_predictions_1step,label='1-step predictions', color='green', marker='.', linestyle='--', markersize=1, linewidth=0.5)
ax1.plot(sorted_predictions_Nstep,label=str(args.prediction_window_size)+'-step predictions', color='blue', marker='.', linestyle='--', markersize=1, linewidth=0.5)
ax1.plot(sorted_errors_mean,label='Absolute prediction errors', color='orange', marker='.', linestyle='--', markersize=1, linewidth=1)

ax1.legend(loc='upper left')
ax1.set_ylabel('Value',fontsize=15)
ax1.set_xlabel('Index',fontsize=15)

ax2 = ax1.twinx()
ax2.plot(scores,label='Anomaly scores from \nmultivariate normal distribution', color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
ax2.legend(loc='upper right')
ax2.set_ylabel('anomaly score',fontsize=15)
plt.xlim([0, endPoint])
plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.xlim([1500,endPoint])
plt.savefig('result/'+args.data+'/fig_scores.png')
plt.show()
