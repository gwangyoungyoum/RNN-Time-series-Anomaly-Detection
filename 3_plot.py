import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

losses_train=[]
losses_valid=[]
with open('./result/ky/trial_5_CNN/losses_train_protocol.txt','r') as f:
    for line in f:
        losses_train.append(float(line))

with open('./result/ky/trial_5_CNN/losses_valid_protocol.txt','r') as f:
    for line in f:
        losses_valid.append(float(line))


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

print len(losses_train)
print len(losses_valid)
lists_train = split_list(losses_train, len(losses_train) / 3212)
lists_valid = split_list(losses_valid, len(losses_valid) / 1272)


losses_train_perEpoch=[]
losses_valid_perEpoch=[]

for i in range(len(lists_train)):
    losses_train_perEpoch.append(np.mean(lists_train[i]))
for i in range(len(lists_valid)):
    losses_valid_perEpoch.append(np.mean(lists_valid[i]))



plt.plot(losses_train_perEpoch,'r',label='train_200_2')
plt.plot(losses_valid_perEpoch,'b',label='test_200_2')

losses_train=[]
losses_valid=[]
with open('./result/ky/trial_3/losses_train_protocol.txt','r') as f:
    for line in f:
        losses_train.append(float(line))

with open('./result/ky/trial_3/losses_valid_protocol.txt','r') as f:
    for line in f:
        losses_valid.append(float(line))


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

print len(losses_train)
print len(losses_valid)
lists_train = split_list(losses_train, len(losses_train) / 2520)
lists_valid = split_list(losses_valid, len(losses_valid) / 920)


losses_train_perEpoch=[]
losses_valid_perEpoch=[]

for i in range(len(lists_train)):
    losses_train_perEpoch.append(np.mean(lists_train[i]))
for i in range(len(lists_valid)):
    losses_valid_perEpoch.append(np.mean(lists_valid[i]))



plt.plot(losses_train_perEpoch,'r',label='train_100_3', ls='dashed')
plt.plot(losses_valid_perEpoch,'b',label='test_100_3', ls='dashed')
plt.title('TimeDiff train and valid loss ')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()



plt.show()