import os
import torch
import glob
import datetime
import numpy as np
import shutil

def normalization(seqData,max,min):
    return (seqData -min)/(max-min)

def standardization(seqData,mean,std):
    return (seqData-mean)/std

def reconstruct(seqData,mean,std):
    return seqData*std+mean


def DataLoad(data_type):
    try:
        if data_type == 'nyc_taxi':
            return NYCDataLoad('./dataset/'+ data_type +'/')
        elif data_type == 'ecg':
            return ECGDataLoad('./dataset/' + data_type + '/')
        else:
            raise ValueError
    except ValueError:
        print("Uknown data type: " + data_type)


class NYCDataLoad(object):
    def __init__(self, path,augumentation=True):
        self.augumentation=augumentation
        self.trainData = self.preprocessing(path + 'trainset/nyc_taxi.csv', trainData=True)
        self.validData = self.preprocessing(path + 'testset/nyc_taxi.csv', trainData=False)

    def preprocessing(self, path, trainData=True):
        """ Read, Standardize, Augment """
        seqData = []
        timeOfDay = []
        dayOfWeek = []
        dataset={}

        noise_val=0.8 if self.augumentation else 0.1
        for std in np.arange(0,noise_val,0.1):
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i > 0:
                        line_splited = line.strip().split(',')
                        # print line_splited
                        seqData.append(float(line_splited[1]) + np.random.normal(0, std))
                        timeOfDay.append(float(line_splited[2]))
                        dayOfWeek.append(float(line_splited[3]))

        if trainData:
            dataset['seqData_mean'] = np.mean(seqData)
            dataset['seqData_std']= np.std(seqData, ddof=1)
            dataset['timeOfDay_mean'] = np.mean(timeOfDay)
            dataset['timeOfDay_std'] = np.std(timeOfDay,ddof=1)
            dataset['dayOfWeek_mean'] = np.mean(dayOfWeek)
            dataset['dayOfWeek_std'] = np.std(dayOfWeek,ddof=1)

            #print 'seqData_mean =', seqData_mean
            #print 'seqData_std =', seqData_std
        else:
            dataset['seqData_mean'] = self.trainData['seqData_mean']
            dataset['seqData_std'] = self.trainData['seqData_std']
            dataset['timeOfDay_mean'] = self.trainData['timeOfDay_mean']
            dataset['timeOfDay_std'] = self.trainData['timeOfDay_std']
            dataset['dayOfWeek_mean'] = self.trainData['dayOfWeek_mean']
            dataset['dayOfWeek_std'] = self.trainData['dayOfWeek_std']

        seqData = torch.FloatTensor(seqData)
        seqData = standardization(seqData, dataset['seqData_mean'], dataset['seqData_std'])
        if self.augumentation:
            seqData_corrupted1 = seqData + 0.1*torch.randn(seqData.size(0))
            seqData_corrupted2 = seqData + 0.2*torch.randn(seqData.size(0))
            seqData = torch.cat([seqData,seqData_corrupted1],0)
            seqData = torch.cat([seqData, seqData_corrupted2], 0)
        dataset['seqData'] = seqData

        timeOfDay = torch.FloatTensor(timeOfDay)
        timeOfDay = standardization(timeOfDay, dataset['timeOfDay_mean'], dataset['timeOfDay_std'])

        if self.augumentation:
            timeOfDay_temp = torch.cat([timeOfDay, timeOfDay], 0)
            timeOfDay = torch.cat([timeOfDay, timeOfDay_temp], 0)
        dataset['timeOfDay'] = timeOfDay

        dayOfWeek = torch.FloatTensor(dayOfWeek)
        dayOfWeek = standardization(dayOfWeek, dataset['dayOfWeek_mean'], dataset['dayOfWeek_std'])

        if self.augumentation:
            dayOfWeek_temp = torch.cat([dayOfWeek, dayOfWeek], 0)
            dayOfWeek = torch.cat([dayOfWeek, dayOfWeek_temp], 0)
        dataset['dayOfWeek'] = dayOfWeek


        return dataset


class ECGDataLoad(object):
    def __init__(self, path,augumentation=True):
        self.augumentation=augumentation
        self.trainData = self.preprocessing(path + 'trainset/chfdb_chf13_45590.txt', trainData=True)
        self.validData = self.preprocessing(path + 'testset/chfdb_chf13_45590.txt', trainData=False)



    def preprocessing(self, path, trainData=True):
        """ Read, Standardize, Augment """
        seqData1 = []
        seqData2 = []

        dataset={}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    line_splited = line.strip().split(' ')
                    line_splited = [x for x in line_splited if x != '']
                    #print(line_splited)
                    seqData1.append(float(line_splited[1].strip()))
                    seqData2.append(float(line_splited[2].strip()))


        if trainData:
            dataset['seqData1_mean'] = np.mean(seqData1)
            dataset['seqData1_std']= np.std(seqData1, ddof=1)
            dataset['seqData2_mean'] = np.mean(seqData2)
            dataset['seqData2_std'] = np.std(seqData2, ddof=1)

            #print 'seqData_mean =', seqData_mean
            #print 'seqData_std =', seqData_std
        else:
            dataset['seqData1_mean'] = self.trainData['seqData1_mean']
            dataset['seqData1_std'] = self.trainData['seqData1_std']
            dataset['seqData2_mean'] = self.trainData['seqData2_mean']
            dataset['seqData2_std'] = self.trainData['seqData2_std']

        seqData1 = torch.FloatTensor(seqData1)
        seqData1_original = seqData1.clone()
        seqData2 = torch.FloatTensor(seqData2)
        seqData2_original = seqData2.clone()
        seq_length = len(seqData1)
        if self.augumentation:
            noise_ratio=0.2
            for i in np.arange(0,noise_ratio,0.01):
                tempSeq1 = noise_ratio*dataset['seqData1_std']* torch.randn(seq_length)
                seqData1 = torch.cat([seqData1,seqData1_original+tempSeq1])
                tempSeq2 = noise_ratio * dataset['seqData2_std'] * torch.randn(seq_length)
                seqData2 = torch.cat([seqData2, seqData2_original + tempSeq2])

        seqData1 = standardization(seqData1,dataset['seqData1_mean'],dataset['seqData1_std'])
        seqData2 = standardization(seqData2,dataset['seqData2_mean'],dataset['seqData2_std'])

        dataset['seqData1'] = seqData1
        dataset['seqData2'] = seqData2


        return dataset

