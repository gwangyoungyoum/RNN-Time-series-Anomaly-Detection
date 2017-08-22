import os
import torch
import glob
import datetime
import numpy as np
import shutil

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.add_word('<pad>')
        self.add_word('<eos>')


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        #self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

        self.train_protocol = self.tokenize(path + 'ptb.train.txt')
        self.valid_protocol = self.tokenize(path + 'ptb.valid.txt')
        self.test_protocol = self.tokenize(path + 'ptb.test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        print
        #assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:

                words = line.split() + ['<eos>']
                print words
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Corpus_ky(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        #self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

        self.train = self.tokenize(path + 'trainset/*')
        self.valid = self.tokenize(path +'validset/*')
        self.test = self.tokenize(path + 'validset/*')

    def tokenize(self, path):
        """Tokenizes a text file."""
        #assert os.path.exists(path)
        # Add words to the dictionary
        tokens = 0
        g = glob.glob(path)
        for file in g:
            with open(file, 'r') as f:
                words=[]
                for line in f:
                    #print line.strip().split(' ')[2]
                    words += [line.strip().split(' ')[2]]
                words += ['<pad>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token=0
        g = glob.glob(path)
        for file in g:
            with open(file, 'r') as f:
                words = []
                for line in f:
                    words += [line.strip().split(' ')[2]]
                words += ['<pad>']

                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Corpus_ky_timeDifference(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        #self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

        self.train_protocol, self.train_timeDiff, self.train_timeDiff_mean, self.train_timeDiff_std = self.tokenize(path + 'trainset/*',trainData=True)
        self.valid_protocol, self.valid_timeDiff, self.valid_timeDiff_mean, self.valid_timeDiff_std  = self.tokenize(path +'validset/*',trainData=False)
        self.test_protocol, self.test_timeDiff, self.test_timeDiff_mean, self.test_timeDiff_std  = self.tokenize(path + 'validset/*',trainData=False)

    def getTimeDifference(self,line_splited, prevTime):


        line_splited[0] = line_splited[0][1:]
        line_date = [line_splited[0][:4], line_splited[0][4:6], line_splited[0][6:]]
        line_splited[1] = line_splited[1][:len(line_splited[1]) - 2]
        line_time = line_splited[1].split(':')
        nowTime = datetime.datetime(int(line_date[0]), int(line_date[1]), int(line_date[2]),
                                    int(line_time[0]), int(line_time[1]), int(line_time[2]), int(line_time[3]))

        time_diff = (nowTime - prevTime).total_seconds() #* 1000


        return time_diff, nowTime

    def tokenize(self, path,trainData=True):
        """Tokenizes a text file."""
        #assert os.path.exists(path)
        # Add words to the dictionary
        tokens = 0
        timeDifferences = []
        prevTime = datetime.datetime.today()
        g = glob.glob(path)
        ErrorFileCount=0
        TimeFixedFileCount=0
        maxValue=0
        for file in g:
            with open(file, 'r') as f:
                words=[]
                for i, line in enumerate(f):
                    #print line.strip().split(' ')[2]

                    try:
                        time_diff, prevTime = self.getTimeDifference(line.strip().split(' '), prevTime)
                    except:
                        print 'file reading error from:',file
                        ErrorFileCount += 1
                        shutil.move(file,os.path.dirname(file)+'/../wrongFormatset/'+os.path.basename(file))
                        break
                    if i==0:
                        time_diff=0.0

                    words += [line.strip().split(' ')[2]]
                    #print time_diff, line.strip().split(' ')[2]
                    if 'operror' in line.strip().split(' ')[2]:
                        print file
                        print time_diff, line.strip().split(' ')[2]
                        ErrorFileCount+=1
                        shutil.move(file, os.path.dirname(file) + '/../anomalyset/' + os.path.basename(file))
                        break
                        #raw_input()
                    elif 'OPERROR' in line.strip().split(' ')[2]:
                        print file
                        print time_diff, line.strip().split(' ')[2]
                        shutil.move(file, os.path.dirname(file) + '/../anomalyset/' + os.path.basename(file))
                        ErrorFileCount += 1
                        break

                    '''
                    if time_diff>60: # if time_dif is longer than 10 minute
                        print 'time_diff max value is 1 minute! | time_diff {:10.2f} minute | protocol {}'.format(time_diff/60, line.strip().split(' ')[2])
                        time_diff = 60
                        TimeFixedFileCount +=1
                    '''
                    timeDifferences.append(time_diff)
                words += ['<pad>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        if trainData:
            print 'timeDiff_maxValue=', max(timeDifferences)
            timeDiff_mean = np.mean(timeDifferences)
            print 'timeDiff_mean =', timeDiff_mean
            timeDiff_std = np.std(timeDifferences, ddof=1)
            print 'timeDiff_std =', timeDiff_std
        else:
            timeDiff_mean=self.train_timeDiff_mean
            timeDiff_std=self.train_timeDiff_std

        print 'Error file count= ',ErrorFileCount
        print 'Time fixed point count=',TimeFixedFileCount

        timeDifferences=[]
        # Tokenize file content
        indices = torch.LongTensor(tokens)
        print tokens
        token=0
        prevLine=' '
        g = glob.glob(path)
        for file in g:
            with open(file, 'r') as f:
                words = []
                for i,line in enumerate(f):
                    if line:
                        words += [line.strip().split(' ')[2]]
                        time_diff, prevTime = self.getTimeDifference(line.strip().split(' '), prevTime)
                        if time_diff<0:
                            timeDifferences.append(0)
                        elif time_diff>60: # if time_dif is longer than 10 minute
                            #print 'time_diff max value is 10 minute! | time_diff {:10.2f} minute | protocol {}'.format(time_diff/60, line.strip().split(' ')[2])
                            time_diff=60
                            timeDifferences.append((time_diff - timeDiff_mean) / timeDiff_std)
                        elif i==0: #or time_diff>600:
                            timeDifferences.append(0.0)
                        else:
                            timeDifferences.append((time_diff - timeDiff_mean) / timeDiff_std)

                        prevLine=line


                words += ['<pad>']
                timeDifferences.append(0.0)

                for word in words:
                    indices[token] = self.dictionary.word2idx[word]
                    token += 1
        #print timeDifferences

        print 'timeDiff_maxValue_normalized=', max(timeDifferences)
        print 'timeDiff_minValue normalized=', min(timeDifferences)
        timeDifferences = torch.FloatTensor(timeDifferences)

        return indices, timeDifferences, timeDiff_mean, timeDiff_std


class Corpus_nyc_taxi(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        #self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

        self.train_timeDiff, self.train_timeDiff_mean, self.train_timeDiff_std = self.tokenize(path + 'trainset/nyc_taxi.csv',trainData=True)
        self.valid_timeDiff, self.valid_timeDiff_mean, self.valid_timeDiff_std  = self.tokenize(path +'validset/nyc_taxi.csv',trainData=False)
        self.test_timeDiff, self.test_timeDiff_mean, self.test_timeDiff_std  = self.tokenize(path + 'validset/nyc_taxi.csv',trainData=False)



    def tokenize(self, path,trainData=True):
        """Tokenizes a text file."""
        timeDifferences = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1]))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.1))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.2))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.3))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.4))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.5))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.6))
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i>0:
                    line_splited = line.strip().split(',')
                    print line_splited
                    timeDifferences.append(float(line_splited[1])+np.random.normal(0,0.7))

        if trainData:
            timeDiff_mean = np.mean(timeDifferences)
            print 'timeDiff_mean =', timeDiff_mean
            timeDiff_std = np.std(timeDifferences, ddof=1)
            print 'timeDiff_std =', timeDiff_std
        else:
            timeDiff_mean=self.train_timeDiff_mean
            timeDiff_std=self.train_timeDiff_std

        timeDifferences = torch.FloatTensor(timeDifferences)
        timeDifferences = (timeDifferences-timeDiff_mean) / timeDiff_std

        timeDifferences_noise1 = timeDifferences + 0.1*torch.randn(timeDifferences.size(0))
        timeDifferences_noise2 = timeDifferences + 0.2*torch.randn(timeDifferences.size(0))
        timeDifferences = torch.cat([timeDifferences,timeDifferences_noise1],0)
        timeDifferences = torch.cat([timeDifferences, timeDifferences_noise2], 0)

        return timeDifferences, timeDiff_mean, timeDiff_std
