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

class Corpus_ky(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
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
        self.tempCount=0
        self.timeDiff_sorted = [[], []]  # for <pad>, <eos>
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

        prevTime = datetime.datetime.today()
        g = glob.glob(path)
        for file in g:
            with open(file, 'r') as f:
                words=[]
                for i, line in enumerate(f):
                    #print line.strip().split(' ')[2]

                    try:
                        time_diff, prevTime = self.getTimeDifference(line.strip().split(' '), prevTime)
                    except:
                        print 'file reading error from:',file
                        shutil.move(file,os.path.dirname(file)+'/../wrongFormatset/'+os.path.basename(file))
                        break
                    if i==0:
                        time_diff=0.0
                    word = line.strip().split(' ')[2]
                    words += [word]

                    if word not in self.dictionary.word2idx:
                        self.dictionary.add_word(word)
                        self.timeDiff_sorted.append([])

                    idx= self.dictionary.word2idx[word]
                    try:
                        self.timeDiff_sorted[idx].append(time_diff)
                    except:
                        print idx

                words += ['<pad>']
                idx = self.dictionary.word2idx['<pad>']
                self.timeDiff_sorted[idx].append(0.0)
                tokens += len(words)

        #print len(self.timeDiff_sorted)
        #print len(self.dictionary.idx2word)

        timeDiff_means=[]
        timeDiff_stds=[]
        for chunk in self.timeDiff_sorted:
            timeDiff_means.append(np.mean(chunk))
            timeDiff_stds.append(np.std(chunk))

        '''
        #self.timeDiff_sorted=[]
        # Tokenize file content
        indices = torch.LongTensor(tokens)
        print tokens
        token=0

        g = glob.glob(path)
        for file in g:
            with open(file, 'r') as f:
                words = []
                for i,line in enumerate(f):
                    if line:
                        words += [line.strip().split(' ')[2]]
                        time_diff, prevTime = self.getTimeDifference(line.strip().split(' '), prevTime)
                        if time_diff<0:
                            self.timeDiff_sorted.append(0)
                        elif time_diff>60: # if time_dif is longer than 10 minute
                            #print 'time_diff max value is 10 minute! | time_diff {:10.2f} minute | protocol {}'.format(time_diff/60, line.strip().split(' ')[2])
                            time_diff=60
                            self.timeDiff_sorted.append((time_diff - timeDiff_mean) / timeDiff_std)
                        elif i==0: #or time_diff>600:
                            self.timeDiff_sorted.append(0.0)
                        else:
                            self.timeDiff_sorted.append((time_diff - timeDiff_mean) / timeDiff_std)

                        prevLine=line


                words += ['<pad>']
                self.timeDiff_sorted.append(0.0)

                for word in words:
                    indices[token] = self.dictionary.word2idx[word]
                    token += 1
        #print self.timeDiff_sorted

        print 'timeDiff_maxValue_normalized=', max(self.timeDiff_sorted)
        print 'timeDiff_minValue normalized=', min(self.timeDiff_sorted)
        self.timeDiff_sorted = torch.FloatTensor(self.timeDiff_sorted)

        '''
        return 0.0, 0.0, timeDiff_means, timeDiff_stds
