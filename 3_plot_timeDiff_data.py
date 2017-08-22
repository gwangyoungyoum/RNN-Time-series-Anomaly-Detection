import data_v2
import matplotlib.pyplot as plt
import numpy as np
corpus = data_v2.Corpus_ky_timeDifference('dataset/ky/')
timeDiff_total=[]
for i, chunk in enumerate(corpus.timeDiff_sorted[2:]):
    timeDiff_total = timeDiff_total+chunk
timeDiff_total.sort(reverse=True)
plt.hist(timeDiff_total[130:-100], alpha=0.5, bins=250)
#plt.xscale('log', nonposy='clip')
plt.yscale('log', nonposy='clip')
plt.title('Histogram of Elapsed Time of Protocols',fontsize=20)
plt.xlabel('Elapsed time(sec) of protocol',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.savefig('result/ky/fig/histogram_total.pdf')
plt.clf()
#plt.show()
#plt.pause(0)

for i, chunk in enumerate(corpus.timeDiff_sorted[2:]):
    try:
        protocol = corpus.dictionary.idx2word[i+2]
        chunk.sort()
        plt.hist(chunk[2:-2],alpha=0.5,bins=100)
        #plt.xscale('log', nonposy='clip')
        plt.yscale('log', nonposy='clip')
        plt.title('Histogram of Elapsed Time, Protocol ='+protocol+')',fontsize=17)
        plt.xlabel('Elapsed time(sec)',fontsize=14)
        plt.ylabel('Frequency',fontsize=14)
        plt.savefig('result/ky/fig/histogram_'+str(i)+'.pdf')
        plt.clf()
        #plt.show()
        #plt.pause(0)
    except:
        print i, protocol