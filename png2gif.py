import imageio
import glob
import os
images = []
filenames = glob.glob('./result/nyc_taxi/*')
filenames.sort(key=os.path.getmtime)
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./result/nyc_taxi/fig.gif',images,duration=0.1)

