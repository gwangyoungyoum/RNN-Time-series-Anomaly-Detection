import imageio
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='PNG to GIF')
parser.add_argument('--path', type=str, default='./result/nyc_taxi',
                    help='file path ')

args = parser.parse_args()


images = []
filenames = glob.glob(args.path+'/*')

filenames.sort(key=os.path.getmtime)
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(args.path+'/fig.gif',images,duration=0.1)

