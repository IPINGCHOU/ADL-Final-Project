import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rw", type=str, default="rw.npy")
parser.add_argument("--loss", type=str, default="loss.npy")
opt = parser.parse_args()

def moving_average(data, window=10):
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

rw = np.load(opt.rw)
loss = np.load(opt.loss)

rw = moving_average(rw, 100)
loss = moving_average(loss, 100)
#rw = rw[:8000]
#loss = loss[:8000]
x = [i*10 for i in np.arange(len(rw))]

plt.plot(x, rw)
plt.title("Reward")
plt.savefig("rw.png")
plt.clf()


plt.plot(x, loss)
plt.title("Loss")
plt.savefig("loss.png")
plt.clf()

