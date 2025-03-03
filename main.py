import sys
import time
import csv
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


from work import *
from reader import *
from mainwork import mainwork
# read packets from file
if __name__ == "__main__":
    if not os.path.exists(Config.outfolder): os.makedirs(Config.outfolder)

    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    tstandard = cp.arange(Config.nsamp) / Config.fs
    refchirp = cp.exp(-1j * 2 * cp.pi * (-Config.bw * 0.5 * tstandard + 0.5 * beta * tstandard * tstandard))
    dpath = "/data/djl/datasets/msudata_ljk_cut/farm/cover_sf10/groundtruth"
    for fname in os.listdir(dpath):
        for fname2 in os.listdir(os.path.join(dpath, fname)):
            rawdata = cp.fromfile(os.path.join(dpath, fname, fname2))
            data1 = cp.matmul(Config.decode_matrix_a, rawdata)
            data2 = cp.matmul(Config.decode_matrix_b, rawdata)
            vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
            coderet = cp.argmax(vals).item()
            pltfig1(None, vals, title=os.path.join('farm/cover_sf10/groundtruth', fname, fname2), addvline=(int(fname2.split('_')[1]),)).show()
            sys.exit(0)


