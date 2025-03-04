import sys
import time
import csv
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


from work import *
from reader import *
from mainwork import mainwork

def decode_loratrimmer(rawdata, name):
    data1 = cp.matmul(Config.decode_matrix_a, rawdata)
    data2 = cp.matmul(Config.decode_matrix_b, rawdata)
    vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
    coderet = cp.argmax(vals).item()
    amp = cp.zeros(Config.n_classes, dtype=float)
    for i in range(Config.n_classes):
        tsplit = around(Config.nsampf / Config.n_classes * (Config.n_classes - i))
        amp[i] = cp.sum(cp.abs(rawdata[:tsplit])) ** 2 + cp.sum(cp.abs(rawdata[tsplit:])) ** 2
    logger.warning(cp.mean(cp.abs(rawdata)))
    fig = pltfig1(None, vals, title=os.path.join(f'farm/cover_sf10/{name}', fname, fname2) + ' : ' + str(coderet))#, addvline=(int(fname2.split('_')[1]),))
    pltfig1(None, amp, fig = fig).show()
    pltfig1(None, rawdata.imag).show()

# read packets from file
if __name__ == "__main__":
    if not os.path.exists(Config.outfolder): os.makedirs(Config.outfolder)

    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    tstandard = cp.arange(Config.nsamp) / Config.fs
    refchirp = cp.exp(-1j * 2 * cp.pi * (-Config.bw * 0.5 * tstandard + 0.5 * beta * tstandard * tstandard))
    # name = "1mile"
    name = "groundtruth"
    dpath = f"/data/djl/datasets/msudata_ljk_cut/farm/cover_sf10/{name}"
    for fname in os.listdir(dpath):
        for fname2 in os.listdir(os.path.join(dpath, fname))[2:]:
            print(fname2.split("_")[1])
            rawdata = cp.fromfile(os.path.join(dpath, fname, fname2), dtype=cp.complex64)
            decode_loratrimmer(rawdata, name)
            sys.exit(0)



