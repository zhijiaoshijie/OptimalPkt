import os.path
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

    for Config.name in ['farm', 'lot']:
        dpath = f"/data/djl/datasets/msudata_ljk_cut_old/{Config.name}/clean_data/sf{Config.sf}"
        opath = f"/data/djl/datasets/msudata_ljk_cut/{Config.name}/clean_data/sf{Config.sf}"
        for fname in os.listdir(dpath):
                for fname3 in os.listdir(os.path.join(dpath, fname)):
                    rawdata = cp.fromfile(os.path.join(dpath, fname, fname3))
                    data1 = cp.matmul(Config.decode_matrix_a, rawdata)
                    data2 = cp.matmul(Config.decode_matrix_b, rawdata)
                    vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
                    coderet = cp.argmax(vals).item()
                    pidx, _, pktidx, sf = fname3.split("_")
                    outpath = os.path.join(opath, fname)
                    if not os.path.exists(outpath): os.makedirs(outpath)
                    rawdata.tofile(os.path.join(outpath, f"{pidx}_{str(coderet)}_{pktidx}_{sf}"))
        os.system(f"tar -czf {Config.name}_clean_sf{Config.sf}.tar.gz -C /data/djl/datasets/msudata_ljk_cut/{Config.name}/clean_data sf{Config.sf}")


