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

    script_path = __file__
    mod_time = os.path.getmtime(script_path)
    readable_time = time.ctime(mod_time)
    logger.warning(f"Last modified time of the script: {readable_time}")

    fulldata = []
    # Main loop read files
    pkt_idx_cnt = 0
    logger.warning("ERR this only work for integer")

    dpath = "/data/djl/datasets/msudata_ljk/"
    outpath = f"/data/djl/datasets/msudata_ljk_test2/{Config.name}/cover_sf{Config.sf}"
    for fname in os.listdir(dpath):
        if f"{Config.name}-cover-{Config.sf}-" in fname:
            if 'lot-cover-10-4-E63' in fname: continue
            file_path = os.path.join(dpath, fname)
            opath = os.path.join(outpath, fname.split("-")[-1])
            preprocess_file(file_path, opath)
