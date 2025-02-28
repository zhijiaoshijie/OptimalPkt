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


    pkt_idx = 0
    for file_path in Config.file_paths_zip:
        pkt_idx = preprocess_file(file_path, pkt_idx, fftflag=True)
