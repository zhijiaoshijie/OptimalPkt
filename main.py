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



    for file_path in Config.file_paths_zip:

        thresh = preprocess_file(file_path, fftflag=True, draw=True, thresh_manual=0.54)
        sys.exit(0)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        pbar = tqdm(total=os.path.getsize(file_path), unit='B', unit_scale=True, disable=False)
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=Config.total_len)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1 = pkt_data

            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == 0: continue
            # if read_idx!=12764: continue
            # data1.tofile(os.path.join(Config.outpath, "data1.sigdat"))
            est_cfo_f, est_to_s = mainwork(pkt_idx, data1)

            pkt_idx_cnt += 1

            pbar.n = read_idx * Config.nsamp * 8
            pbar.last_print_n = read_idx * Config.nsamp * 8
            pbar.update(0)
            pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} ")
