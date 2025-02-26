import time
import csv
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


from work import *
from reader import *

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
    for file_path in Config.file_paths_zip:
        thresh = preprocess_file(file_path)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=Config.total_len)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1 = pkt_data

            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == 0: continue
            # if read_idx!=12764: continue
            # data1.tofile(os.path.join(Config.outpath, "data1.sigdat"))

            nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
            logger.warning(f"Prework {pkt_idx=} {len(data1)/nsamp_small=} {cp.mean(cp.abs(data1))=}")

            # <<< PLOT WHOLE DATA1 TO SEE LENGTH OF PREAMBLE AND PAYLOAD >>>
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=np.arange(len(data1)) / nsamp_small, y=tocpu(cp.unwrap(cp.angle(data1)))))
            # for i in range(math.ceil(len(data1) / nsamp_small)): fig.add_vline(x=i)
            # fig.show()
            # y_values = cp.asnumpy(cp.unwrap(cp.angle(data1)))
            # x_values = np.arange(len(data1)) / nsamp_small
            # plt.plot(x_values, y_values)
            # for i in range(math.ceil(len(data1) / nsamp_small)):
            #     plt.axvline(x=i, color='r', linestyle='--', linewidth=1)
            # plt.title('Plot with Vertical Lines')
            # plt.show()

            est_cfo_f = Config.guess_f
            est_to_s = 0
            trytimes = 2
            vals = np.zeros((trytimes, 3))
            # iterate trytimes times to detect, each time based on estimations of the last time
            for tryi in range(trytimes):

                    # main detection function with up-down
                    est_cfo_f, est_to_s, retval = coarse_work_fast(data1, est_cfo_f, est_to_s, False)# tryi >= 1)
                    pktlen = int((len(data1) - est_to_s) / Config.nsampf - 0.25)
                    est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                    logger.warning(f"coarse_work_fast() end: {Config.sf=} {pkt_idx=} inputf={est_cfo_f=} {est_to_s=} {read_idx=}")

                    if est_to_s < 0:
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {est_cfo_f=} {est_to_s=} {file_path=} {pkt_idx=}")
                        est_to_s = 0
                        break
            est_to_s, flag = find_power(est_cfo_f, est_to_s, data1)
            # if not flag: continue ## !!!TODO debug
            est_cfo_f, est_to_s = refine_ft(est_cfo_f, est_to_s, data1)
            # showfit(est_cfo_f, est_to_s, data1, 0)
            # showfit(est_cfo_f, est_to_s, data1, 7)
            # showfit(est_cfo_f, est_to_s, data1, 8)
            # showfit(est_cfo_f, est_to_s, data1, 9)
            # showfit(est_cfo_f, est_to_s, data1, 10)
            # showfit(est_cfo_f, est_to_s, data1, 11)
            # showfit(est_cfo_f, est_to_s, data1, 12)
            # showfit(est_cfo_f, est_to_s, data1, 13)
            # showpower(est_cfo_f, est_to_s, data1, "PLT")
            # codes1 = objective_decode(est_cfo_f, est_to_s, data1)
            # logger.warning(est_cfo_f"ours {codes1=}")
            # codes2 = objective_decode_baseline(est_cfo_f, est_to_s, data1)
            # logger.warning(est_cfo_f"base {codes2=}")
            # logger.warning(est_cfo_f"codes1 and codes2 acc: {sum(1 for a, b in zip(codes1, codes2) if a == b)/len(codes1)}")

            # continue # <<< FIRST CONTINUE HERE TO MAKE SURE PAYLOAD LEN IS CORRECT AND CAN DECODE >>>
            objective_cut(est_cfo_f, est_to_s, data1, pkt_idx_cnt)
            pkt_idx_cnt += 1
