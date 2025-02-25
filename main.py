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

            nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
            logger.info(f"Prework {pkt_idx=} {len(data1)/nsamp_small=} {cp.mean(cp.abs(data1))=}")

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
                    f, t, retval = coarse_work_fast(data1, est_cfo_f, est_to_s, False)# tryi >= 1)
                    pktlen = int((len(data1) - t) / Config.nsampf - 0.25)
                    est_cfo_f, est_to_s = f, t
                    est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                    logger.warning(f"coarse_work_fast() end: {Config.sf=} {pkt_idx=:3d} inputf={est_cfo_f=:.6f} {est_to_s=:.6f} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")

                    if t < 0:
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                        break
            t, flag = find_power(f, t, data1)
            if not flag: continue
            f, t = refine_ft(f, t, data1)
            # showpower(f, t, data1, "PLT")
            codes1,freqs,phases = objective_decode(f, t, data1)
            logger.warning(f"ours {codes1=}")
            codes2,_,_ = objective_decode_baseline(f, t, data1)
            logger.warning(f"base {codes2=}")
            logger.warning(f"codes1 and codes2 acc: {sum(1 for a, b in zip(codes1, codes2) if a == b)/len(codes1)}")
            # continue # <<< FIRST CONTINUE HERE TO MAKE SURE PAYLOAD LEN IS CORRECT AND CAN DECODE >>>
            # objective_cut(f, t, data1, pkt_idx_cnt)
            # pkt_idx_cnt += 1

            reps = 1

            snrrange = np.arange(-40, 10, 1)
            accs = cp.zeros((2, len(snrrange), reps), dtype=float)
            fccs = cp.zeros((2, len(snrrange), reps), dtype=float)
            pccs = cp.zeros((2, len(snrrange), reps), dtype=float)

            pbar = tqdm(total=len(snrrange) * reps)
            for snridx, snr in enumerate(snrrange):
                for rep in range(reps):
                    amp = math.pow(0.1, snr / 20) * cp.mean(
                        cp.abs(data1[around(len(data1) / 4):around(len(data1) * 0.75)]))
                    noise = (amp / math.sqrt(2) * cp.random.randn(len(data1)) + 1j * amp / math.sqrt(
                        2) * cp.random.randn(len(data1)))
                    dataX = data1 + noise  # dataX: data with noise
                    codesx1,freqs1,phases1 = objective_decode(f, t, dataX)
                    codesx2,freqs2,phases2 = objective_decode_baseline(f, t, dataX)
                    accs[0, snridx, rep] = sum(1 for a, b in zip(codesx1, codes1) if a == b) / len(codes1)
                    accs[1, snridx, rep] = sum(1 for a, b in zip(codesx2, codes1) if a == b) / len(codes1)
                    fccs[0, snridx, rep] = cp.mean(freqs1).item()
                    fccs[1, snridx, rep] = cp.mean(freqs2).item()
                    pccs[0, snridx, rep] = cp.mean(cp.abs(wrap(phases1-phases))).item()
                    pccs[1, snridx, rep] = cp.mean(cp.abs(wrap(phases2-phases))).item()

                    pbar.update(1)
                    #logger.warning(f"{snr=} {accs[0, snridx, rep]} {accs[1, snridx, rep]} {fccs[0, snridx, rep]} {fccs[1, snridx, rep]} {pccs[0, snridx, rep]} {pccs[1, snridx, rep]}")
            accs = cp.mean(accs, axis=2)

            for snridx, snr in enumerate(snrrange):
                if pkt_idx == 1: logger.warning(f"{pkt_idx=}, {snr=}, {accs[0, snridx]=}, {accs[1, snridx]=}")
                fulldata.append([pkt_idx, snr, accs[0, snridx], accs[1, snridx], fccs[0, snridx], fccs[1, snridx], pccs[0, snridx], pccs[1, snridx]])
            pbar.close()

            with open(f"{Config.sf}data_no_dt.pkl", "wb") as fi:
                pickle.dump(accs, fi)
            header = ["pktID", "SNR", "ACCOurs", "ACCBaseline", "FreqErrOurs", "FreqErrBaseline", "PhaseErrOurs", "PhaseErrBaseline"]
            csv_file_path = f'data_out_no_dt_{Config.sf}.csv'
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)  # Write the header
                for row in fulldata:
                    csvwriter.writerow(row)
