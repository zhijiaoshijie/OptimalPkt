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
    logger.warning("ERR this only work for integer")



    for file_path in Config.file_paths_zip:

        current_sequence1 = []

        read_idx = -1
        data2s = cp.zeros((Config.preamble_len, Config.nsamp), dtype=cp.float32)
        data3s = []
        data4s = []
        data5s = []
        for rawdata1 in read_large_file(file_path):
            read_idx += 1
            if read_idx > 500: break
            beta = Config.bw / ((2 ** Config.sf) / Config.bw)
            tstandard = cp.arange(Config.nsamp) / Config.fs
            refchirp = cp.exp(-1j * 2 * cp.pi * (-Config.bw * 0.5 * tstandard + 0.5 * beta * tstandard * tstandard))
            data2 = rawdata1 * refchirp
            data3 = cp.abs(myfft(data2, Config.nsamp, Config.plan2))
            data3a = cp.convolve(data3, cp.array([1,1,1]), mode='same')
            data2s[read_idx % 8] = data3a
            datav = cp.sum(data2s, axis=0)
            data3s.append(cp.argmax(datav).item())
            data4s.append(cp.max(datav).item())
            data5s.append(cp.argmax(data3a).item())
        pltfig1(None, data3s, title="argmax").show()
        pltfig1(None, data4s, title="powers").show()
        pltfig1(None, data5s, title="argmax any").show()
        sys.exit(0)

        thresh = preprocess_file(file_path)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        pbar = tqdm(total=os.path.getsize(file_path), unit='B', unit_scale=True, disable=True)
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=Config.total_len)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1 = pkt_data

            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == 0: continue
            # if read_idx!=12764: continue
            # data1.tofile(os.path.join(Config.outpath, "data1.sigdat"))

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



            beta = Config.bw / ((2 ** Config.sf) / Config.bw)
            x = cp.arange(Config.nsamp)
            upchirp = cp.exp(2j * cp.pi * (beta / 2 * x ** 2 / Config.fs ** 2 + (- Config.bw / 2) * x / Config.fs))
            downchirp = cp.conj(upchirp)
            for pidx in range(0, Config.preamble_len + Config.detect_range_pkts):
                start_pos = round(Config.nsamp * pidx + est_to_s)
                sig1 = data1[start_pos: Config.nsamp + start_pos]
                sig2 = sig1 * downchirp

                data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
                Config.fft_ups_x[pidx] = data0
            print(cp.max(cp.abs(Config.fft_ups_x), axis=1))
            print(cp.argmax(cp.abs(Config.fft_ups_x), axis=1))




            trytimes = 2
            # iterate trytimes times to detect, each time based on estimations of the last time
            for tryi in range(trytimes):

                    # main detection function with up-down
                    est_cfo_f, est_to_s, retval = coarse_work_fast(data1, est_cfo_f, est_to_s, False)# tryi >= 1)
                    pktlen = int((len(data1) - est_to_s) / Config.nsampf - 0.25)
                    est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                    pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} Cw")
                    logger.warning(f"Cw {pkt_idx} f={est_cfo_f} t={est_to_s}")

                    if est_to_s < 0:
                        logger.error(f"ERROR in Coarsework {est_cfo_f=} {est_to_s=} out {est_cfo_f=} {est_to_s=} {file_path=} {pkt_idx=}")
                        est_to_s = 0
                        break
            pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} Fw")
            est_to_s, flag = find_power(est_cfo_f, est_to_s, data1)
            logger.warning(f"Fw {pkt_idx} f={est_cfo_f} t={est_to_s}")
            # if not flag: continue ## !!!TODO debug
            pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} Rw")
            est_cfo_f, est_to_s = refine_ft(est_cfo_f, est_to_s, data1)
            logger.warning(f"Rw {pkt_idx} f={est_cfo_f} t={est_to_s}")
            est_to_s = find_power_new(est_cfo_f, est_to_s, data1)
            logger.warning(f"FF {pkt_idx} f={est_cfo_f} t={est_to_s}")

            est_cfo_fs = cp.arange(10) * 10 + est_cfo_f
            c1 = []
            c2 = []
            for est_cfo_f in est_cfo_fs:
                xx = []
                xx2 = []
                for pidx in range(Config.preamble_len):
                    xx.append(cp.angle(showfit(est_cfo_f, est_to_s, data1, pidx)))
                    xx2.append(cp.abs(showfit(est_cfo_f, est_to_s, data1, pidx)))
                x_values = cp.arange(Config.preamble_len)
                coefficients = cp.polyfit(x_values, cp.array(sqlist(xx)), 1)
                print(coefficients,est_cfo_f)
                c1.append(coefficients[0])
                c2.append(coefficients[1])
            pltfig1(est_cfo_fs, c1).show()
            pltfig1(est_cfo_fs, c2).show()


            sys.exit(0)
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
            pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} Cut")
            objective_cut(est_cfo_f, est_to_s, data1, pkt_idx_cnt)
            pkt_idx_cnt += 1

            pbar.n = read_idx * Config.nsamp * 8
            pbar.last_print_n = read_idx * Config.nsamp * 8
            pbar.update(0)
            pbar.set_description(f"{os.path.basename(file_path)} sf={Config.sf} {pkt_idx_cnt} f={est_cfo_f:.2f} t={est_to_s:.2f} ")
