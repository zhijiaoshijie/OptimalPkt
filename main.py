import time
import csv
import pickle
from tqdm import tqdm


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
    for file_path, file_path_id in Config.file_paths_zip:
        # if file_path_id < 24: continue
        thresh = preprocess_file(file_path)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        codescl = []
        canglescl = []
        # fig = go.Figure(layout_title_text=f"Angle {file_path_id=}")
        # fig.add_vline(x=Config.preamble_len)
        # fig.add_vline(x=Config.sfdpos + 2)
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path, thresh, min_length=30)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1, data2 = pkt_data
            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == 0: continue
            # plt.plot(tocpu(cp.unwrap(cp.angle(data1[:200000]))))
            # plt.show()
            # plt.plot(tocpu(cp.unwrap(cp.angle(data2[:200000]))))
            # plt.show()
            # if pkt_idx < 1: continue
            # if pkt_idx > 2: break

            # normalization
            nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
            logger.info(f"Prework {pkt_idx=} {len(data1)/nsamp_small=} {cp.mean(cp.abs(data1))=} {cp.mean(cp.abs(data2))=}")
            # data1 /= cp.mean(cp.abs(data1))
            # data2 /= cp.mean(cp.abs(data2))
            # objective_decode(-41890.277+25, 12802.113, data1)
            # continue#sys.exit(0)

            # xval = np.arange(-200, 200, 1) -41890.277+25
            # xval2 = []
            # yval2 = []
            # for x in xval:
            #     ret = objective_decode(x, 12802.113, data1)
            #     if ret:
            #         xval2.append(ret[0])
            #         yval2.append(ret[1])
            # fig = px.scatter(x=xval2, y=yval2)
            # fig.show()
            # sys.exit(0)

            est_cfo_f = -44000
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
                    logger.warning(f"est f{file_path_id:3d} {est_cfo_f=:.6f} {est_to_s=:.6f} {pkt_idx=:3d} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")

                    if t < 0:
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                        break

            codes1 = objective_decode(f, t, data1)
            logger.warning(f"{codes1=}")
            codes2 = objective_decode_baseline(f, t, data1)
            logger.warning(f"{codes2=}")
            reps = 1
            accs = cp.zeros((2, 41, reps), dtype=float)
            snrrange = np.arange(-30, -10, 3)
            pbar = tqdm(total=len(snrrange) * reps)
            for snr in snrrange:
                for rep in range(reps):
                    num_samples = len(data1)
                    amp = math.pow(0.1, snr / 20) * cp.mean(cp.abs(data1[round(len(data1)/4):round(len(data1)*0.75)]))
                    noise = (amp / math.sqrt(2) * cp.random.randn(num_samples) + 1j * amp / math.sqrt(2) * cp.random.randn(num_samples))
                    dataX = data1 + noise  # dataX: data with noise
                    codesx1 = objective_decode(f, t, dataX)
                    codesx2 = objective_decode_baseline(f, t, dataX)
                    accs[0, -snr, rep] = sum(1 for a, b in zip(codesx1, codes1) if a == b) / len(codes1)
                    logger.warning(f"{accs[0, -snr, rep]}")
                    accs[1, -snr, rep] = sum(1 for a, b in zip(codesx2, codes1) if a == b) / len(codes1)
                    logger.warning(f"{accs[1, -snr, rep]}")
                    pbar.update(1)
            accs = cp.mean(accs, axis=2)
            for snr in snrrange:
                logger.warning(f"{pkt_idx=}, {snr=}, {accs[0, -snr]=}, {accs[1, -snr]=}")
                fulldata.append([pkt_idx, snr, accs[0, -snr], accs[1, -snr]])
            pbar.close()

        with open(f"{Config.sf}data.pkl", "wb") as fi: pickle.dump(accs, fi)
        header = ["pktID", "SNR", "ACCours", "ACCbaseline"]
        csv_file_path = f'data_out{Config.sf}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)  # Write the header
            for row in fulldata:
                csvwriter.writerow(row)

        if False:
            bytes_per_complex = 8
            byte_offset = round(est_to_s_full) * bytes_per_complex
            L = Config.nsamp * Config.total_len

            with open(file_path, 'rb') as f:
                # Move the file pointer to the desired byte offset
                f.seek(byte_offset)

                # Read L complex64 numbers (L * 8 bytes)
                data = f.read(L * bytes_per_complex)

                # Ensure that the correct amount of data was read
                if len(data) != L * bytes_per_complex:
                    raise ValueError(f"Expected to read {L * bytes_per_complex} bytes, but got {len(data)} bytes.")

            # Convert the byte data to a CuPy array of type complex64
            rawdata = cp.frombuffer(data, dtype=cp.complex64)
            rawdata.tofile(f"fout/dataZ_test_{file_path_id}_pkt_{pkt_idx}")

        if False:
            fig = go.Figure(layout_title_text=f"decode angles")
            for i in range(len(codescl)):
                codes = codescl[i]
                angdiffs = canglescl[i]
                fig.add_trace(go.Scatter(x=codes, y=angdiffs, mode="markers"))
            # fig.write_html("codeangles.html")
            fig.show()

