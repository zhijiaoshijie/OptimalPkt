import time
import csv
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
    for file_path, file_path_id in Config.file_paths_zip:
        thresh = preprocess_file(file_path)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        codescl = []
        canglescl = []
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1, data2 = pkt_data



            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == 0: continue#len(data1) // Config.nsamp: continue
            # if pkt_idx < 2: continue
            # if pkt_idx > 2: break
            if True:
                data1.tofile(os.path.join(Config.outfolder, f"data_test"))
            if True:
                plt.plot(tocpu(cp.unwrap(cp.angle(data1))))
                plt.show()
                print(read_idx, len(data1)/Config.nsamp, Config.total_len)
                sig1 = data1[Config.nsamp * 3: Config.nsamp * math.ceil(Config.total_len) - 3]
                sig2 = data2[Config.nsamp * 3: Config.nsamp * math.ceil(Config.total_len) - 3]
                sigangle = [tocpu(sig1[Config.nsamp * i: Config.nsamp * (i+1)].dot(cp.conj(sig2[Config.nsamp * i: Config.nsamp * (i+1)])))
                            for i in range(math.ceil(Config.total_len) - 6)]
                plt.plot(np.abs(sigangle)/Config.nsamp)
                plt.show()
                plt.plot(np.angle(sigangle))
                plt.show()
                sys.exit(3)
            # normalization
            data1 /= cp.mean(cp.abs(data1))
            data2 /= cp.mean(cp.abs(data1))

            logger.info(f"Prework {pkt_idx=} {len(data1)=}")
            est_cfo_f = -40000
            est_to_s = 0
            trytimes = 2
            vals = np.zeros((trytimes, 3))
            # iterate trytimes times to detect, each time based on estimations of the last time
            for tryi in range(trytimes):

                    # main detection function with up-down
                    f, t, retval = coarse_work_fast(data1, est_cfo_f, est_to_s,  tryi >= 1)

                    if t < 0:
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                        break

            # compute angles
            if False:
                data_angles = []
                est_cfo_f, est_to_s = f, t
                for pidx in range(10, Config.total_len):
                    sig1 = data1[Config.nsamp * pidx + est_to_s: Config.nsamp * (pidx + 1) + est_to_s]
                    sig2 = data2[Config.nsamp * pidx + est_to_s: Config.nsamp * (pidx + 1) + est_to_s]
                    sigtimes = sig1 * sig2.conj()
                    sigangles = cp.cumsum(sigtimes[::-1])[::-1]
                    fig.add_trace(go.Scatter(y=cp.angle(sigangles).get(), mode="lines"))
                    break
                    # fig.add_trace(go.Scatter(y=cp.unwrap(cp.angle(sig2)).get()))

                    # save data for output line
            if t > 0 and abs(f + 38000) < 8000:
                est_cfo_f, est_to_s = f, t
                est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                logger.warning(f"est f{file_path_id:3d} {est_cfo_f=:.6f} {est_to_s=:.6f} {pkt_idx=:3d} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")
                fulldata.append([file_path_id, pkt_idx, est_cfo_f, est_to_s_full , retval])
                if True:
                    sig1 = data1[round(est_to_s): Config.nsamp * math.ceil(Config.total_len) + round(est_to_s)]
                    sig2 = data2[round(est_to_s): Config.nsamp * math.ceil(Config.total_len) + round(est_to_s)]
                    sig1.tofile(os.path.join(Config.outfolder, f"data0_test_{file_path_id}_pkt_{pkt_idx}"))
                    sig2.tofile(os.path.join(Config.outfolder, f"data1_test_{file_path_id}_pkt_{pkt_idx}"))
                    # logger.warning(f"write {len(sig1)} {round(est_to_s)-Config.nsamp} {Config.nsamp * math.ceil(Config.total_len) + round(est_to_s)}")
            else:
                est_cfo_f, est_to_s = f, t
                logger.error(f"ERR f{file_path_id} {est_cfo_f=} {est_to_s=} {pkt_idx=} {read_idx=} tot {est_to_s + read_idx * Config.nsamp} {retval=}")
            # save data for plotting
            # ps.extend(data_angles)
            # psa1.append(len(ps))
            # ps2.append(est_cfo_f)
            # ps3.append(est_to_s)
            # print("only compute 1 pkt, ending")
            # sys.exit(0)
        # the length of each pkt (for plotting)
                # save info of all the file to csv (done once each packet, overwrite old)
            if True:  # !!!!!!
                header = ["fileID", "pktID", "CFO", "Time offset", "Power"]
                # header.extend([f"Angle{x}" for x in range(Config.total_len)])
                # header.extend([f"Abs{x}" for x in range(Config.total_len)])
                csv_file_path = 'data_out.csv'
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

