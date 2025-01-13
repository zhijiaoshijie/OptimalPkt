import time
import csv

from work import *
from reader import *
from newwork import *



# read packets from file
if __name__ == "__main__":
    if not os.path.exists(Config.outfolder): os.makedirs(Config.outfolder)

    script_path = __file__
    mod_time = os.path.getmtime(script_path)
    readable_time = time.ctime(mod_time)
    logger.warning(f"Last modified time of the script: {readable_time}")

    fulldata = []
    for file_path, file_path_id in Config.file_paths_zip:
        thresh = preprocess_file(file_path)
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):
            read_idx, data1, data2 = pkt_data
            if read_idx == 0: continue
            # if pkt_idx < 1: continue

            estf = -40000

            nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
            nwindows = len(data1) / nsymblen
            logger.info(f"Prework {pkt_idx=} {nwindows=} {cp.mean(cp.abs(data1))=} {cp.mean(cp.abs(data2))=}")
            data1 /= cp.mean(cp.abs(data1))
            data2 /= cp.mean(cp.abs(data2)) # TODO remove normalization for production

            estt = coarse_est_f_t(data1, estf, 10)
            assert estt >= 0
            logger.warning(f"Coarse estimate time from {estf=} at window_idx=10: {estt=}")

            start_pidx = start_pidx_pow_detect(data1, estf, estt)
            logger.warning(f"Coarse estimate start pkt from {estf=} {estt=}: {start_pidx=}")
            estt += start_pidx * nsymblen / Config.fs

            logger.warning(f"fixed {estt=} from {start_pidx=}")
            fname = f"coeftpkt_{pkt_idx}_f1.pkl"
            # fname = f"coefout2.pkl"
            if not os.path.exists(fname):
                # coeflist = fitcoef(estf, estt, data1)
                coeflist = fitcoef(estf, estt, data1, fitmethod='1dfit', searchquad=True)
                with open(fname, "wb") as fl: pickle.dump(coeflist, fl)
            else:
                with open(fname, "rb") as fl: coeflist = pickle.load(fl)

            betai = Config.bw / ((2 ** Config.sf) / Config.bw)
            nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
            for pidx in range(0, Config.preamble_len):
                nstart = pidx * nsymblen + estt * Config.fs
                nsymbr = cp.arange(around(nstart) + 1000, around(nstart + nsymblen) - 1000)
                coeflist[pidx, 2] += cp.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coeflist[pidx], tsymbr))))
            pltfig1(None, coeflist[:, 0], title=f"{pkt_idx=} coef0", addhline = [np.pi * betai * (1 + x * estf / Config.sig_freq) for x in range(3)]).show()

            pltfig1(None, coeflist[:, 1], title=f"{pkt_idx=} coef1").show()
            pltfig1(None, coeflist[:, 2], title=f"{pkt_idx=} coef2").show()


            # objective_decode(estf, estt, data1)
            sys.exit(0)

            if True:
                # objective_decode(-41890.277+25, 12802.113, data1)
                f = -40900.000
                t = 72.197+10082
                pktlen = int((len(data1) - t) / Config.nsampf - 0.25)
                if False:
                    pktdata_in = data1
                    yi = gen_refchirp(f, t, 20*Config.nsamp)
                    estf = f
                    nstart = t
                    
                    nsymbr = cp.arange(around(nstart) - 30000, around(nstart) + 10 * Config.nsamp)
                    if True:#pkt_idx==2:
                        fig2 = FigureResampler(go.Figure(layout_title_text=f"mainplt {pkt_idx=} {f=:.3f} {t=:.3f}"))
                        fig2.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(pktdata_in)[nsymbr]))))
                        fig2.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(data2)[nsymbr]))))
                        # fig.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(yi)[nsymbr]))))
                        fig2.add_vline(x=t)
                        fig2.add_vline(x=t + Config.nsampf)
                        fig2.add_vline(x=t + Config.nsampf * 2)
                        fig2.show()
                objective_decode(f, t, data1)
                for pidx in range(pktlen):
                    pidx2 = pidx
                    if pidx > Config.sfdpos: pidx2 += 0.25
                    xrange = cp.arange(around(Config.nsampf * pidx2 + t), around(Config.nsampf * (pidx2 + 1) + t))
                    sig1 = data1[xrange]
                    if pidx % 10 == 0:
                        fig = px.scatter(y=tocpu(cp.unwrap(cp.angle(sig1))), title=f"{pidx=}")
                        fig.show()
                # sys.exit(0)

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

            est_cfo_f = -37661.500
            est_to_s =  0
            trytimes = 2
            vals = np.zeros((trytimes, 3))
            # iterate trytimes times to detect, each time based on estimations of the last time
            for tryi in range(trytimes):

                    # main detection function with up-down
                    f, t, retval = coarse_work_fast(data1, est_cfo_f, est_to_s, False)# tryi >= 1)
                    pktlen = int((len(data1) - t) / Config.nsampf - 0.25)
                    est_cfo_f, est_to_s = f, t
                    # est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                    # logger.warning(f"est f{file_path_id:3d} {est_cfo_f=:.6f} {est_to_s=:.6f} {pkt_idx=:3d} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")

                    if t < 0:
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                        break
            estf = f
            nstart = t
            

            pktdata_in = data1
            if False:#pkt_idx==2:
                yi = gen_refchirp(f, t, len(pktdata_in))
                nsymbr = cp.arange(around(nstart) - 30000, around(nstart) + Config.preamble_len * Config.nsamp + 60000)
                fig2 = FigureResampler(go.Figure(layout_title_text=f"mainplt {pkt_idx=} {f=:.3f} {t=:.3f}"))
                fig2.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(pktdata_in)[nsymbr]))))
                fig2.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(data2)[nsymbr]))))
                # fig.add_trace(go.Scatter(x=nsymbr, y=tocpu(cp.unwrap(cp.angle(yi)[nsymbr]))))
                fig2.add_vline(x=t)
                fig2.add_vline(x=t + Config.nsampf)
                fig2.add_vline(x=t + Config.nsampf * 2)
                fig2.add_vline(x=t + Config.nsampf * (Config.preamble_len + 2))
                fig2.add_vline(x=t + Config.nsampf * (Config.preamble_len + 4.25))
                fig2.show()
            pktlen2 = min(pktlen, int(Config.total_len))
            data_angles=cp.zeros(pktlen2, dtype=cp.complex64)


            # if True:
            #     fig.add_trace(go.Scatter(y=tocpu(cp.angle(data_angles))))
            # plt.plot(tocpu(cp.abs(data_angles)))
            # plt.title("abs")
            # plt.show()


            # objective_decode(f, t, data1)
            # save data for output line
            if t > 0 and abs(f + 38000) < 10000:
                est_cfo_f, est_to_s = f, t
                est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                logger.warning(f"est f{file_path_id:3d} {est_cfo_f=:.6f} {est_to_s=:.6f} {pkt_idx=:3d} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")

                fulldata.append([file_path_id, pkt_idx, est_cfo_f, est_to_s_full , retval,  *(np.angle(np.array(data_angles))), *(np.abs(np.array(data_angles)))])
                if False:
                    sig1 = data1[around(est_to_s): Config.nsamp * math.ceil(Config.total_len) + around(est_to_s)]
                    sig2 = data2[around(est_to_s): Config.nsamp * math.ceil(Config.total_len) + around(est_to_s)]
                    sig1.tofile(os.path.join(Config.outfolder, f"{os.path.basename(file_path)}_pkt_{pkt_idx}"))
                    sig2.tofile(os.path.join(Config.outfolder, f"{os.path.basename(file_path)}_pkt_{pkt_idx}"))
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
            if False:  # !!!!!!
                header = ["fileID", "pktID", "CFO", "Time offset", "Power"]
                header.extend([f"Angle{x}" for x in range(int(Config.total_len))])
                header.extend([f"Abs{x}" for x in range(int(Config.total_len))])
                csv_file_path = 'data_out.csv'
                with open(csv_file_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(header)  # Write the header
                    for row in fulldata:
                        csvwriter.writerow(row)
        # fig.show()
        if False:
            bytes_per_complex = 8
            byte_offset = around(est_to_s_full) * bytes_per_complex
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

