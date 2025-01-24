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

            estf= -40971.948630148894
            estt =  0.01015242
            # coeflist = fitcoef(estf, estt, data1, margin=10, fitmethod='1dfit', searchquad=False)
            with open('1dfittemp.pkl', "rb") as fl: coeflist = pickle.load(fl)
            for pidx in range(Config.preamble_len - 1):
                a1 = []#np.zeros(20001, dtype=np.float64)
                margin = 1000
                estcoef = [0.01008263, 0.01015365]
                nestt = estt * Config.fs
                nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
                logger.warning(f"{nsymblen / Config.fs=} {estcoef[0]=} TT")
                nstart = (pidx) * nsymblen + nestt
                tstart = nstart / Config.fs
                nstart2 = (pidx + 1) * nsymblen + nestt
                tstart2 = nstart2 / Config.fs
                nstart3 = (pidx + 2) * nsymblen + nestt
                tstart3 = nstart3 / Config.fs

                x1 = []#np.zeros(20001, dtype=np.float64)
                for i in range(margin, 10001):
                    coefa = coeflist[pidx]
                    coefb = coeflist[pidx + 1]
                    xv1 = np.arange(around(tstart2 * Config.fs - i), around(tstart2 * Config.fs - i + margin), dtype=int)
                    a1v = cp.angle(data1[xv1].dot(cp.exp(-1j * cp.polyval(coefa, xv1 / Config.fs))))
                    # a1[10000 - i] = a1v
                    # x1[10000 - i] = around(tstart2 * Config.fs - i) / Config.fs
                    a1.append(a1v)
                    x1.append(around(tstart2 * Config.fs - i) / Config.fs)
                    xv1 = np.arange(around(tstart2 * Config.fs + i - margin), around(tstart2 * Config.fs + i), dtype=int)
                    a1v = cp.angle(data1[xv1].dot(cp.exp(-1j * cp.polyval(coefb, xv1 / Config.fs))))
                    a1.append(a1v)
                    x1.append(around(tstart2 * Config.fs + i) / Config.fs)
                a1 = togpu(cp.array(a1))
                x1 = togpu(cp.array(x1))
                pltfig1(x1, a1, addvline=(tstart2, tstart, tstart3), mode='markers', title="angle difference").show()

            estt, estf = symbtime(estf, estt, data1, coeflist)
            # with open('1dfittemp.pkl', "wb") as fl: pickle.dump(coeflist, fl)
            logger.warning(f"symbtime end: {estt=} {estf=}")
            sys.exit(0)

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
            fit1d = True
            if fit1d: fname = f"coeftpkt_{pkt_idx}_f1.pkl"
            else: fname = f"coeftpkt_{pkt_idx}_nf.pkl"
            # fname = f"coefout2.pkl"
            if not os.path.exists(fname):
                if fit1d: coeflist = fitcoef(estf, estt, data1, margin=1000,fitmethod='1dfit', searchquad=False)
                else: coeflist = fitcoef(estf, estt, data1, margin=1000, fitmethod='2dfit', searchquad=True)
                with open(fname, "wb") as fl: pickle.dump(coeflist, fl)
            else:
                with open(fname, "rb") as fl: coeflist = pickle.load(fl)

            # show_fit_results(data1, estf, estt, coeflist, pkt_idx)

            estt, estf = symbtime(estf, estt, data1, coeflist)
            logger.warning(f"symbtime end: {estt=} {estf=}")

            fit1d = True
            if fit1d:
                fname = f"coeftpktA_{pkt_idx}_f1.pkl"
            else:
                fname = f"coeftpktA_{pkt_idx}_nf.pkl"
            # fname = f"coefout2.pkl"
            if not os.path.exists(fname):
                if fit1d:
                    coeflist = fitcoef(estf, estt, data1,margin=10, fitmethod='1dfit', searchquad=True)
                else:
                    coeflist = fitcoef(estf, estt, data1, margin=10)
                with open(fname, "wb") as fl:
                    pickle.dump(coeflist, fl)
            else:
                with open(fname, "rb") as fl:
                    coeflist = pickle.load(fl)
            estt, estf = symbtime(estf, estt, data1, coeflist, margin=10, draw=True)
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

