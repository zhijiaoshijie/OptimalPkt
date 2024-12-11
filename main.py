import time
from sklearn.mixture import GaussianMixture

from work import *
from reader import *


# read packets from file
if __name__ == "__main__":

    script_path = __file__
    mod_time = os.path.getmtime(script_path)
    readable_time = time.ctime(mod_time)
    logger.info(f"Last modified time of the script: {readable_time}")


    ps = []
    ps2 = []
    psa1 = []
    psa2 = []
    ps3 = []
    fulldata = []

    # Main loop read files
    vfilecnt = 0
    fig = go.Figure()
    for file_path in Config.file_paths:
        # file_path = "/data/djl/temp/OptimalPkt/hou2"

        #  read file and count size
        file_path_id = int(file_path.split('_')[-1])

        logger.info(f"FILEPATH { file_path}")
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
        logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')

        # read max power of first 5000 windows, for envelope detection
        power_eval_len = 5000
        nmaxs = []
        for idx, rawdata in enumerate(read_large_file(file_path)):
            nmaxs.append(cp.max(cp.abs(rawdata)))
            if idx == power_eval_len - 1: break
        nmaxs = cp.array(nmaxs).get()

        # clustering
        data = nmaxs.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data)
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        sorted_indices = np.argsort(means)
        mean1, mean2 = means[sorted_indices]
        covariance1, covariance2 = covariances[sorted_indices]
        weight1, weight2 = weights[sorted_indices]

        # threshold to divide the noise power from signal power
        thresh = (mean1 * covariance2 + mean2 * covariance1) / (covariance1 + covariance2)

        # if threshold may not work set this to True
        # plot the power map
        if False:
            counts, bins = cp.histogram(nmaxs, bins=100)
            # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
            logger.debug(f"cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
            threshpos = np.searchsorted(tocpu(bins), thresh).item()
            logger.debug(f"lower: {cp_str(counts[:threshpos])}")
            logger.debug(f"higher: {cp_str(counts[threshpos:])}")
            fig = px.line(nmaxs.get())
            fig.add_hline(y=thresh)
            fig.update_layout(
                title=f"{file_path} pow {len(nmaxs)}",
                legend=dict(x=0.1, y=1.1))
            fig.show()

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        codescl = []
        canglescl = []
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1, data2 = pkt_data
            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == len(data1) // Config.nsamp: continue
            # if pkt_idx < 2: continue
            # if pkt_idx > 2: break

            # normalization
            data1 /= cp.mean(cp.abs(data1))
            data2 /= cp.mean(cp.abs(data1))

            logger.info(f"Prework {pkt_idx=} {len(data1)=}")
            est_cfo_f = 0
            est_to_s = 0
            trytimes = 2
            vals = np.zeros((trytimes, 3))
            # iterate trytimes times to detect, each time based on estimations of the last time
            for tryi in range(trytimes):

                    # main detection function with up-down
                    f, t, _ = coarse_work_fast(data1, est_cfo_f, est_to_s,  tryi >= 1)

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
                    if False:
                        est_cfo_f, est_to_s = f, t
                        logger.info(f"EST {est_cfo_f=} {est_to_s=}")
                        fulldata.append([file_path_id, est_cfo_f, est_to_s])
                        if est_to_s > 0:
                            sig1 = data1[round(est_to_s): Config.nsamp * (Config.total_len + Config.sfdend) + round(est_to_s)]
                            sig2 = data2[round(est_to_s): Config.nsamp * (Config.total_len + Config.sfdend) + round(est_to_s)]
                            sig1.tofile(f"fout/data0_test_{file_path_id}_pkt_{pkt_idx}")
                            sig2.tofile(f"fout/data1_test_{file_path_id}_pkt_{pkt_idx}")
                    # save data for plotting
                    # ps.extend(data_angles)
                    # psa1.append(len(ps))
                    # ps2.append(est_cfo_f)
                    # ps3.append(est_to_s)
            # print("only compute 1 pkt, ending")
            # sys.exit(0)
        # the length of each pkt (for plotting)

        if False:
            fig = go.Figure(layout_title_text=f"decode angles")
            for i in range(len(codescl)):
                codes = codescl[i]
                angdiffs = canglescl[i]
                fig.add_trace(go.Scatter(x=codes, y=angdiffs, mode="markers"))
            # fig.write_html("codeangles.html")
            fig.show()

        psa1 = psa1[:-1]
        psa2.append(len(ps))

        # save info of all the file to csv (done once each packet, overwrite old)
        if False: # !!!!!!
            header = ["fileID", "CFO", "Time offset"]
            # header.extend([f"Angle{x}" for x in range(Config.total_len)])
            # header.extend([f"Abs{x}" for x in range(Config.total_len)])
            csv_file_path = 'data_out.csv'
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)  # Write the header
                for row in fulldata:
                    csvwriter.writerow(row)
