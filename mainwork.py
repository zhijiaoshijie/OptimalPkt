from work import *

def mainwork(pkt_idx, data1, outpath):
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
    # iterate trytimes times to detect, each time based on estimations of the last time
    for tryi in range(trytimes):

        # main detection function with up-down
        est_cfo_f, est_to_s = coarse_work_fast(data1, est_cfo_f, est_to_s)  # tryi >= 1) # !!! tODO
        logger.info(f"Cw {pkt_idx} f={est_cfo_f} t={est_to_s}")

        if est_to_s < 0:
            logger.error(f"ERROR in Coarsework {est_cfo_f=} {est_to_s=} out {est_cfo_f=} {est_to_s=} {pkt_idx=}")
            est_to_s = 0
    # est_to_s, flag = find_power(est_cfo_f, est_to_s, data1)
    # logger.warning(f"Fw {pkt_idx} f={est_cfo_f} t={est_to_s}")
    # if not flag: continue ## !!!TODO debug
    # est_cfo_f, est_to_s = refine_ft(est_cfo_f, est_to_s, data1)
    # logger.warning(f"Rw {pkt_idx} f={est_cfo_f} t={est_to_s}")
    # est_cfo_f, est_to_s = find_power_new(est_cfo_f, est_to_s, data1)
    # logger.warning(f"FF {pkt_idx} f={est_cfo_f} t={est_to_s}")

    # for totlen in range(Config.total_len - 10, Config.total_len + 20, 2):
    #     data1[around(est_to_s) : around(2 ** Config.sf / Config.bw * Config.fs * (totlen + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s)].tofile(f"out{totlen}")
    # sys.exit(0)
    # data1[around(est_to_s) : around(2 ** Config.sf / Config.bw * Config.fs * (Config.total_len + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s)].tofile(f"out{pkt_idx}.sigdat")

    est_cfo_f, est_to_s = coarse_work_check(data1, est_cfo_f, est_to_s)  # tryi >= 1) # !!! tODO
    # logger.warning(f"Ow {pkt_idx} f={est_cfo_f} t={est_to_s}")

    codes = objective_cut(est_cfo_f, est_to_s, data1, pkt_idx, outpath)
    return est_cfo_f.item(), est_to_s.item(), codes
