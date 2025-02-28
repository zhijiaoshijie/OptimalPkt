from work import *

def mainwork(pkt_idx, data1):
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
        est_cfo_f, est_to_s, retval = coarse_work_fast(data1, est_cfo_f, est_to_s, False)  # tryi >= 1)
        logger.warning(f"Cw {pkt_idx} f={est_cfo_f} t={est_to_s}")

        if est_to_s < 0:
            logger.error(f"ERROR in Coarsework {est_cfo_f=} {est_to_s=} out {est_cfo_f=} {est_to_s=} {pkt_idx=}")
            est_to_s = 0
            break
    est_to_s, flag = find_power(est_cfo_f, est_to_s, data1)
    logger.warning(f"Fw {pkt_idx} f={est_cfo_f} t={est_to_s}")
    # if not flag: continue ## !!!TODO debug
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
        print(coefficients, est_cfo_f)
        c1.append(coefficients[0])
        c2.append(coefficients[1])
    # pltfig1(est_cfo_fs, c1).show()
    # pltfig1(est_cfo_fs, c2).show()

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
    objective_cut(est_cfo_f, est_to_s, data1, pkt_idx_cnt)
    return est_cfo_f, est_to_s