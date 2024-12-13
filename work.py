from objective import *


def work(pktdata_in, fstart, tstart):
    assert tstart >= 0

    est_cfo_f, est_to_s, retval = coarse_updown_detect(pktdata_in, fstart, tstart)

    logger.info(f"updown result:{est_cfo_f=} {est_to_s=} {retval=}")

    # linear_dfreq, linear_dtime = objective_linear(est_cfo_f, est_to_s, pktdata_in)
    # logger.info(f"linear optimization {linear_dfreq=} {linear_dtime=}")
    # est_cfo_f -= linear_dfreq
    # est_to_s -= linear_dtime

    if est_to_s < 0: return 0, 0, None  # !!!

    return est_cfo_f, est_to_s, retval# None# (codes, codeangles)


def coarse_updown_detect(pktdata_in, fstart, tstart):
    tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    t1 = 2 ** Config.sf / Config.bw
    upchirp = mychirp(tstandard, f0=-Config.bw / 2, f1=Config.bw / 2, t1=t1)
    downchirp = mychirp(tstandard, f0=Config.bw / 2, f1=-Config.bw / 2, t1=t1)
    fft_sig_n = Config.bw / Config.fs * Config.fft_n  # round(Config.bw / Config.fs * Config.fft_n) # 4096 fft_n=nsamp*fft_upsamp, nsamp=t*fs=2**sf/bw*fs, fft_sig_n=2**sf * fft_upsamp
    # upchirp dechirp
    for pidx in range(
            Config.preamble_len + Config.detect_range_pkts):  # assume chirp start at one in [0, Config.detect_range_pkts) possible windows
        data0 = dechirp_fft(tstart, fstart, pktdata_in, downchirp, pidx, True)
        Config.fft_ups_x[pidx] = data0
    for pidx in range(Config.sfdpos, Config.sfdpos + 2 + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, upchirp, pidx, False)
        Config.fft_downs_x[pidx - Config.sfdpos] = data0
    fft_ups_add = (Config.fft_ups_x[:-1, :int(-Config.bw / Config.fs * Config.fft_n)] +
                   Config.fft_ups_x[1:, int(Config.bw / Config.fs * Config.fft_n):])
    # fit the up chirps with linear, intersect with downchirp
    results = []
    for detect_pkt in range(
            Config.detect_range_pkts - 1):  # try all possible starting windows, signal start at detect_pkt th window

        # for direct add # x[d] + roll(x[d+1], -bw). peak at (-bw, 0), considering CFO, peak at (-3bw/2, bw/2). # argmax = yvalue.
        # if yvalue > -bw/2, consider possibility of yvalue - bw; else consider yvalue + bw.
        buff_freqs = round(Config.cfo_range * Config.fft_n / Config.fs)
        lower = - Config.bw - buff_freqs + Config.fft_n // 2
        higher = buff_freqs + Config.fft_n // 2
        y_value = tocpu(cp.argmax(cp.sum(
            cp.abs(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt, lower:higher]),
            axis=0))) + lower
        if y_value > - Config.bw // 2 * Config.fft_n / Config.fs + Config.fft_n // 2:
            y_value_secondary = -1
        else:
            y_value_secondary = 1

        k = fstart / Config.sig_freq * Config.bw
        coefficients = (k, y_value - (Config.skip_preambles + detect_pkt) * k)
        polynomial = np.poly1d(coefficients)

        # up-down algorithm
        # the fitted line intersect with the fft_val_down, compute the fft_val_up in the same window with fft_val_down (at fdown_pos)
        # find the best downchirp among all possible downchirp windows
        fdown_pos, fdown = cp.unravel_index(cp.argmax(cp.abs(Config.fft_downs_x[detect_pkt: detect_pkt + 2])),
                                            Config.fft_downs_x[detect_pkt: detect_pkt + 2].shape)
        fdown_pos = fdown_pos.item() + detect_pkt + Config.sfdpos  # position of best downchirp
        fdown = fdown.item()  # freq of best downchirp
        if fdown > Config.fft_n // 2:
            fdown2 = -1
        else:
            fdown2 = 1

        fft_val_up = (polynomial(fdown_pos) - (Config.fft_n // 2)) / fft_sig_n
        fft_val_down = (fdown - (Config.fft_n // 2)) / fft_sig_n

        # try all possible variations (unwrap f0, t0 if their real value exceed [-0.5, 0.5))
        deltaf, deltat = np.meshgrid(np.array((0, y_value_secondary)), np.array((0, fdown2)))
        for i in range(deltaf.shape[0]):
            for j in range(deltaf.shape[1]):
                fu = fft_val_up + deltaf[i, j]
                fd = fft_val_down + deltat[i, j]
                f0 = (fu + fd) / 2
                t0 = (f0 - fu)
                f1 = f0 * Config.bw
                t1 = t0 * Config.tsig + tstart + detect_pkt * Config.nsampf

                # linear_dfreq, linear_dtime = objective_core_new(f1, t1, pktdata_in)
                # logger.info(f"linear optimization {f1=} {t1=} {linear_dfreq=} {linear_dtime=}")
                # f1 -= linear_dfreq
                # t1 -= linear_dtime
                retval, f2, t2 = objective_core_new(f1, t1, pktdata_in)

                results.append((f2, t2, retval))

    max_result = max(results, key=lambda x: x[2])  # x[2] refers to retval
    f2_max, t2_max, max_retval = max_result
    return f2_max, t2_max, max_retval
