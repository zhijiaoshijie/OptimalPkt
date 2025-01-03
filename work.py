from objective import *


def coarse_work_fast(pktdata_in, fstart, tstart, sigD=False):
    assert tstart >= 0

    # if really have to linfit:
    # (1) try different initial cfo guesses, get the max one that falls in correct range
    # (2) wrap everybody into appropriate range *as the first symbol peak* and fit, get the slope,
    # then try intercept / intercept - bw / intercept + bw, get the highest

    # tstart = round(tstart) # !!!!! TODO tstart rounded !!!!!

    # plot angle of input
    # plt.plot(cp.unwrap(cp.angle(pktdata_in)).get()[round(tstart):round(tstart)+Config.nsamp*20])
    # plt.axvline(Config.nsamp)
    # plt.axvline(Config.nsamp*2)
    # plt.show()

    tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    cfoppm = fstart / Config.sig_freq
    t1 = 2 ** Config.sf / Config.bw * (1 - cfoppm)
    upchirp = mychirp(tstandard, f0=-Config.bw / 2, f1=Config.bw / 2, t1=t1)
    downchirp = mychirp(tstandard, f0=Config.bw / 2, f1=-Config.bw / 2, t1=t1)

    fft_sig_n = Config.bw / Config.fs * Config.fft_n  # round(Config.bw / Config.fs * Config.fft_n) # 4096 fft_n=nsamp*fft_upsamp, nsamp=t*fs=2**sf/bw*fs, fft_sig_n=2**sf * fft_upsamp

    # upchirp dechirp
    x1 = []
    x2 = []
    # assume chirp start at one in [0, Config.detect_range_pkts) possible windows
    # downchirp = cp.conj(gen_refchirp(0, -4e4, Config.nsamp))
    estf = fstart
    x = cp.arange(Config.nsamp) * (1 + estf / Config.sig_freq)
    yi = cp.zeros_like(x, dtype=np.complex64)
    bwnew = Config.bw * (1 + estf / Config.sig_freq)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    upchirp = cp.exp(2j * cp.pi * (betanew / 2 * x ** 2 / Config.fs ** 2 + (- bwnew / 2) * x / Config.fs))
    downchirp = cp.conj(upchirp)
    if False: # not knowing how many preambles: search for sfd
        maxdowns = []
        for pidx in range(0, int((len(pktdata_in)-tstart) / Config.nsamp)-10):
            data0 = dechirp_fft(tstart, fstart, pktdata_in, upchirp, pidx, False)
            maxdowns.append(tocpu(cp.max(cp.abs(data0))))
        fig=px.line(y=maxdowns)
        fig.show()
    for pidx in range(Config.skip_preambles, Config.preamble_len + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, downchirp, pidx, True)
        Config.fft_ups_x[pidx] = data0
        x1.append(tocpu(cp.argmax(cp.abs(data0[:len(data0)//2]))))
        x2.append(tocpu(cp.argmax(cp.abs(data0[len(data0)//2:]))+len(data0)//2))
        # plt.plot(np.arange(x1[-1]-1000, x1[-1]+1000),tocpu(cp.abs(data0[x1[-1]-1000: x1[-1]+1000])))
        # plt.axvline(x=x1[-1], color='k')
        # plt.title(f"{pidx=} {tstart=}")
        # plt.show()
    bwnew2 = Config.bw * (1 - 2 * estf / Config.sig_freq)

    # fig = px.line(y=np.array(x1)-np.array(x2))
    # print(-np.mean(np.array(x1)-np.array(x2))-Config.bw, bwnew2-Config.bw)
    # fig.add_hline(y=-bwnew2)
    # fig.show()

    for pidx in range(Config.sfdpos, Config.sfdpos + 2 + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, upchirp, pidx, False)
        Config.fft_downs_x[pidx - Config.sfdpos] = data0

    # todo SFO是否会导致bw不是原来的bw
    fft_ups_add = (cp.abs(Config.fft_ups_x[:-1, :round(-bwnew2 / Config.fs * Config.fft_n)]) +
                   cp.abs(Config.fft_ups_x[1:, round(bwnew2 / Config.fs * Config.fft_n):])) # TODO!!!Abs
    # for i in range(fft_ups_add.shape[0]):
    #     k = estf / Config.sig_freq * Config.bw
    #     fft_ups_add[i] = cp.roll(fft_ups_add[i], -round(k * i))

    # fig = FigureResampler(go.Figure(layout_title_text=f"fft_ups values"))
    # fig = go.Figure(layout_title_text=f"fft_ups values")
    # for i in range(Config.skip_preambles, Config.preamble_len, 10):
    #     # fig.add_trace(go.Scatter(y=tocpu(cp.abs(Config.fft_ups_x[i, 348200:348900]))))
    #     fig.add_trace(go.Scatter(y=tocpu(cp.abs(fft_ups_add[i, 348200:348900]))))
    # fig.show()
    xx = []
    for i in range(Config.skip_preambles, Config.preamble_len):
        xx.append(tocpu(cp.argmax(cp.abs(fft_ups_add[i, :]))))
    # plt.plot(xx)
    # plt.show()
    # sys.exit(2)
    # fft_downs_add = Config.fft_downs_x[:-1, :-Config.bw / Config.fs * Config.fft_n] + Config.fft_downs_x[1:, Config.bw / Config.fs * Config.fft_n:]

    # fit the up chirps with linear, intersect with downchirp
    detect_vals = np.zeros((Config.detect_range_pkts, 3))

    # try all possible starting windows, signal start at detect_pkt th window
    for detect_pkt in range(Config.detect_range_pkts - 1):

        if False:
            fig = go.Figure(layout_title_text="plot fft ups add")
            for i in range(Config.skip_preambles + detect_pkt, Config.preamble_len + detect_pkt):
                fig.add_trace(go.Scatter(x=np.arange(0, fft_ups_add.shape[1], 10), y=np.abs(fft_ups_add[i, ::10].get()),
                                         mode="lines"))
            fig.update_layout(xaxis=dict(range=[y_value_debug - 500, y_value_debug + 500]))
            fig.show()
        # for direct add # x[d] + roll(x[d+1], -bw). peak at (-bw, 0), considering CFO, peak at (-3bw/2, bw/2). # argmax = yvalue.
        # if yvalue > -bw/2, consider possibility of yvalue - bw; else consider yvalue + bw.
        buff_freqs = round(Config.cfo_range * Config.fft_n / Config.fs)
        lower = - Config.bw - buff_freqs + Config.fft_n // 2
        higher = buff_freqs + Config.fft_n // 2
        y_value = tocpu(cp.argmax(cp.sum(
            cp.abs(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt, lower:higher]),
            axis=0))) + lower
        # y_value_debug = tocpu(cp.argmax(
        #     cp.abs(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt, lower:higher]),
        #     axis=0))) + lower
        if y_value > - Config.bw // 2 * Config.fft_n / Config.fs + Config.fft_n // 2:
            y_value_secondary = -1
        else:
            y_value_secondary = 1

        k = fstart / Config.sig_freq * Config.bw
        k = 0
        coefficients = (k, y_value - (Config.skip_preambles + detect_pkt) * k)
        polynomial = np.poly1d(coefficients)
        # logger.warning(f"Wpoly {coefficients} {-41774.000/Config.sig_freq * Config.bw}")

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

        fft_val_up = (polynomial(fdown_pos) - (
                    Config.fft_n // 2)) / fft_sig_n  # rate, [-0.5, 0.5) if no cfo and to it should be zero  #!!! because previous +0.5
        fft_val_down = (fdown - (Config.fft_n // 2)) / fft_sig_n

        # try all possible variations (unwrap f0, t0 if their real value exceed [-0.5, 0.5))
        deltaf, deltat = np.meshgrid(np.array((0, y_value_secondary)), np.array((0, fdown2)))
        values = np.zeros((2, 2, 3)).astype(float)
        nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
        for i in range(deltaf.shape[0]):
            for j in range(deltaf.shape[1]):
                fu = fft_val_up + deltaf[i, j]
                fd = fft_val_down + deltat[i, j]
                f0 = (fu + fd) / 2
                t0 = (f0 - fu)
                f1 = f0 * Config.bw
                t1 = t0 * Config.tsig + tstart + detect_pkt * nsamp_small

                retval= 0#objective_core_new(f1 , t1, pktdata_in)
                logger.warning(f"linear optimization {retval=:8.5f} {f1=:11.3f} {t1=:11.3f}") # TODO debug

                values[i][j] = [f1, t1, retval]

        best_idx = np.argmax(values[:, :, 2])
        est_cfo_f = values[:, :, 0].flat[best_idx]
        est_to_s = values[:, :, 1].flat[best_idx]
        dvals = np.max(values[:, :, 2])
        detect_vals[detect_pkt] = (dvals, est_cfo_f, est_to_s)  # save result

    # find max among all detect windows
    detect_pkt_max = np.argmax(detect_vals[:, 0])
    est_cfo_f, est_to_s = detect_vals[detect_pkt_max, 1], detect_vals[detect_pkt_max, 2]

    logger.info(f"updown result:{est_cfo_f=} {est_to_s=}")

    # linear_dfreq, linear_dtime = objective_linear(est_cfo_f, est_to_s, pktdata_in)
    # logger.warning(f"linear optimization {linear_dfreq=} {linear_dtime=}")
    # est_cfo_f -= linear_dfreq
    # est_to_s -= linear_dtime

    if est_to_s < 0: return 0, 0, None  # !!!

    if sigD:
        logger.warning(f"pre sigD parameters:{est_cfo_f=} {est_to_s=}")
        dphaselist = []
        for pidx in range(Config.preamble_len):  # assume chirp start at one in [0, Config.detect_range_pkts) possible windows
            start_pos_all = nsamp_small * pidx + est_to_s
            start_pos = round(start_pos_all)
            start_pos_d = start_pos_all - start_pos
            # use input cfo for sfo

            # pass

            # t1 = 2 ** Config.sf / Config.bw * (1 - cfoppm)
            start_pos_all_new = nsamp_small * pidx * (1 - est_cfo_f / Config.sig_freq) + est_to_s
            start_pos = round(start_pos_all_new)
            # t1 = nsamp_small * (pidx + 1) * (1 + est_cfo_f / Config.sig_freq * Config.bw) + est_to_s - start_pos_all_new
            tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1] + (
                        start_pos - start_pos_all_new) / Config.fs
            # print(tstandard)
            cfoppm1 = (1 + est_cfo_f / Config.sig_freq)  # TODO!!!
            downchirp = mychirp(tstandard, f0=Config.bw / 2 * cfoppm1 - est_cfo_f,
                                f1=-Config.bw / 2 * cfoppm1 - est_cfo_f, t1=2 ** Config.sf / Config.bw * cfoppm1)
            sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
            sig2 = sig1 * downchirp

            data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
            # print("new", np.max(np.abs(data0)), (start_pos - start_pos_all_new)/Config.fs, tstandard[0], tstandard[-1], Config.bw / 2 * (1 - est_cfo_f / Config.sig_freq )  - est_cfo_f, -Config.bw / 2* (1 - est_cfo_f / Config.sig_freq )  - est_cfo_f, 2 ** Config.sf / Config.bw  * (1 - est_cfo_f / Config.sig_freq ) )
            yval2 = cp.argmax(cp.abs(data0)).item()
            dval2 = np.array(tocpu(cp.angle(data0[yval2])).item())  # - dphase
            # dval2 = np.array(cp.angle(data0[Config.fft_n//2]).get().item())# - dphase
            # linear, the difference on angle = -0.03270806338636364 * bin so 1 bin(1hz) = 0.03 rad, angle[y]=angle[n/2]-0.03*(y-n/2)
            # print("newres", yval2 - Config.fft_n//2, dval2, cp.max(cp.abs(data0)).item(), np.abs(data0[Config.fft_n//2]))
            # plt.plot(cp.abs(data0).get())
            # plt.title("new fft result")
            # plt.show()

            dphaselist.append(dval2)
        uplist = np.unwrap(dphaselist)
        # debug !!!
        if uplist[-1] < 0:
            uplist[:2] += 2 * np.pi
            uplist[:1] += 2 * np.pi
        # uplist = np.array(dphaselist)
        x_val = np.arange(Config.skip_preambles, Config.preamble_len - 1)
        y_val = uplist[x_val]
        coefficients = np.polyfit(x_val, y_val, 1)
        fit_dfreq = coefficients[0] / (2 * np.pi) / Config.tsig * Config.fs
        if False:
            # fig = px.line(y=uplist, title=f"add1 {coefficients[0]=:.5f} {fit_dfreq=}")
            fig = go.Figure()
            x_val2 = np.arange(Config.preamble_len)
            y_val2 = np.polyval(coefficients, x_val2)
            # fig.add_trace(go.Scatter(x=x_val2, y=y_val2, mode="lines"))
            fig.add_trace(go.Scatter(x=x_val2, y=y_val2 - uplist, mode="lines"))
            fig.show()
        # print(f"sigd preobj {objective_core(est_cfo_f, est_to_s, pktdata_in)=}")
        # print(f"sigd preobj {objective_core_phased(est_cfo_f, est_to_s, pktdata_in)=}")
        if False:
            dxval = np.linspace(-100, 100, 1000)
            beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
            dyval = [objective_core_phased(est_cfo_f + x2, est_to_s - x2 / beta, pktdata_in) for x2 in dxval]
            fig = go.Figure(layout_title_text="plot neighbor of objective")
            fig.add_trace(go.Scatter(x=dxval, y=dyval, mode="lines"))
            # fig.add_vline(x=0, line=dict(color="black", dash="dash"))
            fig.show()
        retval = objective_core_new(est_cfo_f, est_to_s, pktdata_in)
        # retval2, est_cfo_f, est_to_s = objective_core_new(est_cfo_f, est_to_s, pktdata_in)
        logger.warning(f"final fit dphase {coefficients=} {fit_dfreq=} {retval=} {est_cfo_f=} {est_to_s=}")
    # codes, codeangles = objective_decode(est_cfo_f, est_to_s, pktdata_in)
    # print("work ending")
    # sys.exit(0)

        # print(f"sigd preobj {objective_core(est_cfo_f - fit_dfreq, est_to_s - fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core(est_cfo_f + fit_dfreq, est_to_s + fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core(est_cfo_f + fit_dfreq, est_to_s - fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core(est_cfo_f - fit_dfreq, est_to_s + fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core_phased(est_cfo_f - fit_dfreq, est_to_s - fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core_phased(est_cfo_f + fit_dfreq, est_to_s + fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core_phased(est_cfo_f + fit_dfreq, est_to_s - fit_dfreq / beta, pktdata_in)=}")
        # print(f"sigd preobj {objective_core_phased(est_cfo_f - fit_dfreq, est_to_s + fit_dfreq / beta, pktdata_in)=}")

    return est_cfo_f, est_to_s, retval# None# (codes, codeangles)
