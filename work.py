
from objective import *
from utils import *


def coarse_work_fast(pktdata_in, fstart, tstart, sigD=False):
    assert tstart >= 0

    # if really have to linfit:
    # (1) try different initial cfo guesses, get the max one that falls in correct range
    # (2) wrap everybody into appropriate range *as the first symbol peak* and fit, get the slope,
    # then try intercept / intercept - bw / intercept + bw, get the highest

    # tstart = around(tstart) # !!!!! TODO tstart rounded !!!!!

    # plot angle of input
    # plt.plot(cp.unwrap(cp.angle(pktdata_in)).get()[around(tstart):around(tstart)+Config.nsamp*20])
    # plt.axvline(Config.nsamp)
    # plt.axvline(Config.nsamp*2)
    # plt.show()

    # tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    # cfoppm = fstart / Config.sig_freq
    # t1 = 2 ** Config.sf / Config.bw * (1 - cfoppm)
    # upchirp = mychirp(tstandard, f0=-Config.bw / 2, f1=Config.bw / 2, t1=t1)
    # downchirp = mychirp(tstandard, f0=Config.bw / 2, f1=-Config.bw / 2, t1=t1)

    fft_sig_n = Config.bw / Config.fs * Config.fft_n  # around(Config.bw / Config.fs * Config.fft_n) # 4096 fft_n=nsamp*fft_upsamp, nsamp=t*fs=2**sf/bw*fs, fft_sig_n=2**sf * fft_upsamp

    # upchirp dechirp
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    # assume chirp start at one in [0, Config.detect_range_pkts) possible windows
    # downchirp = cp.conj(gen_refchirp(0, -4e4, Config.nsamp))
    estf = fstart
    x = cp.arange(Config.nsamp) * (1 + estf / Config.sig_freq) / Config.fs
    bwnew = Config.bw * (1 + estf / Config.sig_freq)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    upchirp = cp.exp(2j * cp.pi * (betanew / 2 * x ** 2  + (- bwnew / 2) * x ))
    downchirp = cp.conj(upchirp)

    freqlowidx = Config.fft_n // 2 - (3 * bwnew / 2 + Config.cfo_range) / Config.fs * Config.fft_n
    freqhighidx = Config.fft_n // 2 + (bwnew / 2 + Config.cfo_range) / 2 / Config.fs * Config.fft_n

    for pidx in range(Config.skip_preambles, Config.preamble_len + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, downchirp, pidx, True)
        Config.fft_ups_x[pidx] = data0

    freqlowidx = Config.fft_n // 2 - (bwnew / 2 + Config.cfo_range) / Config.fs * Config.fft_n
    freqhighidx = Config.fft_n // 2 + (3 * bwnew / 2 + Config.cfo_range) / 2 / Config.fs * Config.fft_n

    for pidx in range(Config.sfdpos, Config.sfdpos + 2 + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, upchirp, pidx, False)
        Config.fft_downs_x[pidx - Config.sfdpos] = data0


    # fig = px.line(y=cp.array(x1)-cp.array(x2))
    # print(-cp.mean(cp.array(x1)-cp.array(x2))-Config.bw, bwnew2-Config.bw)
    # fig.add_hline(y=-bwnew2)
    # fig.show()


    # todo SFO是否会导致bw不是原来的bw
    # adding: highpeak of previous + lowpeak of following
    idx = around(bwnew / Config.fs * Config.fft_n)
    fft_ups_add = (cp.abs(Config.fft_ups_x[:-1, : - idx]) + cp.abs(Config.fft_ups_x[1:, idx:])) # 前一帧的低点和后一帧的高点是同一个symb。因为refchirp是-bw/2到bw/2 所以两个峰分别在高点+bw/2和-bw/2处
    fft_downs_add = (cp.abs(Config.fft_downs_x[:-1, idx:]) + cp.abs(Config.fft_downs_x[1:, : - idx])) # 前一帧的高点和后一帧的低点是同一个symb，argmax值不用上移。因为refchirp是-bw/2到bw/2 所以两个峰分别在高点+bw/2和-bw/2处
    # preamble：前一半峰值在上，后一半峰值在下。取每个symbol的靠后一半的和后一个的symbol的靠前一半，=取每个symbol的靠下的和后一个的靠上的
    # sfd: 前一半峰值在下，后一半峰值在上

    #[: -1]：前一个symbol [1:] 后一个symbol（拉前一个symb）
    #[：-idx]：低的freq [idx:]：高的freq，拉下来一个bw


    xx = []
    for i in range(Config.skip_preambles, Config.preamble_len + Config.detect_range_pkts - 1 ):
        xx.append(tocpu(cp.max(cp.abs(fft_ups_add[i, :]))))
    plt.plot(xx)
    plt.show()
    pltfig(((cp.arange(len(x1)), y1), (cp.arange(len(x1)), y2), (cp.arange(len(xx)),xx)), title="y").show()


    # fit the up chirps with linear, intersect with downchirp
    detect_vals = cp.zeros((Config.detect_range_pkts, 3))

    # try all possible starting windows, signal start at detect_pkt th window
    # y_values = []
    # for detect_pkt in range(Config.detect_range_pkts - 1):
    #     buff_freqs = around(Config.cfo_range * Config.fft_n / Config.fs)
    #     lower = around(- Config.bw - buff_freqs + Config.fft_n // 2)
    #     higher = around(buff_freqs + Config.fft_n // 2)
    #     y_value = tocpu(cp.max(cp.sum(
    #         cp.abs(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt, lower:higher]),
    #         axis=0)))
    #     y_values.append(y_value)
    # pltfig1(None, y_values).show()
    # sys.exit(0)

    for detect_pkt in range(Config.detect_range_pkts - 1):

        if False:
            fig = go.Figure(layout_title_text="plot fft ups add")
            for i in range(Config.skip_preambles + detect_pkt, Config.preamble_len + detect_pkt):
                fig.add_trace(go.Scatter(x=cp.arange(0, fft_ups_add.shape[1], 10), y=cp.abs(fft_ups_add[i, ::10].get()),
                                         mode="lines"))
            fig.update_layout(xaxis=dict(range=[y_value_debug - 500, y_value_debug + 500]))
            fig.show()
        # for direct add # x[d] + roll(x[d+1], -bw). peak at (-bw, 0), considering CFO, peak at (-3bw/2, bw/2). # argmax = yvalue.
        # if yvalue > -bw/2, consider possibility of yvalue - bw; else consider yvalue + bw.
        buff_freqs = around(Config.cfo_range * Config.fft_n / Config.fs)

        data1 = cp.sum(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt],axis=0)
        data2 = cp.sum(fft_downs_add[ detect_pkt:  2 + detect_pkt],axis=0)



        logger.warning(f"{detect_pkt=} {cp.max(data1)=} {cp.max(data2)=} {cp.mean(data1)=} {cp.mean(data2)=} {cp.argmax(data1)-Config.fft_n // 2=} {cp.argmax(data2)-Config.fft_n // 2=}")
        preamble_amax = cp.argmax(data1).item() - Config.fft_n // 2
        sfd_amax = cp.argmax(data2).item() - Config.fft_n // 2
        est_cfo_f = (preamble_amax + sfd_amax + Config.bw) / 2
        est_to_s = (- preamble_amax + sfd_amax + Config.bw) / 2 / beta * Config.fs
        logger.warning(f"{est_cfo_f=} {est_to_s=}")

        # y_value = beta * to + cfo + fft_n // 2 - 1.5 * bw

        # preamble：前一半峰值在上，后一半峰值在下。取每个symbol的靠后一半的和后一个的symbol的靠前一半，=取每个symbol的靠下的和后一个的靠上的
        # sfd: 前一半峰值在下，后一半峰值在上

        # 因为refchirp是 - bw / 2 到bw / 2, symb也是- bw / 2 到bw / 2,
        # 在窗口对齐的时候只有一个峰，均为0.
        # p[ppp p]ppp pppp s[sss s]sss (est_to_s < 0)（symbol起始位置处于实际window的第几个采样点上，rawdata[est_to_s:] = signal
        # 向右错一点点时，preamble上移少量beta dt，然后后一半出现一个小的峰在-bw + beta dt处，argmax在-bw + beta dt + cfo
        # sfd下移少量至-beta dt，然后后一半出现一个小的峰值在bw - beta dt处，argmax在- beta dt + cfo
        # (preamble + sfd + bw) / 2 = cfo
        # (- preamble + sfd + bw) / 2 / beta * fs = to （symbol起始位置处于实际window的第几个采样点上。）

        dvals = cp.max(data1).item() + cp.max(data2).item()
        detect_vals[detect_pkt] = cp.array(sqlist((dvals, est_cfo_f, est_to_s)))  # save result
    detect_pkt_max = cp.argmax(detect_vals[:, 0])
    est_cfo_f, est_to_s = detect_vals[detect_pkt_max, 1], detect_vals[detect_pkt_max, 2] + detect_pkt_max * Config.nsamp
    logger.warning(f"{est_cfo_f=} {est_to_s=} {detect_pkt_max=}")
    plt.plot(tocpu(cp.unwrap(cp.angle(pktdata_in[est_to_s:est_to_s + Config.nsamp * (Config.sfdpos + 2)]))))
    plt.show()

    return est_cfo_f, est_to_s
