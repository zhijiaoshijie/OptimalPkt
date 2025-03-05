
from objective import *
from utils import *
import sys

def coarse_work_fast(pktdata_in, fstart, tstart):
    assert tstart >= 0

    x = cp.arange(Config.nsamp) * (1 - fstart / Config.sig_freq) / Config.fs
    bwnew = Config.bw * (1 + fstart / Config.sig_freq)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew = beta * (1 + 2 * fstart / Config.sig_freq)
    upchirp = cp.exp(2j * cp.pi * (betanew / 2 * x ** 2  + (- bwnew / 2) * x ))
    downchirp = cp.conj(upchirp)

    for pidx in range(Config.skip_preambles, Config.preamble_len + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, downchirp, pidx, True)
        Config.fft_ups_x[pidx] = data0

    for pidx in range(Config.sfdpos, Config.sfdpos + 2 + Config.detect_range_pkts):
        data0 = dechirp_fft(tstart, fstart, pktdata_in, upchirp, pidx, False)
        Config.fft_downs_x[pidx - Config.sfdpos] = data0

    idx = around(bwnew / Config.fs * Config.fft_n)
    fft_ups_add = (cp.abs(Config.fft_ups_x[:-1, : - idx]) + cp.abs(Config.fft_ups_x[1:, idx:])) # 前一帧的低点和后一帧的高点是同一个symb。因为refchirp是-bw/2到bw/2 所以两个峰分别在高点+bw/2和-bw/2处
    fft_downs_add = (cp.abs(Config.fft_downs_x[:-1, idx:]) + cp.abs(Config.fft_downs_x[1:, : - idx])) # 前一帧的高点和后一帧的低点是同一个symb，argmax值不用上移。因为refchirp是-bw/2到bw/2 所以两个峰分别在高点+bw/2和-bw/2处
    # preamble：前一半峰值在上，后一半峰值在下。取每个symbol的靠后一半的和后一个的symbol的靠前一半，=取每个symbol的靠下的和后一个的靠上的
    # sfd: 前一半峰值在下，后一半峰值在上

    #[: -1]：前一个symbol [1:] 后一个symbol（拉前一个symb）
    #[：-idx]：低的freq [idx:]：高的freq，拉下来一个bw


    detect_vals = cp.zeros((Config.detect_range_pkts, 3))

    for detect_pkt in range(Config.detect_range_pkts - 1):
        data1 = cp.sum(fft_ups_add[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt],axis=0)
        data2 = cp.sum(fft_downs_add[ detect_pkt:  2 + detect_pkt],axis=0)

        preamble_amax = cp.argmax(data1).item() - Config.fft_n // 2
        sfd_amax = cp.argmax(data2).item() - Config.fft_n // 2
        est_cfo_f = (preamble_amax + sfd_amax + bwnew) / 2  + fstart
        est_to_s = (- preamble_amax + sfd_amax + bwnew) / 2 / betanew * Config.fs + tstart # 注意fft是从tstart开始做的

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
    est_cfo_f, est_to_s = detect_vals[detect_pkt_max, 1], detect_vals[detect_pkt_max, 2] + detect_pkt_max * Config.nsamp * (1 - fstart / Config.sig_freq)

    return est_cfo_f, est_to_s

def coarse_work_check(pktdata_in, fstart, tstart):
    assert tstart >= 0

    x = cp.arange(Config.nsamp) * (1 - fstart / Config.sig_freq) / Config.fs
    bwnew = Config.bw * (1 + fstart / Config.sig_freq)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    betanew = beta * (1 + 2 * fstart / Config.sig_freq)
    upchirp = cp.exp(2j * cp.pi * (betanew / 2 * x ** 2  + (- bwnew / 2) * x ))
    downchirp = cp.conj(upchirp)

    preamble_amax = optimize_chirp(fstart, tstart, pktdata_in, downchirp, range(Config.skip_preambles, Config.preamble_len), True)
    sfd_amax = optimize_chirp(fstart, tstart, pktdata_in, upchirp, range(Config.sfdpos, Config.sfdpos + 2), False)

    est_cfo_f = (preamble_amax + sfd_amax) / 2  + fstart
    est_to_s = (- preamble_amax + sfd_amax) / 2 / betanew * Config.fs + tstart # 注意fft是从tstart开始做的

    return est_cfo_f, est_to_s


def optimize_chirp(fstart, tstart, pktdata_in, refchirp, pidx_range, ispreamble):
    sig1as = []
    for pidx in pidx_range:
        nsamp_small = 2 ** Config.sf / Config.bw * Config.fs * (1 - fstart / Config.sig_freq)
        start_pos_all = nsamp_small * pidx + tstart
        start_pos = around(start_pos_all)
        start_pos_d = start_pos_all - start_pos
        freqdiff = start_pos_d / nsamp_small * Config.bw * (1 + fstart / Config.sig_freq) / Config.fs * Config.fft_n
        if not ispreamble: freqdiff *= -1
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos] * refchirp
        sig2 = add_freq(sig1, freqdiff - fstart)
        sig1as.append(sig2)
        # logger.warning(f"{pidx=} {start_pos=} {freqdiff=} {cp.abs(cp.sum(add_freq(sig2, 1)))=}")
    sig1as = cp.vstack(sig1as)
    tsymbr = cp.arange(Config.nsamp) / Config.fs * (1 - fstart / Config.sig_freq)

    def obj1(freq, xdata, ydata):
        return -cp.sum(cp.abs(ydata.dot(cp.exp(xdata * -1j * 2 * cp.pi * cp.array(freq))))).item()

    margin = Config.bw / 4
    result = minimize(obj1, 0, args=(tsymbr, sig1as), bounds=[(- margin, + margin)])  # !!!
    freq = result.x[0]
    # logger.warning(f"{freq=} {-obj1(0, tsymbr, sig1as)=}  {-obj1(1, tsymbr, sig1as)=} {-obj1(freq, tsymbr, sig1as)=}")

    # <<< SEE IF SEARCH ACCURATE USE GRID SEARCH >>>
    # freq = 0
    # for i in range(20):
    #     xvals = cp.linspace(freq - margin, freq + margin, 1001)
    #     yvals = [obj1(f, tsymbr, sig1as) for f in xvals]
    #     yvals = cp.array(sqlist(yvals))
    #     freq = xvals[cp.argmin(yvals)]
    #     valnew = cp.min(yvals)
    #     margin /= 2
    #     logger.warning(f"{freq=:.12f} {valnew=} {xvals[1] - xvals[0]=}")
    # logger.warning(f"{freq=} {-obj1(0, tsymbr, sig1as)=}  {-obj1(1, tsymbr, sig1as)=} {-obj1(freq, tsymbr, sig1as)=}")

    return freq
