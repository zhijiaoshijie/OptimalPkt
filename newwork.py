import numpy as np

from utils import *
import plotly.express as px
import plotly.graph_objects as go
from find_intersections import find_intersections
from pltfig import *
import scipy.stats as stats


def coarse_est_f_t(data1, estf, window_idx):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi  # = xie lv bian hua lv / 2 = pin lv bian hua lv * 2pi / 2
    betat = betai * (1 + estf / Config.sig_freq)
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs

    nstart = window_idx * nsymblen
    tstart = nstart / Config.fs

    nsymbr = cp.arange(around(nstart), around(nstart + nsymblen))
    polyref = togpu((betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0))
    sig_dcp = data1[nsymbr] * cp.exp(-1j * cp.polyval(polyref, nsymbr / Config.fs))
    data0 = myfft(sig_dcp, n=Config.fft_n, plan=Config.plan)
    fmax = cp.argmax(cp.abs(data0)) - Config.fs / 2  # fftshift: -500000 to 500000
    estt = - fmax / Config.bw * tsymblen # time shift +: sig move right, freq -
    return estt

def start_pidx_pow_detect(data1, estf, estt, window_idx = 10):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi  # = xie lv bian hua lv / 2 = pin lv bian hua lv * 2pi / 2
    betat = betai * (1 + estf / Config.sig_freq)
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs

    nstart = window_idx * nsymblen
    tstart = nstart / Config.fs

    nsymbr = cp.arange(around(nstart), around(nstart + nsymblen))
    polyref = togpu((betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0))
    pow1 = cp.abs(data1[nsymbr].dot(cp.exp(-1j * cp.polyval(polyref, nsymbr / Config.fs))))

    for pidx in range(window_idx):
        nstart = pidx * nsymblen
        tstart = nstart / Config.fs

        nsymbr = cp.arange(around(nstart), around(nstart + nsymblen))
        polyref = togpu((betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0))
        pow2 = cp.abs(data1[nsymbr].dot(cp.exp(-1j * cp.polyval(polyref, nsymbr / Config.fs))))
        if pow2 > pow1 * 0.5:
            return pidx

def show_fit_results(pktdata_in, estf, estt, coeflist_in, pkt_idx, margin):
    coeflist = coeflist_in.copy()
    betai = Config.bw / ((2 ** Config.sf) / Config.bw)
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs
    for pidx in range(0, Config.preamble_len):
        nstart = pidx * nsymblen + estt * Config.fs
        nsymbr = cp.arange(around(nstart) + margin, around(nstart + nsymblen) - margin)
        tsymbr = nsymbr / Config.fs
        coeflist[pidx, 2] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coeflist[pidx], tsymbr))))
    c0mean = cp.mean(coeflist[8:, 0])
    estf_from_c0 = Config.sig_freq * (c0mean / (cp.pi * betai) - 1)
    fig = pltfig1(None, coeflist[:, 0], title=f"{pkt_idx=} coef0 {estf_from_c0=}",
            addhline=[cp.pi * betai * (1 + x * estf / Config.sig_freq) for x in range(3)])
    fig.add_hline(y=c0mean, line=dict(dash='dot'))
    fig.show()
    logger.warning(f"show_fit_results: coef0: {pkt_idx=} input {estf=} {estt=} result {c0mean=} {estf_from_c0=}")


    # coef1

    freqnew = (coeflist[:, 0] * 2 * (estt + cp.arange(coeflist.shape[0]) * tsymblen) + coeflist[:, 1]) / 2 / cp.pi
    freqnew_mean = cp.mean(freqnew)
    pltfig1(None, freqnew, title=f"{pkt_idx=} coef1+coef0, iniFreq", addhline=[- Config.bw / 2 + estf, freqnew_mean], line_dash=['dash', 'dot']).show()
    logger.warning(f"show_fit_results: coef0+1: {pkt_idx=} {- Config.bw / 2 + estf =} {freqnew_mean=}")

    # coef1 linear trend
    pltfig1(None, coeflist[:, 1], title=f"{pkt_idx=} coef1").show()
    pltfig1(None, coeflist[:, 2], title=f"{pkt_idx=} coef2").show()



def plot_fit2d(coefficients_2d_in, estf, estt, pidx, pktdata_in, margin):
    logger.warning(f'''
    plt_fit2d called with {coefficients_2d_in=} {estf=} {pidx=}
    draw 3 imgs:
        1. the original 2d unwrapped, with fitted 2d poly curve
        2. the difference between them, unwrapped
        3. the original 2d nounwrapped, with fitted 2d poly curve wrapped in 
    ''')
    coefficients_2d = coefficients_2d_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs
    nstart = pidx * nsymblen + estt * Config.fs
    nstart2 = (pidx + 1) * nsymblen + estt * Config.fs
    tstart = nstart / Config.fs
    tend = tstart + tsymblen
    nsymbr = cp.arange(around(nstart) + margin, around(nstart2) - margin)
    tsymbr = nsymbr / Config.fs
    p0idx = margin
    nsymbrp = cp.arange(around(nstart) - p0idx, around(nstart) + Config.nsamp + margin)
    assert nsymbrp[p0idx] == around(nstart)
    tsymbrp = nsymbrp / Config.fs

    uarp = cp.unwrap(cp.angle(pktdata_in[nsymbrp]))
    coefficients_2d[-1] -= (cp.polyval(coefficients_2d, tsymbrp[p0idx]) - uarp[p0idx])
    coefficients_2d[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    logger.warning(f"inside plt_fit2d: {uarp[p0idx] - cp.polyval(coefficients_2d, tsymbrp[p0idx])}")
    pltfig(((tsymbrp, uarp),
            (tsymbrp, cp.polyval(coefficients_2d, tsymbrp))),
           title=f"{pidx=} fit 2d uwa curve",
           addvline=(tstart, tend)).show()
    pltfig(((tsymbrp, uarp - cp.polyval(coefficients_2d, tsymbrp)),),
           title=f"{pidx=} fit diff between 2d curve and uwa",
           modes='lines+markers',
           marker=dict(size=1),
           yaxisrange=[-2, 2],
           addvline=(tstart, tend)).show()
    tinytsymbrp = cp.linspace(tstart, tend, int(1e5))
    pltfig(((tsymbrp, cp.angle(pktdata_in[nsymbrp])),
            (tinytsymbrp, (wrap(cp.polyval(coefficients_2d, tinytsymbrp))))),
           modes=('markers', 'lines'),
           title=f"{pidx=} fit 2d no-uw angles",
           addvline=(tstart, tend)).show()


def plot_fit2d_after_refine(coefficients_2d_in, coef2d_refined_in, estf, estt, pidx, pktdata_in, margin):
    logger.warning(f'''
    plt_fit2d after_refine called with {coefficients_2d_in=} {coef2d_refined_in=} {estf=} {pidx=}
    draw 3 imgs:
        1. the original 2d unwrapped, with fitted 2d poly curve
        2. the difference between them, unwrapped
        3. the original 2d nounwrapped, with fitted 2d poly curve wrapped in 
    ''')
    coefficients_2d = coefficients_2d_in.copy()
    coef2d_refined = coef2d_refined_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs
    nstart = pidx * nsymblen + estt * Config.fs
    nstart2 = (pidx + 1) * nsymblen + estt * Config.fs
    tstart = nstart / Config.fs
    tend = tstart + tsymblen
    nsymbr = cp.arange(around(nstart) + margin, around(nstart2) - margin)
    tsymbr = nsymbr / Config.fs
    p0idx = margin
    nsymbrp = cp.arange(around(nstart) - p0idx, around(nstart) + Config.nsamp + margin)
    assert nsymbrp[p0idx] == around(nstart)
    tsymbrp = nsymbrp / Config.fs

    uarp = cp.unwrap(cp.angle(pktdata_in[nsymbrp]))
    coefficients_2d[-1] -= (cp.polyval(coefficients_2d, tsymbrp[p0idx]) - uarp[p0idx])
    coefficients_2d[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    coef2d_refined[-1] -= (cp.polyval(coef2d_refined, tsymbrp[p0idx]) - uarp[p0idx])
    coef2d_refined[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coef2d_refined, tsymbr))))
    pltfig(((tsymbrp, uarp),
                  (tsymbrp, cp.polyval(coefficients_2d, tsymbrp)),
                  (tsymbrp, cp.polyval(coef2d_refined, tsymbrp))),
           title = f"{pidx=} fit 2d uwa curve after_refine",
           addvline = (tstart, tend)).show()
    pltfig(((tsymbrp,uarp),
           (tsymbrp, uarp - cp.polyval(coefficients_2d, tsymbrp)),
           (tsymbrp, uarp - cp.polyval(coef2d_refined, tsymbrp))),
            title=f"{pidx=} fit diff between 2d curve and uwa after_refine",
            modes='lines+markers',
            marker=dict(size=1),
            yaxisrange=[-2, 2],
            addvline = (tstart, tend)).show()
    tinytsymbrp = cp.linspace(tstart, tend, int(1e5))
    pltfig(((tsymbrp, cp.angle(pktdata_in[nsymbrp])),
                    (tinytsymbrp, wrap(cp.polyval(coefficients_2d, tinytsymbrp))),
                    (tinytsymbrp, wrap(cp.polyval(coef2d_refined, tinytsymbrp)))),
                  modes=('markers', 'lines', 'lines'),
                  title=f"{pidx=} fit 2d no-uw angles after_refine",
                  addvline=(tstart, tend)).show()

def symbtime(estf, estt, pktdata_in, coeflist, draw=False, margin=1000):
    estcoef = [0.010082632769, 0.01015366531] #todo !!!
    nestt = estt * Config.fs
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    diffs = cp.zeros(Config.preamble_len - 1)
    dd2 = []
    # dd31 = []
    for pidx in range(0, Config.preamble_len - 1):
        nstart = pidx * nsymblen + nestt
        tstart = nstart / Config.fs
        nstart2 = (pidx + 1) * nsymblen + nestt
        tstart2 = nstart2 / Config.fs
        nstart3 = (pidx + 2) * nsymblen + nestt
        tstart3 = nstart3 / Config.fs
        nsymbr = cp.arange(around(nstart) + margin, around(nstart + nsymblen) - margin)
        tsymbr = nsymbr / Config.fs
        nsymbr2 = cp.arange(around(nstart2) + margin, around(nstart3) - margin)
        tsymbr2 = nsymbr2 / Config.fs
        coefa = coeflist[pidx].copy()
        coefb = coeflist[pidx + 1].copy()

        d2vala = np.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coefa, tsymbr))))
        d2valb = np.angle(pktdata_in[nsymbr2].dot(np.exp(-1j * np.polyval(coefb, tsymbr2))))
        assert d2vala < 1e-4 and d2valb < 1e-4, f"ERR anglediff>1e-4, {pidx=} angle={d2vala} {pidx+1=} angle={d2valb}"
        coefa[2] += d2vala
        coefb[2] += d2valb

        nsymba = cp.arange(around(nstart) - margin, around(nstart + nsymblen * 2) + margin)
        tsymba = nsymba / Config.fs
        ysymba = cp.zeros_like(pktdata_in, dtype=cp.float64)
        ysymba[nsymba] = cp.unwrap(cp.angle(pktdata_in[nsymba]))
        # pltfig1(tsymba, ysymba[nsymba], addvline=(np.polyval(estcoef, pidx), np.polyval(estcoef, pidx + 1)), title=f"{pidx=} duelsymb").show()

    pidx_range = cp.arange(Config.preamble_len)
    diffs2 = cp.zeros(Config.preamble_len - 1)
    for pidx in range(0, Config.preamble_len - 1):
        tstart2 = np.polyval(estcoef, pidx + 1)
        tdiffs, coefa, coefb = find_intersections(coeflist[pidx], coeflist[pidx + 1], tstart2, pktdata_in, 1e-5, margin=margin, draw=False)#draw= (pidx == 120))
        tdiff = min(tdiffs, key=lambda x: abs(x - tstart2))
        diffs2[pidx] = tdiff
    coeff_time = cp.polyfit(pidx_range[1 + 8:], diffs2[8:], 1)
    logger.warning(f"estimated time 2:coeff_time={coeff_time[0]:.12f},{coeff_time[1]:.12f} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    # pltfig(((pidx_range[1:], diffs2), (pidx_range[1:], np.polyval(coeff_time, pidx_range[1:]))),
    #        title="intersect points fitline").show()
    # pltfig1(pidx_range[1:], diffs2 - np.polyval(coeff_time, pidx_range[1:]), title="intersect points diff").show()


    dd = []
    for pidx in range(240):
        dd.append(- coeflist[pidx, 1] / 2 / coeflist[pidx, 0] - np.polyval(coeff_time, pidx))
    dd = tocpu(cp.array(dd))
    avgdd = np.mean(dd)
    logger.warning(f"coef2 time for freq=0 {avgdd=} estf={(0.5 - avgdd / coeff_time[0]) * Config.bw}")
    # pltfig1(None, dd, addhline=((-estf / Config.bw + 0.5) * coeff_time[0], avgdd), title="coef2 time for freq=0").show()

    avgdds = []
    for ixx in range(2):
        dd = []
        if ixx == 0: bwdiff = -Config.bw * (1 + estf / Config.sig_freq) / 2
        else: bwdiff = Config.bw * (1 + estf / Config.sig_freq) / 2
        for pidx in range(240):
            dd.append((coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx + ixx) + coeflist[pidx, 1]) / 2 / np.pi)
        dd = sqlist(dd) - bwdiff
        pidx_range2 = np.arange(50, Config.preamble_len)
        estfcoef_to_time = np.polyfit(np.polyval(coeff_time, pidx_range2), dd[pidx_range2], 1)
        avgdd1 = np.mean(dd)
        logger.warning(f"{avgdd1=:.12f}")
        avgdds.append(avgdd1)
        logger.warning(f"coef2 {'start' if ixx == 0 else 'end'} cfo={avgdd1} estf at t=0: {estfcoef_to_time[1]:.12f} estf change rate per sec: {estfcoef_to_time[0]:.12f}")
        estfcoef_to_num = np.polyfit(pidx_range2, tocpu(dd[pidx_range2]), 1)
        logger.warning(f"coef2 {'start' if ixx == 0 else 'end'} cfo={avgdd1} estf at t=0: {estfcoef_to_num[1]:.12f} estf change rate per symb: {estfcoef_to_num[0]:.12f}")

    logger.warning(f"{avgdds[1]=:.12f} {avgdds[0]=:.12f} {avgdds[1] - avgdds[0]=:.12f} {(avgdds[1] - avgdds[0]) / Config.bw=:.12f}")




    dd = []
    for pidx in range(240):
        a1 = (coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx) + coeflist[pidx, 1]) / 2 / np.pi
        a2 = (coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx + 1) + coeflist[pidx, 1]) / 2 / np.pi
        dd.append(a2 - a1)
    # pltfig1(None, dd, title=f"diff of each single symb").show()
    for ixx in range(1, 3):
        estbw = (1 + ixx * estf / Config.sig_freq) * Config.bw
        logger.warning(f"{np.mean(sqlist(dd))=:.12f} (1+{ixx}*ppm)=>{estbw=}")\



    dd1 = np.zeros(Config.preamble_len)
    dd2 = np.zeros(Config.preamble_len)
    for pidx in range(Config.preamble_len):
        dd1[pidx] = (wrap(np.polyval(coeflist[pidx - 1], np.polyval(coeff_time, pidx))))
        dd2[pidx] = (wrap(np.polyval(coeflist[pidx], np.polyval(coeff_time, pidx))))
    pidx_range3 = np.arange(Config.preamble_len - 1)
    pidx_range4 = np.arange(15, 90)
    estfcoef_phase_curve = np.polyfit(pidx_range4, tocpu(dd2[pidx_range4]), 2)
    estf1 = 2 * estfcoef_to_num[0] * 2 * np.pi
    logger.warning(f"{estfcoef_phase_curve=} {estf1=}")
    pidx_range4 = np.arange(110, 230)
    estfcoef_phase_curve2 = np.polyfit(pidx_range4, tocpu(dd2[pidx_range4]), 2)
    estf1 = estfcoef_to_num[0] * 2 * np.pi * coeff_time[0]
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * np.pi
    betat = betai * (1 + 2 * estfcoef_to_num[1] / Config.sig_freq)
    pidx_range = tocpu(pidx_range)
    logger.warning(f"{estfcoef_to_num[0]=:.12f} {estfcoef_to_num[1]=:.12f}")

    coeffitlist = np.zeros((Config.preamble_len, 3), dtype=np.float64)
    coeffitlist[:, 0] = betai * (1 + 2 * np.polyval(estfcoef_to_num, pidx_range) / Config.sig_freq)

    # dd.append((coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx + ixx) + coeflist[pidx, 1]) / 2 / np.pi)
    # estfcoef_to_num = np.polyfit(pidx_range2, tocpu(dd[pidx_range2]), 1)
    bwdiff = - Config.bw * (1 + estfcoef_to_num[1] / Config.sig_freq) / 2
    coeffitlist[:, 1] = 2 * np.pi * np.polyval(estfcoef_to_num, pidx_range) - np.polyval(tocpu(coeff_time), pidx_range) * 2 * coeffitlist[:, 0] + bwdiff * 2 * np.pi

    for pidx in pidx_range[1:]:
        coeffitlist[pidx, 2] -= np.polyval(coeffitlist[pidx], np.polyval(tocpu(coeff_time), pidx)) - np.polyval(coeffitlist[pidx - 1], np.polyval(tocpu(coeff_time), pidx))
    # pltfig1(None, dd,  title="each curve end minus start").show()


    estf2 = estfcoef_to_num[1] * 2 * np.pi * coeff_time[0] + betat * coeff_time[0] * coeff_time[0]
    logger.warning(f"{estfcoef_phase_curve2=} 2*differential={estfcoef_phase_curve2[0]*2:.12f} {estf1=:.12f} equal startdiff={estfcoef_phase_curve2[1]} {estf2=:.12f} ")
    # estfcoef_phase_curve2[0] = estfcoef_to_num[0] * np.pi * coeff_time[0], ax2+bx+c a=cfo_change_rate_per_symb * tsymb * pi. ax+b = (ax2+bx+c)-(ax2+bx+c) 2a = 2pi(b2-b1)(t2-t1)
    dx = []
    for pidx in range(Config.preamble_len):
        x1 = math.ceil(np.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(np.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs
        data = cp.exp(-1j * cp.polyval(togpu(coeffitlist[pidx]), tsymbr))
        pow = pktdata_in[nsymbr].dot(data) / cp.sum(cp.abs(pktdata_in[nsymbr]))
        dx.append(pow)
        if False:# pidx == 0 or pidx == 10 or pidx == 120 or pidx == 220:
            pltfig(((tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr]))), (tsymbr, cp.unwrap(-cp.angle(data)))),
                   title=f"{pidx=} fit curve {pow=}").show()
            pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * data), title=f"{pidx=} fit curve diff angle {pow=}").show()
            pltfig1(tsymbr, cp.abs(cp.cumsum(pktdata_in[nsymbr] * data)) / cp.cumsum(cp.abs(pktdata_in[nsymbr])), title=f"{pidx=} fit curve diff powercurve {pow=}").show()

    dx = sqlist(dx)
    # pltfig1(None, cp.angle(dx), title=f"fit phase diff").show()
    # pltfig1(None, cp.abs(dx), title=f"fit power").show()

    for pidx in range(Config.preamble_len + 2, Config.preamble_len + 4):
        x1 = math.ceil(np.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(np.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs
        coefficients_2d = cp.polyfit(tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr])), 2)
        logger.warning(f"downchirp {pidx=} {coefficients_2d=}")
        coef2d_refined = refine_coef(nsymbr, pidx, pktdata_in, coefficients_2d, margin=margin, searchquad=True)
        logger.warning(f"downchirp {pidx=} {coef2d_refined=}")

        estcoef_this = np.polyval(estfcoef_to_num, pidx)
        beta1 = - betai * (1 + 2 * estcoef_this / Config.sig_freq)
        bwdiff = - Config.bw * (1 + estcoef_this / Config.sig_freq) / 2
        beta2 = 2 * np.pi * (np.polyval(estfcoef_to_num, pidx) - bwdiff) - np.polyval(tocpu(coeff_time), pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        logger.warning(f"{beta1=} {beta2=} {(coef2d_refined[0]-beta1)/beta1=} {(coef2d_refined[1]-beta2)/beta2=}")
        coef2d_est = (beta1, beta2, 0)

        data = cp.exp(-1j * cp.polyval(togpu(coef2d_refined), tsymbr))
        pow = cp.abs(pktdata_in[nsymbr].dot(data)) / cp.sum(cp.abs(pktdata_in[nsymbr]))
        pow = pow.item()
        logger.warning(f"search {pow=}")
        # pltfig(((tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr]))), (tsymbr, cp.unwrap(-cp.angle(data)))),
        #        title=f"{pidx=} search fit curve {pow=}").show()
        # pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * data), title=f"{pidx=} search fit curve diff angle {pow=}").show()
        # pltfig1(tsymbr, cp.abs(cp.cumsum(pktdata_in[nsymbr] * data)) / cp.cumsum(cp.abs(pktdata_in[nsymbr])),
        #         title=f"{pidx=} search fit curve diff powercurve {pow=}").show()

        data = cp.exp(-1j * cp.polyval(togpu(coef2d_est), tsymbr))
        pow = cp.abs(pktdata_in[nsymbr].dot(data)) / cp.sum(cp.abs(pktdata_in[nsymbr]))
        pow = pow.item()
        logger.warning(f"fit {pow=}")
        # pltfig(((tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr]))), (tsymbr, cp.unwrap(-cp.angle(data)))),
        #        title=f"{pidx=}  est fit curve {pow=}").show()
        # pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * data), title=f"{pidx=} fit curve diff angle {pow=}").show()
        # pltfig1(tsymbr, cp.abs(cp.cumsum(pktdata_in[nsymbr] * data)) / cp.cumsum(cp.abs(pktdata_in[nsymbr])),
        #         title=f"{pidx=} est fit curve diff powercurve {pow=}").show()

        sig1 = pktdata_in[nsymbr]
        refchirp = data
        sig2 = sig1 * refchirp
        # freqdiff = start_pos_d / nsymblen * Config.bw / Config.fs * Config.fft_n
        # if ispreamble:
        #     freqdiff -= fstart / Config.sig_freq * Config.bw * pidx
        # else:
        #     freqdiff += fstart / Config.sig_freq * Config.bw * pidx
        # sig2 = add_freq(sig2, freqdiff)
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        freq = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
        logger.warning(f"fft {freq=} maxpow={cp.max(cp.abs(data0))}")
        pltfig1(None, cp.unwrap(cp.angle(sig2)), title=f"sig2 phase {pidx=}").show()
        # pltfig1(None,cp.abs(data0), title=f"fft {freq=} maxpow={cp.max(cp.abs(data0))}").show()



    if draw:
        for pidx in [0, 120, 239, 240]:
            estt = np.polyval(coeff_time, pidx)
            nestt = estt * Config.fs
            nsymbii = cp.arange(around(nestt) - 10, around(nestt) + 10)
            tsymbii = nsymbii / Config.fs
            pltfig(((tsymbii, cp.unwrap(cp.angle(pktdata_in[nsymbii]))),),
                   title=f"symbtime {pidx=} intersect smallview",
                   modes=('lines+markers',),
               addvline=(estt, )).show()
    estf_ret = np.mean(sqlist(avgdds))
    return coeff_time[0], estf_ret



def fitcoef(estf, estt, pktdata_in, margin, fitmethod = "2dfit", searchquad = True):
    nestt = estt * Config.fs
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi
    betat = betai * (1 + estf / Config.sig_freq)

    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    # coeff_time = [0.01008263, 0.01015366]
    coeflist = []
    for pidx in range(0, Config.preamble_len):
        nstart = pidx * nsymblen + nestt
        nsymbr = cp.arange(around(nstart) + margin, around(nstart + nsymblen) - margin)
        tsymbr = nsymbr / Config.fs
        # nstart = cp.polyval(coeff_time, pidx) * Config.fs

        #
        # 2d fit on unwrapped angle
        #

        if fitmethod == "2dfit":
            coefficients_2d = cp.polyfit(tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr])), 2)
        elif fitmethod == "1dfit":
            coefficients_2d = cp.polyfit(
                tsymbr,
                cp.unwrap(cp.angle(pktdata_in[nsymbr])) - cp.polyval(togpu((betat, 0, 0)), tsymbr),
                1
            )
            coefficients_2d = togpu([betat, *tocpu(coefficients_2d)])
        else: assert False, "fitmethod not in (2dfit, 1dfit)"
        logger.warning(f"{pidx=} {nstart=} {fitmethod=} {coefficients_2d=} {betat=} {betai=}")

        #
        # plot the results:
        #
        # plot_fit2d(coefficients_2d, estf, estt, pidx, pktdata_in)



        #
        # find accurate coefficients_2d by power
        #
        coef2d_refined = refine_coef(nsymbr, pidx, pktdata_in, coefficients_2d, margin=margin, searchquad = searchquad)
        coeflist.append(coef2d_refined)



        # plot_fit2d_after_refine(coefficients_2d, coef2d_refined, estf, estt, pidx, pktdata_in)
    return cp.array(coeflist)

def refine_coef(nsymbr, pidx, pktdata_in, coefficients_2d_in, margin, searchquad = False):
    coefficients_2d = coefficients_2d_in.copy()
    tsymbr = nsymbr / Config.fs

    val = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
    logger.info(
        f'find accurate coefficients_2d by power. before start, result from linear fit {coefficients_2d=} {val=}')

    rangeval = 0.005
    for i in range(10):

        # search for both 1st (quad) and 2nd (linear) value, or only 2nd value
        for j in range(0 if searchquad else 1, 2):
            xvals = cp.linspace(coefficients_2d[j] * (1 - rangeval), coefficients_2d[j] * (1 + rangeval), 1001)
            yvals = []
            for x in xvals:
                coef2 = coefficients_2d.copy()
                coef2[j] = x
                yvals.append(obj(tsymbr, pktdata_in[nsymbr], coef2))
            yvals = cp.array(yvals)
            oldv = coefficients_2d[j]
            coefficients_2d[j] = xvals[cp.argmax(yvals)]
            valnew = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
            if valnew < val*(1-1e-7):
                pltfig1(xvals, yvals, addvline=(oldv,), title=f"{pidx=} {i=} {j=} {val=} {valnew=}").show()
            assert valnew >= val * (1 - 1e-7), f"{val=} {valnew=} {i=} {j=} {coefficients_2d=} {val-valnew=}"
            if abs(valnew - val) < 1e-7: rangeval /= 4
            val = valnew

    # align with phase. here may have a 2pi difference when evaluated in unwrap
    coefficients_2d[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    # assert cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr)))) < 1e-4, 'err angle not resolved'
    return coefficients_2d