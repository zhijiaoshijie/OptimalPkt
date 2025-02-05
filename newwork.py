import numpy as np

from utils import *
import plotly.express as px
import plotly.graph_objects as go
from find_intersections import find_intersections
from pltfig import *


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
    logger.warning(f"estimated time 2:{coeff_time[0]:.12f},{coeff_time[1]:.12f} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    # pltfig(((pidx_range[1:], diffs2), (pidx_range[1:], np.polyval(coeff_time, pidx_range[1:]))),
    #        title="intersect points fitline").show()
    pltfig1(pidx_range[1:], diffs2 - np.polyval(coeff_time, pidx_range[1:]), title="intersect points diff").show()


    dd = []
    for pidx in range(240):
        dd.append(- coeflist[pidx, 1] / 2 / coeflist[pidx, 0] - np.polyval(coeff_time, pidx))
    dd = tocpu(cp.array(dd))
    avgdd = np.mean(dd)
    logger.warning(f"coef2 time for freq=0 {avgdd=} estf={(0.5 - avgdd / coeff_time[0]) * Config.bw}")
    pltfig1(None, dd, addhline=((-estf / Config.bw + 0.5) * coeff_time[0], avgdd), title="coef2 time for freq=0").show()

    fig = None
    avgdds = []
    for ixx in range(2):
        dd = []
        if ixx == 0: bwdiff = -Config.bw / 2
        else: bwdiff = Config.bw / 2
        for pidx in range(240):
            dd.append((coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx + ixx) + coeflist[pidx, 1]) / 2 / np.pi)
        dd = sqlist(dd) - bwdiff
        pidx_range2 = np.arange(50, Config.preamble_len)
        estfcoef_to_time = np.polyfit(np.polyval(coeff_time, pidx_range2), dd[pidx_range2], 1)
        avgdd1 = np.mean(dd)
        avgdds.append(avgdd)
        logger.warning(f"coef2 {'start' if ixx == 0 else 'end'} cfo={avgdd1} estf at t=0: {estfcoef_to_time[1]:.12f} estf change rate per sec: {estfcoef_to_time[0]:.12f}")
        fig = pltfig1(None, dd, addhline=(estf, avgdd1), title=f"coef2 freq line:freq=-bw/2+estf", fig = fig)
        fig = pltfig1(pidx_range, np.polyval(estfcoef_to_time, np.polyval(coeff_time, pidx_range)), fig = fig)
    fig.show()

    logger.warning(f"{avgdds[1] - avgdds[0]=} {(avgdds[1] - avgdds[0]) / Config.bw=}")

    dd = []
    for pidx in range(240):
        a1 = (coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx) + coeflist[pidx, 1]) / 2 / np.pi
        a2 = (coeflist[pidx, 0] * 2 * np.polyval(coeff_time, pidx + 1) + coeflist[pidx, 1]) / 2 / np.pi
        dd.append(a2 - a1)
    pltfig1(None, dd, title=f"diff of each single symb").show()
    logger.warning(f"{np.mean(sqlist(dd))=:.12f}")

    dd = []
    dd2 = []
    for pidx in range(1, Config.preamble_len):
        dd.append(wrap(np.polyval(coeflist[pidx - 1], np.polyval(coeff_time, pidx))))
        dd2.append(wrap(np.polyval(coeflist[pidx], np.polyval(coeff_time, pidx))))
    dd = sqlist(dd)
    dd2= sqlist(dd2)
    pltfig(((np.arange(1, Config.preamble_len), dd), (np.arange(1, Config.preamble_len), dd2)), title=f"phases 1-240 lastpoly and newpoly").show()
    dx = []
    dy = []
    for pidx in range(Config.preamble_len):
        xx = around(np.polyval(coeff_time, pidx) * Config.fs)
        dx.append(xx / Config.fs)
        dy.append(cp.angle(pktdata_in[xx]))
    pltfig1(dx, dy, title=f"real dot phase").show()



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
    estf_ret = np.mean(avgdds)
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
        coef2d_refined = refine_coef(estf, estt, pidx, pktdata_in, coefficients_2d, margin=margin, searchquad = searchquad)
        coeflist.append(coef2d_refined)

        # plot_fit2d_after_refine(coefficients_2d, coef2d_refined, estf, estt, pidx, pktdata_in)
    return cp.array(coeflist)

def refine_coef(estf, estt, pidx, pktdata_in, coefficients_2d_in, margin, searchquad = False):
    coefficients_2d = coefficients_2d_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    nstart = pidx * nsymblen + estt * Config.fs
    nsymbr = cp.arange(around(nstart) + margin, around(nstart + nsymblen) - margin)
    tsymbr = nsymbr / Config.fs

    val = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
    logger.info(
        f'find accurate coefficients_2d by power. before start, result from linear fit {coefficients_2d=} {val=}')

    rangeval = 0.005
    for i in range(3):

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
            if abs(valnew - val) < 1e-7: rangeval /= 64
            val = valnew

    # align with phase. here may have a 2pi difference when evaluated in unwrap
    coefficients_2d[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    # assert cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr)))) < 1e-4, 'err angle not resolved'
    return coefficients_2d