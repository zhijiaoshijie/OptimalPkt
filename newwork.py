import numpy as cp
from pandas.core.computation.expr import intersection

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


def optimize_1dfreq(sig2, tsymbr, freq):
    def obj1(xdata, ydata, freq):
        return cp.abs(ydata.dot(cp.exp(-1j * 2 * cp.pi * freq * xdata)))

    rangeval = 500
    val = obj1(tsymbr, sig2, freq)
    for i in range(10):
        xvals = cp.linspace(freq - rangeval, freq + rangeval, 1001)
        yvals = [obj1(tsymbr, sig2, f) for f in xvals]
        yvals = sqlist(yvals)
        freq = xvals[cp.argmax(yvals)]
        valnew = cp.max(yvals)
        if valnew < val * (1 - 1e-7):
            pltfig1(xvals, yvals, addvline=(freq,), title=f"{i=} {val=} {valnew=}").show()
        assert valnew >= val * (1 - 1e-7), f"{val=} {valnew=} {i=} {val-valnew=}"
        if abs(valnew - val) < 1e-7: rangeval /= 4
        val = valnew
    return freq, val / cp.sum(cp.abs(sig2))


def symbtime(estf, estt, pktdata_in, coeflist, draw=False, margin=1000):
    tsymblen = 2 ** Config.sf / Config.bw * (1 - estf / Config.sig_freq)

    # coarse estimation of range
    dx = []
    dy = []
    for pidx in cp.arange(10, Config.preamble_len):
        tstart2 = estt + tsymblen * pidx
        selected = find_intersections(coeflist[pidx - 1], coeflist[pidx], tstart2, pktdata_in, 1e-5, margin=margin, draw=False, remove_range=True)#draw= (pidx == 120))
        if selected != None:
            dx.append(pidx)
            dy.append(selected)
    dx = sqlist(dx)
    dy = sqlist(dy)
    coeff_time = cp.polyfit(dx, dy, 1)

    logger.warning(f"guessed: {tsymblen=} {estt=} estimated coeff_time={coeff_time[0]:.12f},{coeff_time[1]:.12f} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    # pltfig(((dx, dy), (dx, cp.polyval(coeff_time, dx))),
    #        title="intersect points fitline").show()
    # pltfig1(dx, dy - cp.polyval(coeff_time, dx), title="intersect points diff").show()


    pidx_range = cp.arange(Config.preamble_len)

    # TODO simplify
    for ixx in range(2):
        dd = []
        if ixx == 0: bwdiff = -Config.bw * (1 + estf / Config.sig_freq) / 2
        else: bwdiff = Config.bw * (1 + estf / Config.sig_freq) / 2
        for pidx in range(240):
            dd.append((coeflist[pidx, 0] * 2 * cp.polyval(coeff_time, pidx + ixx) + coeflist[pidx, 1]) / 2 / cp.pi)
        dd = sqlist(dd) - bwdiff
        pidx_range2 = cp.arange(50, Config.preamble_len)
        estfcoef_to_num = cp.polyfit(pidx_range2, dd[pidx_range2], 1)
        logger.warning(f"coef2 {'start' if ixx == 0 else 'end'} estfcoef_to_num at t=0: {estfcoef_to_num[1]:.12f} estf change rate per symb: {estfcoef_to_num[0]:.12f}")

    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi
    coeffitlist = cp.zeros((Config.preamble_len, 3), dtype=cp.float64)
    coeffitlist[:, 0] = betai * (1 + 2 * cp.polyval(estfcoef_to_num, pidx_range) / Config.sig_freq)

    bwdiff = - Config.bw * (1 + estfcoef_to_num[1] / Config.sig_freq) / 2
    coeffitlist[:, 1] = 2 * cp.pi * cp.polyval(estfcoef_to_num, pidx_range) - cp.polyval(coeff_time, pidx_range) * 2 * coeffitlist[:, 0] + bwdiff * 2 * cp.pi

    for pidx in pidx_range[1:]:
        coeffitlist[pidx, 2] -= cp.polyval(coeffitlist[pidx], cp.polyval(coeff_time, pidx)) - cp.polyval(coeffitlist[pidx - 1], cp.polyval(coeff_time, pidx))


    codephase = []
    powers = []


    # preamble codephase and powers
    for pidx in range(Config.preamble_len):
        x1 = math.ceil(cp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(cp.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs
        res = pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(togpu(coeffitlist[pidx]), tsymbr)))
        codephase.append(cp.angle(res).item())
        powers.append(cp.abs(res).item() / cp.sum(cp.abs(pktdata_in[nsymbr])).item())
    # pltfig1(None, cp.unwrap(codephase), title="unwrap phase").show()

    coeffitlist = cp.concatenate((coeffitlist, cp.zeros((100, 3))), axis=0)
    fig=None
    for pidx in range(Config.preamble_len, Config.preamble_len + 2):

        # FFT find frequency
        x1 = math.ceil(cp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(cp.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs

        estcoef_this = cp.polyval(estfcoef_to_num, pidx)
        beta1 = betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        beta2 = 2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) - estbw / 2) - cp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est = cp.array([beta1.get(), beta2.get(), 0])

        refchirp = cp.exp(-1j * cp.polyval(coef2d_est, tsymbr))
        sig2 = pktdata_in[nsymbr] * refchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        freq = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
        assert cp.max(cp.abs(data0)) > 0.9, f"FFT power <= 0.9, {pidx=} fft {freq=} maxpow={cp.max(cp.abs(data0))}"
        # freq, valnew = optimize_1dfreq(sig2, tsymbr, freq)
        code = freq / estbw * 2 ** Config.sf
        # logger.warning(f"{pidx=} optimized fft {freq=} maxpow={valnew} {code=:.12f}")
        code = around(code)

        x3 = math.ceil(cp.polyval(coeff_time, pidx + 1 - code / 2 ** Config.sf ) * Config.fs)
        nsymbr1 = cp.arange(x1, x3)
        tsymbr1 = nsymbr1 / Config.fs

        beta2 = (2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
                 - cp.polyval(coeff_time, pidx) * 2 * beta1)
        coef2d_est2 = cp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = cp.polyval(coef2d_est2, cp.polyval(coeff_time, pidx)) - cp.polyval(coeffitlist[pidx - 1], cp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        coeffitlist[pidx] = coef2d_est2

        res2 = pktdata_in[nsymbr1].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr1)) )

        codephase.append(cp.angle(res2).item())
        powers.append(cp.abs(res2).item() / cp.sum(cp.abs(pktdata_in[nsymbr1])).item())
        # pltfig1(tsymbr1, cp.angle(sig21 * cp.exp(-1j * 2 * cp.pi * freq1 * tsymbr1)), title=f"residue {pidx=}").show()
        logger.warning(f"{pidx=} {code=} {cp.angle(res2)=} pow={cp.abs(res2)/cp.sum(cp.abs(pktdata_in[nsymbr1]))}")
        fig=pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)

    for pidx in range(Config.preamble_len + 2, Config.preamble_len + 4):
        x1 = math.ceil(cp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(cp.polyval(coeff_time, pidx + 1) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs

        estcoef_this = cp.polyval(estfcoef_to_num, pidx)
        beta1 = - betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        # logger.error(f"EEE! {Config.bw * (estcoef_this / Config.sig_freq)=}")
        beta2 = 2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) + estbw / 2) - cp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est2 = cp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = cp.polyval(coef2d_est2, cp.polyval(coeff_time, pidx)) - cp.polyval(
            coeffitlist[pidx - 1], cp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        coeffitlist[pidx] = coef2d_est2
        res2 = pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr)))
        fig = pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)
        codephase.append(cp.angle(res2).item())
        powers.append(cp.abs(res2).item() / cp.sum(cp.abs(pktdata_in[nsymbr])).item())
        # pltfig1(None, cp.angle(pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}").show()
        # freq, power = optimize_1dfreq(pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr)), tsymbr, 0)
        # logger.error(f"EEE! {freq=} {power=}")

    for pidx in range(Config.preamble_len + 4, Config.preamble_len + 5):
        x1 = math.ceil(cp.polyval(coeff_time, pidx) * Config.fs)
        x2 = math.ceil(cp.polyval(coeff_time, pidx + 0.25) * Config.fs)
        nsymbr = cp.arange(x1, x2)
        tsymbr = nsymbr / Config.fs

        estcoef_this = cp.polyval(estfcoef_to_num, pidx)
        beta1 = - betai * (1 + 2 * estcoef_this / Config.sig_freq)
        estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
        beta2 = 2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) + estbw / 2) - cp.polyval(coeff_time, pidx) * 2 * beta1 # 2ax+b=differential b=differential - 2 * beta1 * time
        coef2d_est2 = cp.array([beta1.get(), beta2.get(), 0])
        coef2d_est2_2d = cp.polyval(coef2d_est2, cp.polyval(coeff_time, pidx)) - cp.polyval(
            coeffitlist[pidx - 1], cp.polyval(coeff_time, pidx))
        coef2d_est2[2] -= coef2d_est2_2d
        cd2 = cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr)) ))#!!!!!!!!!!!! TODO here we align phase for the last symbol
        logger.warning(f"WARN last phase {cd2=} manually add phase compensation")
        # coef2d_est2[2] += cd2
        # cd2 = cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr)) ))#!!!!!!!!!!!! TODO here we align phase for the last symbol
        # assert abs(cd2) < 1e-4
        coeffitlist[pidx] = coef2d_est2
        res2 = pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr)))
        codephase.append(cp.angle(res2).item())
        powers.append(cp.abs(res2).item() / cp.sum(cp.abs(pktdata_in[nsymbr])).item())
        fig=pltfig1(tsymbr, cp.angle(pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr))), title=f"residue {pidx=}", fig=fig)

    coeff_time[1] -= 0.75 * coeff_time[0]
    coeff_time[1] -= 2.5e-6 #!!!!TODO!!!!!a

    pidx_max = math.floor((len(pktdata_in)/Config.fs-coeff_time[1])/coeff_time[0])
    startphase = cp.polyval(coeffitlist[Config.preamble_len + 4], cp.polyval(coeff_time, Config.preamble_len + 5))
    for pidx in range(Config.preamble_len + 5, pidx_max): #TODO!!!!! Config.preamble_len + 5
        tstart = cp.polyval(coeff_time, pidx)
        tend = cp.polyval(coeff_time, pidx + 1)
        code, endphase, coef2d_est2, coef2d_est2a, res2, res2a = decode_core(pktdata_in, tstart, tend, estfcoef_to_num, startphase, pidx)
        startphase = endphase
        powers.append(cp.abs(res2).item())
        powers.append(cp.abs(res2a).item())
        codephase.append(cp.angle(res2).item())
        codephase.append(cp.angle(res2a).item())

        if pidx%10==0:
            pltfig1(None, cp.unwrap(codephase), title="unwrap phase").show()
            pltfig1(None, powers, title="powers").show()


def decode_core(pktdata_in, tstart, tend, estfcoef_to_num, startphase, pidx):
    x1 = math.ceil(tstart * Config.fs)
    x2 = math.ceil(tend * Config.fs)
    nsymbr = cp.arange(x1, x2)
    tsymbr = nsymbr / Config.fs
    estcoef_this = cp.polyval(estfcoef_to_num, pidx)

    beta1 = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi * (1 + 2 * estcoef_this / Config.sig_freq)
    estbw = Config.bw * (1 + estcoef_this / Config.sig_freq)
    beta2 = 2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) - estbw / 2) - tstart * 2 * beta1  # 2ax+b=differential b=differential - 2 * beta1 * time
    coef2d_est = cp.array([beta1.get(), beta2.get(), 0])

    sig2 = pktdata_in[nsymbr] * cp.exp(-1j * cp.polyval(coef2d_est, tsymbr))
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    freq1 = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
    freq, valnew = optimize_1dfreq(sig2, tsymbr, freq1)
    assert valnew > 0.9, f"{freq=} {freq1=} {valnew=}"
    if freq < 0: freq += estbw
    codex = freq / estbw * 2 ** Config.sf
    code = around(codex)

    tmid = tstart * (1 - code / 2 ** Config.sf) + tend * (code / 2 ** Config.sf)
    tmid = tmid.item()
    x3 = math.ceil(tmid * Config.fs)

    nsymbr1 = cp.arange(x1, x3)
    tsymbr1 = nsymbr1 / Config.fs
    nsymbr2 = cp.arange(x3, x2)
    tsymbr2 = nsymbr2 / Config.fs

    beta2 = (2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) + estbw * (code / 2 ** Config.sf - 0.5))
             - tstart * 2 * beta1)
    coef2d_est2 = cp.array([beta1.get(), beta2.get(), 0])
    coef2d_est2_2d = cp.polyval(coef2d_est2, tstart) - startphase
    coef2d_est2[2] -= coef2d_est2_2d

    beta2a = (2 * cp.pi * (cp.polyval(estfcoef_to_num, pidx) + estbw * (code / 2 ** Config.sf - 1.5))
              - tstart * 2 * beta1)
    coef2d_est2a = cp.array([beta1.get(), beta2a.get(), 0])
    coef2d_est2a_2d = cp.polyval(coef2d_est2a, tmid) - cp.polyval(coef2d_est2, tmid)
    coef2d_est2a[2] -= coef2d_est2a_2d

    res2 = pktdata_in[nsymbr1].dot(cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr1))) / cp.sum(cp.abs(pktdata_in[nsymbr1]))
    res2a = pktdata_in[nsymbr2].dot(cp.exp(-1j * cp.polyval(coef2d_est2a, tsymbr2))) / cp.sum(cp.abs(pktdata_in[nsymbr1]))

    pltfig1(tsymbr1, cp.angle(pktdata_in[nsymbr1] * cp.exp(-1j * cp.polyval(coef2d_est2, tsymbr1))), title=f"{pidx=} 1st angle {codex=}").show()

    assert cp.abs(res2).item() > 0.9, f"{pidx=} 1st power {cp.abs(res2).item()}<0.9"
    assert cp.abs(res2a).item() > 0.9, f"{pidx=} 2nd power {cp.abs(res2a).item()}<0.9"

    endphase = cp.polyval(coef2d_est2a, tend)
    return code, endphase, coef2d_est2, coef2d_est2a, res2, res2a


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
            coefficients_2d = cp.array([betat, coefficients_2d[0], coefficients_2d[1]])
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