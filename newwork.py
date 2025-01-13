from utils import *
import plotly.express as px
import plotly.graph_objects as go

def coarse_est_f_t(data1, estf, window_idx):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi  # = xie lv bian hua lv / 2 = pin lv bian hua lv * 2pi / 2
    betat = betai * (1 + estf / Config.sig_freq)
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs

    nstart = window_idx * nsymblen
    tstart = nstart / Config.fs

    nsymbr = cp.arange(round(nstart), round(nstart + nsymblen))
    polyref = (betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0)
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

    nsymbr = cp.arange(round(nstart), round(nstart + nsymblen))
    polyref = (betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0)
    pow1 = cp.abs(data1[nsymbr].dot(cp.exp(-1j * cp.polyval(polyref, nsymbr / Config.fs))))

    for pidx in range(window_idx):
        nstart = pidx * nsymblen
        tstart = nstart / Config.fs

        nsymbr = cp.arange(round(nstart), round(nstart + nsymblen))
        polyref = (betat, (-Config.bw / 2 + estf) * 2 * cp.pi - 2 * betat * tstart, 0)
        pow2 = cp.abs(data1[nsymbr].dot(cp.exp(-1j * cp.polyval(polyref, nsymbr / Config.fs))))
        if pow2 > pow1 * 0.5:
            return pidx

def plot_fit2d(coefficients_2d_in, estf, estt, pidx, pktdata_in):
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
    tstart = nstart / Config.fs
    tend = tstart + tsymblen
    nsymbr = cp.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
    tsymbr = nsymbr / Config.fs
    p0idx = 1000
    nsymbrp = cp.arange(round(nstart) - p0idx, round(nstart) + Config.nsamp + 1000)
    assert nsymbrp[p0idx] == round(nstart)
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


def plot_fit2d_after_refine(coefficients_2d_in, coef2d_refined_in, estf, estt, pidx, pktdata_in):
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
    tstart = nstart / Config.fs
    tend = tstart + tsymblen
    nsymbr = cp.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
    tsymbr = nsymbr / Config.fs
    p0idx = 1000
    nsymbrp = cp.arange(round(nstart) - p0idx, round(nstart) + Config.nsamp + 1000)
    assert nsymbrp[p0idx] == round(nstart)
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

def pltfig(datas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, fig = None, line=None):
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if not len(datas[0] == 2):
        datas = [ (np.arange(len(datas[x])), datas[x]) for x in range(len(datas))]
    if modes is None:
        modes = ['lines' for _ in datas]
    elif isinstance(modes, str):
        modes = [modes for _ in datas]
    for idx, ((xdata, ydata), mode) in enumerate(zip(datas, modes)):
        if line == None and idx == 1: line = dict(dash='dash')
        fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker, line=line))
    pltfig_hind(addhline, addvline, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, fig = None, line=None):
    if xdata is None: xdata = np.arange(len(ydata))
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker, line=line))
    pltfig_hind(addhline, addvline, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        for x in addvline: fig.add_vline(x=x, line_dash='dash')
    if addhline is not None:
        for y in addhline: fig.add_hline(y=y, line_dash='dash')

def symbtime():
    # time diff
    diffs = []
    dd2 = []
    coeflist2 = coeflist.copy()
    dd31 = []
    for pidx in range(239):
        nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + nestt

        xva = np.arange(round(nstart) - 1000,
                        round(nsymblen * (pidx + 2) * (1 - estf / Config.sig_freq) + nestt) + 1000)
        yplt = np.zeros_like(pktdata_in, dtype=np.float64)
        yplt[xva] = np.unwrap(np.angle(tocpu(pktdata_in[xva])))
        val = (yplt[round(nstart) + 1000] - np.angle(pktdata_in[round(nstart) + 1000])) / 2 / np.pi
        # val = (yplt[round(nstart)+1000] - np.polyval(coeflist[pidx], x_data[round(nstart) + 1000])) / 2 / np.pi
        logger.warning(f"{val=}, {np.polyval(coeflist[pidx], x_data[round(nstart) + 1000])}")
        assert abs(val - round(val)) < 0.1
        dd2.append(val - round(val))
        coeflist[pidx, 2] += round(val) * 2 * np.pi

        start_pos_all_new2 = nsymblen * (pidx + 1) * (1 - estf / Config.sig_freq) + nestt
        start_pos2 = round(start_pos_all_new2)
        val2 = (yplt[start_pos2 + 1000] - np.angle(pktdata_in[start_pos2 + 1000])) / 2 / np.pi
        # val = (yplt[start_pos2+1000] - np.polyval(coeflist[pidx+1], x_data[start_pos2 + 1000])) / 2 / np.pi
        dd2.append(val2 - round(val2))
        logger.warning(f"{val2=}")
        assert abs(val2 - round(val2)) < 0.1
        coeflist[pidx + 1, 2] += round(val2) * 2 * np.pi
        dd31.append(round(val2) - round(val))

        coeffs_diff = np.polysub(coeflist[pidx], coeflist[pidx + 1])
        intersection_x_vals = np.roots(coeffs_diff)
        if len(intersection_x_vals) == 2:
            if abs(intersection_x_vals[0]) < abs(intersection_x_vals[1]):
                diffs.append(intersection_x_vals[0])
            else:
                diffs.append(intersection_x_vals[1])
        else:
            diffs.append(intersection_x_vals[0])

        coplt = [2 * coeflist[pidx, 0] * x_data[round(nstart)] + coeflist[pidx, 1], 0]
        coplt[1] = yplt[round(nstart)] - np.polyval(coplt, x_data[round(nstart)])
        xvb = np.arange(round(nstart) - 1000, round(nstart) + 1000)

        nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + nestt

        nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
        coef2d1 = np.polyfit(tsymbr, np.unwrap(np.angle(yplt[nsymbr])), 2)
        xvp = np.arange(round(nstart) - 100, round(nstart) + Config.nsamp + 100)
        fig.add_trace(go.Scatter(x=x_data[xvp], y=np.polyval(coeflist[pidx], x_data[xvp])))

        xv2 = np.arange(start_pos2 + 1000, start_pos2 + Config.nsamp - 1000)
        coef2d2 = np.polyfit(x_data[xv2], np.unwrap(np.angle(yplt[xv2])), 2)
        logger.warning(
            f"res:{(coef2d1[2] - coeflist[pidx, 2]) / 2 / np.pi}  {(coef2d2[2] - coeflist[pidx + 1, 2]) / 2 / np.pi}")

        xvp2 = np.arange(start_pos2 - 100, start_pos2 + Config.nsamp + 100)
        fig.add_trace(go.Scatter(x=x_data[xvp2], y=np.polyval(coeflist[pidx + 1], x_data[xvp2])))
        fig.add_vline(x=start_pos / Config.fs)
        fig.add_vline(x=start_pos2 / Config.fs)
        fig.show()
        coeflist = coeflist2.copy()
    plt.plot(dd2)
    plt.show()
    fig = go.Figure(layout_title_text="intersect points")
    pidx_range = np.arange(240)
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs))
    coeff_time = np.polyfit(pidx_range[1:], diffs, 1)
    print(coeff_time)
    print(
        f"estimated time:{coeff_time[0]} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=np.polyval(coeff_time, pidx_range[1:])))
    fig.show()
    fig = go.Figure(layout_title_text="intersect points diff")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs - np.polyval(coeff_time, pidx_range[1:])))
    fig.show()

    fig = go.Figure(layout_title_text="match val diff")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=dd31))
    fig.show()

    dd = []
    for pidx in range(240):
        dd.append(- coeflist[pidx, 1] / 2 / coeflist[pidx, 0] - np.polyval(coeff_time, pidx))
    fig = px.line(dd, title="coef2 middle time")
    fig.show()


def fitcoef(estf, estt, pktdata_in, fitmethod = "2dfit", searchquad = True):
    nestt = estt * Config.fs
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * np.pi
    betat = betai * (1 + estf / Config.sig_freq)

    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    # coeff_time = [0.01008263, 0.01015366]
    coeflist = []
    for pidx in range(0, Config.preamble_len):
        nstart = pidx * nsymblen + nestt
        nsymbr = cp.arange(round(nstart) + 1000, round(nstart + nsymblen) - 1000)
        tsymbr = nsymbr / Config.fs
        # nstart = np.polyval(coeff_time, pidx) * Config.fs

        #
        # 2d fit on unwrapped angle
        #

        if fitmethod == "2dfit":
            coefficients_2d = cp.polyfit(tsymbr, cp.unwrap(cp.angle(pktdata_in[nsymbr])), 2)
        elif fitmethod == "1dfit":
            coefficients_2d = cp.polyfit(
                tsymbr,
                cp.unwrap(cp.angle(pktdata_in[nsymbr])) - cp.polyval((betat, 0, 0), tsymbr),
                1
            )
            coefficients_2d = [betat, *coefficients_2d]
        logger.info(f"{pidx=} {nstart=} {fitmethod=} {coefficients_2d=} {betat=} {betai=}")

        # align with phase. here may have a 2pi difference when evaluated in unwrap
        coefficients_2d[-1] -= cp.angle(
            pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))

        #
        # plot the results:
        #
        # plot_fit2d(coefficients_2d, estf, estt, pidx, pktdata_in)

        #
        # find accurate coefficients_2d by power
        #
        coef2d_refined = refine_coef(estf, estt, pidx, pktdata_in, coefficients_2d, searchquad = searchquad)
        coeflist.append(coef2d_refined)

        # plot_fit2d_after_refine(coefficients_2d, coef2d_refined, estf, estt, pidx, pktdata_in)
    return cp.array(coeflist)

def refine_coef(estf, estt, pidx, pktdata_in, coefficients_2d_in, searchquad = False):
    coefficients_2d = coefficients_2d_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    nstart = pidx * nsymblen + estt * Config.fs
    nsymbr = cp.arange(round(nstart) + 1000, round(nstart + nsymblen) - 1000)
    tsymbr = nsymbr / Config.fs

    val = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
    logger.info(
        f'find accurate coefficients_2d by power. before start, result from linear fit {coefficients_2d=} {val=}')

    rangeval = 0.005
    for i in range(20):

        # search for both 1st (quad) and 2nd (linear) value, or only 2nd value
        for j in range(0 if searchquad else 1, 2):
            xvals = cp.linspace(coefficients_2d[j] * (1 - rangeval), coefficients_2d[j] * (1 + rangeval), 1001)
            yvals = []
            for x in xvals:
                coef2 = coefficients_2d.copy()
                coef2[j] = x
                yvals.append(obj(tsymbr, pktdata_in[nsymbr], coef2))
            oldv = coefficients_2d[j]
            coefficients_2d[j] = xvals[cp.argmax(yvals)]
            valnew = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
            if valnew < val*(1-1e-7):
                fig = go.Figure(layout_title_text=f"{pidx=} {i=} {j=} {val=} {valnew=}")
                fig.add_trace(go.Scatter(x=xvals, y=yvals))
                fig.add_vline(x=oldv)
                fig.show()
            assert valnew >= val * (1 - 1e-7), f"{val=} {valnew=} {i=} {j=} {coefficients_2d=} {val-valnew=}"
            if abs(valnew - val) < 1e-7: rangeval /= 2
            val = valnew

    # align with phase. here may have a 2pi difference when evaluated in unwrap
    coefficients_2d[-1] -= cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    return coefficients_2d