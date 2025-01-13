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

def show_fit_results(pktdata_in, estf, estt, coeflist_in, pkt_idx):
    coeflist = coeflist_in.copy()
    betai = Config.bw / ((2 ** Config.sf) / Config.bw)
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    tsymblen = nsymblen / Config.fs
    for pidx in range(0, Config.preamble_len):
        nstart = pidx * nsymblen + estt * Config.fs
        nsymbr = cp.arange(around(nstart) + 1000, around(nstart + nsymblen) - 1000)
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
    nsymbr = cp.arange(around(nstart) + 1000, around(nstart) + Config.nsamp - 1000)
    tsymbr = nsymbr / Config.fs
    p0idx = 1000
    nsymbrp = cp.arange(around(nstart) - p0idx, around(nstart) + Config.nsamp + 1000)
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
    nsymbr = cp.arange(around(nstart) + 1000, around(nstart) + Config.nsamp - 1000)
    tsymbr = nsymbr / Config.fs
    p0idx = 1000
    nsymbrp = cp.arange(around(nstart) - p0idx, around(nstart) + Config.nsamp + 1000)
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

def pltfig(datas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    datas : list of tuples
        Each tuple contains two lists or array-like elements, the data for the x and y axes.
        If only y data is provided, x data will be generated using np.arange.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if not all(len(data) == 2 for data in datas): datas = [(np.arange(len(data)), data) for data in datas]
    if modes is None:
        modes = ['lines' for _ in datas]
    elif isinstance(modes, str):
        modes = [modes for _ in datas]
    for idx, ((xdata, ydata), mode) in enumerate(zip(datas, modes)):
        if line == None and idx == 1: line = dict(dash='dash')
        fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker, line=line))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    xdata : list or array-like or None
        The data for the x-axis.
        If is None, and only y data is provided, x data will be generated using np.arange.
    ydata : list or array-like
        The data for the y-axis.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if xdata is None: xdata = np.arange(len(ydata))
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker, line=line))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        for x in addvline: fig.add_vline(x=x, line_dash='dash')
    if addhline is not None:
        if line_dash is None:
            line_dash = ['dash' for _ in range(len(addhline))]
        elif isinstance(line_dash, str):
            line_dash = [line_dash for _ in range(len(addhline))]
        for y, ldash in zip(addhline, line_dash): fig.add_hline(y=y, line_dash=ldash)

def symbtime(estf, estt, pktdata_in, coeflist):
    nestt = estt * Config.fs
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    diffs = []
    dd2 = []
    coeflist2 = coeflist.copy()
    dd31 = []
    for pidx in range(0, Config.preamble_len - 1):
        nstart = pidx * nsymblen + nestt
        tstart = nstart / Config.fs
        nsymbr = cp.arange(around(nstart) + 1000, around(nstart + nsymblen) - 1000)
        tsymbr = nsymbr / Config.fs
        
        nsymba = cp.arange(around(nstart) - 1000, around(nstart + nsymblen * 2) + 1000)
        tsymba = nsymba / Config.fs
        fig = go.Figure(layout_title_text=f"{pidx=} symb")
        ysymba = cp.zeros_like(pktdata_in, dtype=cp.float64)
        ysymba[nsymba] = cp.unwrap(cp.angle(pktdata_in[nsymba]))
        nrstart = around(nstart + 1000)
        val = (ysymba[nrstart] - cp.angle(pktdata_in[nrstart])) / 2 / cp.pi
        # val = (ysymba[nstart+1000] - cp.polyval(coeflist[pidx], x_data[nstart + 1000])) / 2 / cp.pi
        logger.warning(f"{val=}, {np.polyval(coeflist[pidx], nrstart/Config.fs)}")
        assert abs(val - around(val)) < 0.0001, f'1st uwrap from {nstart=} at {nrstart+1000=} {val=} not int'
        dd2.append(val - around(val))
        coeflist[pidx, 2] += around(val) * 2 * cp.pi

        nstart2 = pidx * nsymblen + nestt
        nrstart2 = around(nstart2 + 1000)
        val = (ysymba[nrstart2] - cp.angle(pktdata_in[nrstart2])) / 2 / cp.pi
        # val = (ysymba[nrstart2] - cp.polyval(coeflist[pidx+1], x_data[nrstart2])) / 2 / cp.pi
        dd2.append(val - around(val))
        logger.warning(f"{val=}")
        assert abs(val - around(val)) < 0.0001
        coeflist[pidx + 1, 2] += around(val) * 2 * cp.pi

        coeffs_diff = cp.polysub(coeflist[pidx], coeflist[pidx + 1])
        intersection_x_vals = cp.roots(coeffs_diff)
        if len(intersection_x_vals) == 2:
            if abs(intersection_x_vals[0]) < abs(intersection_x_vals[1]):
                diffs.append(intersection_x_vals[0])
            else:
                diffs.append(intersection_x_vals[1])
        else:
            diffs.append(intersection_x_vals[0])

        fig.add_trace(go.Scatter(x=tsymba, y=ysymba[nsymba]))
        coplt = [2 * coeflist[pidx, 0] * tstart + coeflist[pidx, 1], 0]
        coplt[1] = ysymba[around(nstart)] - cp.polyval(coplt, tstart)
        xvb = cp.arange(around(nstart) - 1000, around(nstart) + 1000)
        fig.add_trace(go.Scatter(x=x_data[xvb], y=cp.polyval(coplt, x_data[xvb])))
        fig.add_vline(x=diffs[-1], line_dash="dash", line_color="green")

        xv = cp.arange(around(nstart) + 1000, around(nstart) + Config.nsamp - 1000)
        coef2d1 = cp.polyfit(x_data[xv], cp.unwrap(cp.angle(ysymba[xv])), 2)
        xvp = cp.arange(around(nstart) - 100, around(nstart) + Config.nsamp + 100)
        fig.add_trace(go.Scatter(x=x_data[xvp], y=cp.polyval(coeflist[pidx], x_data[xvp])))

        xv2 = cp.arange(nrstart2, nstart2 + Config.nsamp - 1000)
        coef2d2 = cp.polyfit(x_data[xv2], cp.unwrap(cp.angle(ysymba[xv2])), 2)
        logger.warning(
            f"res:{(coef2d1[2] - coeflist[pidx, 2]) / 2 / cp.pi}  {(coef2d2[2] - coeflist[pidx + 1, 2]) / 2 / cp.pi}")

        xvp2 = cp.arange(nstart2 - 100, nstart2 + Config.nsamp + 100)
        fig.add_trace(go.Scatter(x=x_data[xvp2], y=cp.polyval(coeflist[pidx + 1], x_data[xvp2])))
        fig.add_vline(x=nstart / Config.fs)
        fig.add_vline(x=nstart2 / Config.fs)
        fig.show()
        coeflist = coeflist2.copy()
    plt.plot(dd2)
    plt.show()
    fig = go.Figure(layout_title_text="intersect points")
    pidx_range = cp.arange(240)
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
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * cp.pi
    betat = betai * (1 + estf / Config.sig_freq)

    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    # coeff_time = [0.01008263, 0.01015366]
    coeflist = []
    for pidx in range(0, Config.preamble_len):
        nstart = pidx * nsymblen + nestt
        nsymbr = cp.arange(around(nstart) + 1000, around(nstart + nsymblen) - 1000)
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
    nsymbr = cp.arange(around(nstart) + 1000, around(nstart + nsymblen) - 1000)
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
            coefficients_2d[j] = xvals[cp.argmax(togpu(yvals))]
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