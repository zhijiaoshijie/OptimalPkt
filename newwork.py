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

def plot_fit2d(coefficients_2d_in, estf, pidx, pktdata_in):
    logger.warning(f'''
    plt_fit2d called with {coefficients_2d_in=} {estf=} {pidx=}
    draw 3 imgs:
        1. the original 2d unwrapped, with fitted 2d poly curve
        2. the difference between them, unwrapped
        3. the original 2d nounwrapped, with fitted 2d poly curve wrapped in 
    ''')
    coefficients_2d = coefficients_2d_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    nstart = pidx * nsymblen
    tstart = nstart / Config.fs
    tend = (pidx + 1) * nsymblen / Config.fs
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
    pltfig(tsymbrp,
           (uarp, cp.polyval(coefficients_2d, tsymbrp)),
           title = f"{pidx=} fit 2d uwa curve",
           addvline = (tstart, tend)).show()
    pltfig1(tsymbrp,
            uarp - cp.polyval(coefficients_2d, tsymbrp),
            title=f"{pidx=} fit diff between 2d curve and uwa",
            mode='lines+markers',
            marker=dict(size=1),
            yaxisrange=[-2, 2]).show()
    fig = pltfig1(tsymbrp,
                  cp.angle(pktdata_in[nsymbrp]),
                  mode='markers',
                  title=f"{pidx=} fit 2d no-uw angles",
                  addvline=(tstart, tend))
    tinytsymbrp = cp.linspace(tstart, tend, int(1e6))
    pltfig1(tinytsymbrp,
            wrap(cp.polyval(coefficients_2d, tinytsymbrp)),
            fig = fig).show()


def pltfig(xdata, ydatas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, fig = None):
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if modes is None:
        modes = ['lines' for _ in ydatas]
    for ydata, mode in zip(ydatas, modes):
        fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker))
    pltfig_hind(addhline, addvline, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, fig = None):
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=tocpu(xdata), y=tocpu(ydata), mode=mode, marker=marker))
    pltfig_hind(addhline, addvline, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        for x in addvline: fig.add_vline(x=x)
    if addhline is not None:
        for y in addvline: fig.add_hline(y=y)