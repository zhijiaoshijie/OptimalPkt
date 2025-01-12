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
    coefficients_2d = coefficients_2d_in.copy()
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
    nstart = pidx * nsymblen
    tstart = nstart / Config.fs
    tend = (pidx + 1) * nsymblen / Config.fs
    nsymbr = cp.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
    tsymbr = nsymbr / Config.fs
    nsymbrp = cp.arange(round(nstart) - 1000, round(nstart) + Config.nsamp + 1000)
    tsymbrp = nsymbrp / Config.fs
    
    uarp = cp.unwrap(cp.angle(pktdata_in[nsymbrp]))
    coefficients_2d[-1] -= (cp.polyval(coefficients_2d, tsymbrp[1000]) - uarp[1000])
    coefficients_2d[-1] += cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    pltfig(tsymbrp,
           (uarp, cp.polyval(coefficients_2d, tsymbrp)),
           title = f"{pidx=} fit 2d uwa curve",
           addvline = (tstart, tend)).show()
    pltfig1(tsymbrp,
           uarp - cp.polyval(coefficients_2d, tsymbrp),
           title=f"{pidx=} fit diff between 2d curve and uwa",
           mode='markers',
           marker=dict(size=1),
           yaxisrange=[-2, 2]).show()
    fig = pltfig1(tsymbrp,
                 cp.angle(pktdata_in[nsymbrp]),
                 mode='markers',
                 title=f"{pidx=} fit 2d no-uw angles",
                 addvline=(tstart, tend))
    tinytsymbrp = cp.linspace(tstart, tend, 1e6)
    pltfig1(tinytsymbrp,
           cp.angle(cp.polyval(coefficients_2d, tinytsymbrp)),
           fig = fig).show()

    coefficients_2d[-1] -= cp.angle(pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))
    coefficients_2d_old = coefficients_2d.copy()
    return coefficients_2d_old, nstart, nsymblen, nsymbr, x_data, nsymbrp, y_data


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