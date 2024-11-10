import argparse
import logging
import os
import random
import sys
import time
import itertools
import pickle
import cmath
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# import pandas as pd
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
# import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm
# import scipy
import cupyx.scipy.fft as fft

logger = logging.getLogger('my_logger')
level = logging.DEBUG
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)  # Set the console handler level
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(level)  # Set the file handler level
formatter = logging.Formatter('%(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', default=False, help='Use cpu instead of gpu (numpy instead of cupy)')
parser.add_argument('--pltphase', action='store_true', default=False)
parser.add_argument('--searchphase', action='store_true', default=False)
parser.add_argument('--refine', action='store_true', default=False)
parser.add_argument('--searchphase_step', type=int, default=1000000)
parser.add_argument('--plotmap', action='store_true', default=False)
parser.add_argument('--plotline', action='store_true', default=False)
parser.add_argument('--end1', action='store_true', default=False)
parser.add_argument('--savefile', action='store_true', default=False)
parser.add_argument('--noplot', action='store_true', default=False)
parser.add_argument('--fromfile', type=str, default=None)
parser.add_argument('--compLTSNR', action='store_true', default=False)
parser.add_argument('--decode_unknown', action='store_true', default=False)
parser.add_argument('--snrlow', type=int, default=-20)
parser.add_argument('--snrhigh', type=int, default=-19)
parse_opts = parser.parse_args()

use_gpu = not parse_opts.cpu
if use_gpu:
    import cupy as cp
    import cupyx.scipy.fft as fft
else:
    import numpy as cp
    import scipy.fft as fft


def togpu(x):
    if use_gpu and not isinstance(x, cp.ndarray):
        return cp.array(x)
    else:
        return x


def tocpu(x):
    if use_gpu and isinstance(x, cp.ndarray):
        return x.get()
    else:
        return x


def mychirp(t, f0, f1, t1):
    beta = (f1 - f0) / t1
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t)
    sig = cp.exp(1j * togpu(phase))
    return sig


def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda k: f"{k:.2f}"}, floatmode='fixed', suppress_small=suppress_small)


def myscatter(ax, x, y, **kwargs):
    ax.scatter(tocpu(x), tocpu(y), **kwargs)


def myplot(*args, **kwargs):
    if len(args) == 2:
        ax = args[0]
        ax.plot(tocpu(args[1]), **kwargs)
    elif len(args) == 3:
        ax = args[0]
        ax.plot(tocpu(args[1]), tocpu(args[2]), **kwargs)
    else:
        raise ValueError("plot function accepts either 1 or 2 positional arguments")


def myfft(chirp_data, n, plan):
    return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n, plan=plan))


script_path = __file__
mod_time = os.path.getmtime(script_path)
readable_time = time.ctime(mod_time)
logger.info(f"Last modified time of the script: {readable_time}")


def average_modulus(lst, n_classes):
    complex_sum = cp.sum(cp.exp(1j * 2 * cp.pi * cp.array(lst) / n_classes))
    avg_angle = cp.angle(complex_sum)
    avg_modulus = (avg_angle / (2 * cp.pi)) * n_classes
    return avg_modulus


class Config:
    sf = 12
    bw = 406250#*(1-20*1e-6)
    fs = 1e6
    sig_freq = 2.4e9
    # sig_freq = 2400000030.517578#-52e6/(2**18)
    preamble_len = 16  # TODO!!!!

    thresh = None# 0.03
    file_paths = ['/data/djl/temp/OptimalPkt/fingerprint_data/data0_test_3',]# '/data/djl/temp/OptimalPkt/fingerprint_data/data1_test_3']

    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = round(n_classes * fs / bw)
    # f_lower, f_upper = -50000, -30000
    f_lower, f_upper = -41000, -38000
    t_lower, t_upper = 0, nsamp
    fguess = (f_lower + f_upper) / 2
    tguess = nsamp / 2
    code_len = 2

    cfo_freq_est = -39705.621
    time_error = 4321.507

    time_r = time_error / tsig
    cfo_r = cfo_freq_est / bw
    print(time_r + cfo_r, cfo_r - time_r)


    fguess = cfo_freq_est
    tguess = time_error
    f_lower = fguess - 500
    f_upper = fguess + 500
    t_lower = tguess - 50
    t_upper = tguess + 50

    tbeta = (0.00025319588796389843+0.0002531603667156756)/2
    fbeta = (-6.283185307178473e-06-6.302329387166181e-06)/2
    # fguess = -39000
    # tguess = 4320

    gen_refchirp_deadzone = 0
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)

    fft_upsamp = 1024
    detect_range_pkts = 3
    fft_n = nsamp * fft_upsamp
    plan = fft.get_fft_plan(cp.zeros(fft_n, dtype=cp.complex64))
    fft_ups = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)


if use_gpu:
    cp.cuda.Device(0).use()
Config = Config()


def gen_upchirp(t0, td, f0, beta):
    # start from ceil(t0in), end
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t) / Config.fs
    sig = cp.exp(1j * phase)
    return sig

def converter_down(cfofreq, time_error):
    cfofreq -= (Config.f_upper + Config.f_lower) / 2
    cfofreq /= (Config.f_upper - Config.f_lower) / 2
    time_error -= (Config.t_upper + Config.t_lower) / 2
    time_error /= (Config.t_upper - Config.t_lower) / 2
    return cfofreq, time_error

def converter_up(cfofreq, time_error):
    cfofreq *= (Config.f_upper - Config.f_lower) / 2
    cfofreq += (Config.f_upper + Config.f_lower) / 2
    time_error *= (Config.t_upper - Config.t_lower) / 2
    time_error += (Config.t_upper + Config.t_lower) / 2
    return cfofreq, time_error

def objective(params, pktdata2a):
    cfofreq, time_error = converter_up(*params)
    return objective_core(cfofreq, time_error, pktdata2a)


def objective_core(cfofreq, time_error, pktdata2a):
    assert pktdata2a.ndim == 1
    assert cp.mean(cp.abs(pktdata2a)).ndim == 0
    pktdata2a_roll = cp.roll(pktdata2a / cp.mean(cp.abs(pktdata2a)), -math.ceil(time_error))
    detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=Config.gen_refchirp_deadzone)
    res = cp.zeros(len(detect_symb), dtype=cp.complex64)
    ddx = 0  # TODO
    for sidx, ssymb in enumerate(detect_symb):
        ress = cp.conj(ssymb).dot(pktdata2a_roll[ddx: ddx + len(ssymb)])
        ddx += len(ssymb)
        res[sidx] = ress / len(ssymb)
    # print(cp.mean(cp.abs(pktdata2a[:ddx])))  # Negative because we use a minimizer
    # TODO qian zhui he plot
    # TODO remove **2 because res[sidx] is sum not sumofsquare
    # TODO phase consistency?
    return - tocpu(cp.abs(cp.sum(res)) / len(res))

def fine_work_new(pktidx, pktdata2a):
    pktdata2a = togpu(pktdata2a)
    # d2 = cp.diff(cp.angle(pktdata2a)).get()
    # d2 = d2[:Config.nsamp]
    # plt.scatter(range(len(d2)), d2)
    # plt.savefig("1.png")
    # phase = cp.angle(pktdata2a)
    # unwrapped_phase = cp.unwrap(phase)
    # fig = px.line(y=tocpu(unwrapped_phase[:Config.nsamp]), title=f"input data 15 symbol {pktidx=}")
    # # if not parse_opts.noplot: fig.show()
    # fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} input_data.html"))

    # draw_fit(pktidx, pktdata2a, Config.fguess, Config.tguess)
    print('obj0', objective_core(Config.fguess, Config.tguess, pktdata2a))
    if True:
        yval1, yval2 = draw2(pktdata2a, Config.fguess, Config.tguess)
        t1 = -(yval1-yval2)/2/Config.tbeta
        f1 = -(yval1+yval2)/2/Config.fbeta
        print('dtdf', t1, f1)
        print('obj1',objective_core(Config.fguess+f1, Config.tguess+t1, pktdata2a))
        print('draw2slope',draw2(pktdata2a, Config.fguess+f1, Config.tguess+t1))
        draw_fit(pktidx, pktdata2a, Config.fguess+f1, Config.tguess+t1)
        Config.fguess += f1
        Config.tguess += t1

        xvals = np.linspace(Config.tguess - 10, Config.tguess + 10, 100)
        yvals = np.array([ draw2(pktdata2a, Config.fguess, x) for x in xvals])
        fig = px.line(x=xvals, y=yvals[:, 0])
        fig.add_trace(go.Scatter(x=xvals, y=yvals[:, 1]))
        fig.show()
        m1, b1 = np.polyfit(xvals, yvals[:,0], 1)
        m2, b2 = np.polyfit(xvals, yvals[:,1], 1)
        print('FIT', m1, m2)
        xvals = np.linspace(Config.fguess - 100, Config.fguess + 100, 100)
        yvals = np.array([ draw2(pktdata2a, x, Config.tguess) for x in xvals])
        fig = px.line(x=xvals, y=yvals[:, 0])
        fig.add_trace(go.Scatter(x=xvals, y=yvals[:, 1]))
        fig.show()
        m1, b1 = np.polyfit(xvals, yvals[:,0], 1)
        m2, b2 = np.polyfit(xvals, yvals[:,1], 1)
        print('FIT', m1, m2)
        sys.exit(0)

    if True:#parse_opts.pltphase:
        phase = cp.angle(pktdata2a)
        unwrapped_phase = cp.unwrap(phase)
        fig = px.line(y=tocpu(unwrapped_phase[:30 * Config.nsamp]), title=f"input data 15 symbol {pktidx=}")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} input_data.html"))

    # Perform optimization


    if parse_opts.searchphase:
        # Config.f_lower, Config.f_upper = Config.fguess - 3000, Config.fguess + 3000
        bestobj = objective_core(Config.fguess, Config.tguess, pktdata2a)
        logger.debug(
            f"trystart cfo_freq_est = {Config.fguess:.3f}, time_error = {Config.tguess:.3f} {bestobj=} {Config.f_lower=} {Config.f_upper=} {Config.t_lower=} {Config.t_upper=}")
        draw_fit(pktidx, pktdata2a, Config.fguess, Config.tguess)
        # for tryidx in tqdm(range(parse_opts.searchphase_step), disable=True):
        tryidx = 0
        while True:
            tryidx += 1
            start_t = random.uniform(Config.t_lower, Config.t_upper)
            start_f = random.uniform(Config.f_lower, Config.f_upper)
            # noinspection PyTypeChecker
            # print([(converter_down(Config.f_lower, 0)[0], converter_down(Config.f_upper, 0)[0]),
            #           (converter_down(0, Config.t_lower)[1], converter_down(0, Config.t_upper)[1])], converter_down(start_f, start_t), objective(converter_down(start_f, start_t), pktdata2a), objective_core(start_f, start_t, pktdata2a))
            result = opt.minimize(objective, converter_down(start_f, start_t), args=(pktdata2a,),
                                  bounds=[(converter_down(Config.f_lower,0)[0],converter_down(Config.f_upper,0)[0]), (converter_down(0,Config.t_lower)[1], converter_down(0,Config.t_upper)[1])],
                                  method='L-BFGS-B',
                                  options={'gtol': 1e-12, 'disp': False}
                                  )

            if result.fun < bestobj:
                cfo_freq_est, time_error = converter_up(*result.x)
                logger.debug(f"{tryidx=: 6d} cfo_freq_est = {cfo_freq_est:.3f}, time_error = {time_error:.3f} {result.fun=:.5f}")
                # Config.fguess, Config.tguess = result.x
                bestobj = result.fun
                if tryidx > 100: draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error)
            if tryidx == 100: draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error)
        logger.info(f"Optimized parameters:\n{cfo_freq_est=}\n{time_error=}")
    else:
        cfo_freq_est = Config.fguess
        time_error = Config.tguess
        draw_fit(pktidx, pktdata2a, Config.fguess, Config.tguess)

    if parse_opts.refine:
        cfo_search = 1000
        to_search = 40
        search_steps = 10000
        ttx = 1
        for searchidx in range(10):
            xrange = cp.linspace(cfo_freq_est - cfo_search, cfo_freq_est + cfo_search, search_steps)
            yvals = [objective_core(x, time_error, pktdata2a) for x in xrange]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tocpu(xrange)[::ttx], y=yvals[::ttx], mode='lines', name='Input Data'))
            fig.add_vline(x=cfo_freq_est)
            fig.update_layout(title=f"pkt{pktidx} {searchidx=} {cfo_search=} {cfo_freq_est=} {time_error=} objective fit frequency")
            cfo_freq_est_new = xrange[np.argmin(np.array(yvals))]
            cfo_freq_est_delta = abs(cfo_freq_est_new - cfo_freq_est)
            cfo_freq_est = cfo_freq_est_new
            fig.add_vline(x=cfo_freq_est, line_dash="dash", line_color="red")
            if not parse_opts.noplot: fig.show()

            xrange = cp.linspace(time_error - to_search, time_error + to_search, 1000)
            yvals = [objective_core(cfo_freq_est, x, pktdata2a) for x in xrange]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tocpu(xrange)[::ttx], y=yvals[::ttx], mode='lines', name='Input Data'))
            fig.add_vline(x=time_error)
            fig.update_layout(title=f"pkt{pktidx} {searchidx=} {to_search=} objective fit timeerror")
            time_error_new = xrange[np.argmin(np.array(yvals))]
            time_error_delta = abs(time_error_new - time_error)
            time_error = time_error_new
            fig.add_vline(x=time_error, line_dash="dash", line_color="red")
            if not parse_opts.noplot: fig.show()
            # if cfo_freq_est_delta > cfo_search / 2: cfo_search *= 3
            # else: cfo_search = (cfo_search + cfo_freq_est_delta) / 2
            # if time_error_delta > to_search / 2: to_search *= 3
            # else: to_search = (to_search + time_error_delta) / 2
            logger.info(f"pkt{pktidx} {searchidx=} {cfo_freq_est=} {time_error=} objective={min(yvals)} fit")
            draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error)
    logger.info(f"FIN pkt{pktidx} {cfo_freq_est=} {time_error=}")



def draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error):
    # return
    # fig = px.scatter(cp.diff(cp.angle(pktdata2a)[:Config.nsamp*2]).get())
    # fig.show()
    # fig = px.scatter(cp.diff(cp.unwrap(cp.angle(pktdata2a))[:Config.nsamp*2]).get())
    # fig.show()
    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    logger.info(f"{cfo_freq_est=:.3f}, time_error = {time_error:.3f}")
    detect_symb_plt = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    tsfd = sum([len(x) for x in detect_symb_plt[:-3]])
    detect_symb_plt = cp.concatenate(detect_symb_plt)
    # detect_symb_plt *= (pktdata2a_roll[0] / cp.abs(pktdata2a_roll[0]))
    phase1 = cp.angle(pktdata2a)
    xval = cp.arange(len(detect_symb_plt)) + math.ceil(time_error)
    xval2 = cp.arange(len(detect_symb_plt) + Config.nsamp+math.ceil(time_error))
    # xval = cp.arange(len(detect_symb_plt))
    yval1 = cp.unwrap(phase1)
    yval2 = cp.unwrap(cp.angle(detect_symb_plt))
    # tsfd = time_error - math.ceil(time_error) + Config.sfdpos * Config.tsig * (1 + cfo_freq_est / Config.sig_freq)
    yval2[math.ceil(tsfd):] += (yval1[math.ceil(tsfd) + math.ceil(time_error)] - yval2[math.ceil(tsfd)])
    if True:
        fig = go.Figure()
        # view_len = 60
        fig.add_trace(go.Scatter(x=tocpu(xval2), y=tocpu(yval1[xval2]), mode='lines', name='input', line=dict(color='blue')))
        fig.add_trace(
            go.Scatter(x=tocpu(xval), y=tocpu(yval2), mode='lines', name='fit', line=dict(dash='dash', color='red')))
        # fig.add_trace(go.Scatter(
        #     x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
        #     y=[0, Config.f_lower * 2 * np.pi / Config.fs * len(detect_symb_plt)],
        #     mode='lines',
        #     line=dict(color='gray', dash='dash'),
        #     showlegend=False
        # ))
        # fig.add_trace(go.Scatter(
        #     x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
        #     y=[0, Config.f_upper * 2 * np.pi / Config.fs * len(detect_symb_plt)],
        #     mode='lines',
        #     line=dict(color='gray', dash='dash'),
        #     showlegend=False
        # ))
        fig.add_trace(go.Scatter(
            x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
            y=[0, tocpu(cfo_freq_est) * 2 * np.pi / Config.fs * len(detect_symb_plt)],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend = False
        ))
        fig.update_layout(title=f'{pktidx} f={cfo_freq_est:.3f} t={time_error:.3f} obj={objective_core(cfo_freq_est, time_error, pktdata2a):.5f}', legend=dict(x=0.1, y=1.1))
        if not parse_opts.noplot: fig.show()
    if True:
        fig = go.Figure()  # px.line(tocpu(yval1[xval] - yval2[xval])[:3 * Config.nsamp])
        fig.add_vline(Config.nsamp)
        fig.add_vline(Config.nsamp*Config.preamble_len)
        fig.add_vline(Config.nsamp*(Config.sfdpos + 2.25))
        # fig.add_vline(Config.nsamp * 2)
        length = len(detect_symb_plt)
        print(pktdata2a.shape, detect_symb_plt.shape)
        y = tocpu(cp.unwrap(cp.angle(pktdata2a[math.ceil(time_error):length + math.ceil(time_error)] * detect_symb_plt[:length].conj())))
        # y[abs(y)>2000] = 0
        x = np.arange(0, len(y), 1)
        fig.add_trace(
            go.Scatter(x=x, y=y[x], mode="lines", #marker=dict(symbol='circle', size=0.5),
                       showlegend=False))
        fig.show()

    if False:
        fig = go.Figure()#px.line(tocpu(yval1[xval] - yval2[xval])[:3 * Config.nsamp])
        fig.add_vline(Config.nsamp)
        fig.add_vline(Config.nsamp * 2)
        length = len(detect_symb_plt)
        y = tocpu(yval1[math.ceil(time_error):length+math.ceil(time_error)] - yval2[:length])
        # fig.add_trace( go.Scatter(y=y, mode="markers",marker=dict(symbol='circle', size=0.5), showlegend=False))

        # First range for fit
        lst1 = []
        lst2 = []
        for idx in range(Config.sfdend-1):
            if idx >= Config.preamble_len and idx < Config.preamble_len+2: continue
            x1 = np.arange(int(Config.nsamp * (0.1+idx)), int(Config.nsamp * (0.9+idx)))
            m1, b1 = np.polyfit(x1, y[x1], 1)
            # fig.add_trace( go.Scatter(x=x1, y=m1 * x1 + b1, mode="lines", line=dict(color='red', dash='dash'), showlegend=False))
            print("Fit", idx, m1)
            if idx < Config.preamble_len: lst1.append(m1)
            else: lst2.append(m1)
        m1, b1 = np.polyfit(np.arange(len(lst1)), lst1, 1)
        m2, b2 = np.polyfit(np.arange(len(lst2)), lst2, 1)
        # fig.show()
        return m1, m2
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=y[Config.nsamp*Config.sfdpos+100:Config.nsamp * (Config.sfdpos+2)], mode="markers",marker=dict(symbol='circle', size=0.5), showlegend=False))

        # Second range for fit
        x2 = np.arange(int(Config.nsamp * (0.1 + Config.sfdpos)), int(Config.nsamp * (0.9 + Config.sfdpos)))
        # print(x2)
        m2, b2 = np.polyfit(x2, y[x2], 1)
        print('FIT', m1, m2)
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1
        # fig.add_vline(x=x_intersect)
        # print('xxxx', x_intersect, y_intersect)
        fig.add_trace(
            go.Scatter(x=x2-Config.nsamp*Config.sfdpos+100, y=m2 * x2 + b2, mode="lines", line=dict(color='red', dash='dash'), showlegend=False))

        # fig.add_trace(go.Scatter(x=tocpu(xval2) - math.ceil(time_error), y=tocpu(yval1[xval2]), mode='lines', name='input', line=dict(color='blue')))
        # fig.add_trace(
        #     go.Scatter(x=tocpu(xval) - math.ceil(time_error), y=tocpu(yval2), mode='lines', name='fit', line=dict(dash='dash', color='red')))
        # print(cp.argmin(yval1[:Config.nsamp + math.ceil(time_error)]))
        # fig.add_hline(y=cp.min(yval1[:Config.nsamp + math.ceil(time_error)]).get())
        # fig.add_vline(x=cp.argmin(yval1[:Config.nsamp + math.ceil(time_error)]).get() - math.ceil(time_error))



        fig.show()
        print(f'{pktidx} f={cfo_freq_est:.3f} t={time_error:.3f} obj={objective_core(cfo_freq_est, time_error, pktdata2a):.5f}')


def draw2(pktdata2a, cfo_freq_est, time_error):
    detect_symb_plt = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    tsfd = sum([len(x) for x in detect_symb_plt[:-3]])
    detect_symb_plt = cp.concatenate(detect_symb_plt)
    phase1 = cp.angle(pktdata2a)
    yval1 = cp.unwrap(phase1)
    yval2 = cp.unwrap(cp.angle(detect_symb_plt))
    yval2[math.ceil(tsfd):] += (yval1[math.ceil(tsfd) + math.ceil(time_error)] - yval2[math.ceil(tsfd)])
    length = len(detect_symb_plt)
    y = tocpu(yval1[math.ceil(time_error):length+math.ceil(time_error)] - yval2[:length])

    lst1 = []
    lst2 = []
    for idx in range(Config.sfdend - 1):
        if idx >= Config.preamble_len and idx < Config.preamble_len + 2: continue
        x1 = np.arange(int(Config.nsamp * (0.1 + idx)), int(Config.nsamp * (0.9 + idx)))
        m1, b1 = np.polyfit(x1, y[x1], 1)
        # fig.add_trace( go.Scatter(x=x1, y=m1 * x1 + b1, mode="lines", line=dict(color='red', dash='dash'), showlegend=False))
        # print("Fit", idx, m1)
        if idx < Config.preamble_len:
            lst1.append(m1)
        else:
            lst2.append(m1)
        print(m1, b1, idx)
    y_values = lst1
    x_values = np.arange(1, len(y_values) + 1)  # x values from 1 to n

    # Fit a polynomial of degree d (e.g., 2)
    degree = 2
    coefficients = np.polyfit(x_values, y_values, degree)

    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate predicted y values
    predicted_y = polynomial(x_values)

    # Plot the original y values and the fitted polynomial
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'o', label='Original y values')
    plt.plot(x_values, predicted_y, '-', label='Fitted polynomial')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title('Polynomial Fit to y values')
    plt.legend()
    plt.show()

    m1, b1 = np.polyfit(np.arange(len(lst1)), lst1, 1)
    m2, b2 = np.polyfit(np.arange(len(lst2)), lst2, 1)
    # fig.show()
    return m1, m2

def gen_refchirp(cfofreq, tstart, deadzone=0):
    detect_symb = []
    bw = Config.bw * (1 + cfofreq / Config.sig_freq)
    sigt = 2 ** Config.sf / bw * Config.fs #* (1 - cfofreq / Config.sig_freq)
    beta = bw / sigt
    # print(Config.bw / Config.tsig * (1 + cfofreq / Config.sig_freq) / (1 - cfofreq / Config.sig_freq))
    # print(Config.bw / Config.tsig * (1 + cfofreq / Config.sig_freq) )
    # print(Config.bw / Config.tsig )
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt, -bw  / 2 + cfofreq, beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tstart + sigt * (tid + 1)) - math.ceil(tstart + sigt * tid), dtype=cp.complex64))
    for tid in range(Config.sfdpos, Config.sfdend):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt if tid != Config.sfdend - 1 else sigt / 4,
                              bw / 2 + cfofreq, - beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    return detect_symb




def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
            except EOFError:
                logger.warning("file complete with EOF")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"file complete, {len(rawdata)=}")
                break
            yield rawdata



def read_pkt(file_path_in1, file_path_in2, threshold, min_length=20):
    current_sequence1 = []
    current_sequence2 = []

    for rawdata1, rawdata2 in itertools.zip_longest(read_large_file(file_path_in1), read_large_file(file_path_in2)):
        if rawdata1 is None or rawdata2 is None:
            break  # Both files are done

        number1 = cp.max(cp.abs(rawdata1)) if rawdata1 is not None else 0

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
            current_sequence2.append(rawdata2)
        else:
            if len(current_sequence1) > min_length:
                yield cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)
            current_sequence1 = []
            current_sequence2 = []

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * cp.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(cp.complex64)
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a

def coarse_work_fast(pktdata_in, retpflag = False):
    # pktdata_in = cp.roll(pktdata_in, 1000) #this makes t - 1000
    # pktdata_in = add_freq(pktdata_in, 1000) #this makes f + 1000
    est_to_s = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    upchirp = mychirp(est_to_s, f0=-Config.bw / 2, f1=Config.bw / 2, t1=2 ** Config.sf / Config.bw)
    downchirp = mychirp(est_to_s, f0=Config.bw / 2, f1=-Config.bw / 2, t1=2 ** Config.sf / Config.bw)
    plotflag = False
    ld = round(Config.bw / Config.fs * Config.fft_n)
    fups = []
    fret = []
    fdowns = []
    for pidx in range(Config.preamble_len + Config.detect_range_pkts):
        sig1 = pktdata_in[Config.nsamp * pidx: Config.nsamp * (pidx + 1)]
        sig2 = sig1 * downchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        data = cp.abs(data0) + cp.abs(cp.roll(data0, -ld))
        Config.fft_ups[pidx] = data
        # data2 = data.copy()
        # data2[cp.argmax(cp.abs(data)) - 2000 :cp.argmax(cp.abs(data)) + 2000 ]= 0
        # print(cp.max(cp.abs(data2)), cp.argmax(cp.abs(data2)))
        amax = cp.argmax(cp.abs(data))
        fups.append(amax.get())
        fret.append(cp.angle(data0[ld:][amax]).get())
        fret.append(cp.angle(data0[-ld:][amax]).get())


        if False:
            y_values = cp.unwrap(cp.angle(sig1)).get()[3000:]
            x_values = np.arange(len(y_values)) + 3000  # x values from 1 to n
            degree = 2
            coefficients = np.polyfit(x_values, y_values, degree)
            print('c', pidx, coefficients)
            # plt.plot(cp.unwrap(cp.angle(sig1)).get())
            # plt.plot(cp.unwrap(cp.angle(upchirp)).get())
            # polynomial = np.poly1d(coefficients)
            # predicted_y = polynomial(x_values)
            # plt.figure(figsize=(10, 6))
            # plt.plot(x_values, y_values, '-', label='Original y values')
            # plt.plot(x_values, predicted_y, '--', label='Fitted polynomial')
            # plt.xlabel('x values')
            # plt.ylabel('y values')
            # plt.title('Polynomial Fit to y values')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            #
            # plt.title(str(pidx) + 'pa')
            # plt.savefig(str(pidx) + 'p.png')
            # plt.clf()

        if False:#pidx == 8:
            # xrange = cp.arange(cp.argmax(cp.abs(data0)).item() - 5000,cp.argmax(cp.abs(data0)).item() + 5000)
            # plt.plot(xrange.get(), cp.abs(data0[xrange]).get())
            plt.plot(cp.abs(data0).get())
            plt.title(f"up {pidx}")
            plt.show() # TODO

    for pidx in range(0, 2 + Config.detect_range_pkts):
        sig1 = pktdata_in[Config.nsamp * (pidx + Config.sfdpos): Config.nsamp * (pidx + Config.sfdpos + 1)]
        sig2 =  sig1 * upchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        data = cp.abs(data0) + cp.abs(cp.roll(data0, -ld))
        Config.fft_downs[pidx] = data
        # data2 = data.copy()
        # data2[cp.argmax(cp.abs(data)) - 2000 :cp.argmax(cp.abs(data)) + 2000 ]= 0
        # print(cp.max(cp.abs(data2)), cp.argmax(cp.abs(data2)))
        amax = cp.argmax(cp.abs(data))
        fdowns.append(cp.argmax(cp.abs(data)).get() )
        fret.append(cp.angle(data[amax]).get())
        if False:#pidx == 1:
            # xrange = cp.arange(cp.argmax(cp.abs(data0)).item() - 5000,cp.argmax(cp.abs(data0)).item() + 5000)
            # plt.plot(xrange.get(), cp.abs(data0[xrange]).get())
            plt.plot(cp.abs(data0).get())
            plt.title(f"down {pidx}")
            plt.show() # TODO
        # plt.plot(cp.abs(data0[::100]).get())
        # plt.title(f"down {pidx}")
        # plt.show() # TODO
    if retpflag:
        draw_fit(0, pktdata_in, 0, 0)
        return fret


    detect_vals = np.zeros((Config.detect_range_pkts, 3))
    for detect_pkt in range(Config.detect_range_pkts):
        skip_preambles = 8
        y_values = fups[skip_preambles + detect_pkt: Config.preamble_len + detect_pkt]#[cp.argmax(cp.abs(Config.fft_ups[pidx])).get() - (Config.fft_n//2-ld) for pidx in range(midx, Config.preamble_len + midx)][skip_preambles:] # length: 0.5fft_n, delta_t>0
        x_values = np.arange(len(y_values)) + skip_preambles  # x values from 1 to n
        degree = 1
        coefficients = np.polyfit(x_values, y_values, degree)
        plt.scatter(x_values, y_values)
        plt.plot(x_values, np.poly1d(coefficients)(x_values))
        plt.show()
        # print('c',coefficients)
        polynomial = np.poly1d(coefficients)
        fft_val_up = (polynomial(Config.sfdpos + 1) - (Config.fft_n//2)) / ld # rate, [-0.5, )  #!!! because previous +0.5
        # print(polynomial(np.array(range(Config.sfdpos + 2))))
        # print(x_values, y_values)
        fft_val_down = (fdowns[1 + detect_pkt]-(Config.fft_n//2)) / ld # cp.argmax(cp.abs(Config.fft_downs[1])).get() - (Config.fft_n//2-ld) / Config.fft_n  # rate, [0, 1)
        print(fdowns[1 + detect_pkt], fft_val_down)
        dvals = 0
        for pidx in range(skip_preambles, Config.preamble_len):
            dvals += cp.abs(Config.fft_ups[pidx][round(polynomial(pidx))]).get()
            print(pidx, cp.abs(Config.fft_ups[pidx][round(polynomial(pidx))]).get(), cp.max(cp.abs(Config.fft_ups[pidx])), ld, cp.argmax(cp.abs(Config.fft_ups[pidx])), round(polynomial(pidx)))
        dvals += cp.abs(Config.fft_downs[1 + detect_pkt][fdowns[1 + detect_pkt]]).get()
        print(cp.abs(Config.fft_downs[1 + detect_pkt][fdowns[1 + detect_pkt]]).get())
        # pidx = 1
        # print(pidx, cp.abs(Config.fft_downs[pidx][round(polynomial(pidx))]).get(),
        #       cp.max(cp.abs(Config.fft_downs[pidx])), ld, cp.argmax(cp.abs(Config.fft_downs[pidx])), round(polynomial(pidx)))
        # print(cp.abs(Config.fft_downs[1][fdowns[1]]).get(), cp.argmax(cp.abs(Config.fft_downs[1])), fdowns[1])
        # newdown = fdowns[1] - coefficients[0]
        # print(cp.abs(Config.fft_downs[0][newdown]).get(), cp.argmax(cp.abs(Config.fft_downs[0])),  cp.max(cp.abs(Config.fft_downs[0])), newdown)
        # newdown = fdowns[1] + coefficients[0]
        # print(coefficients)
        # print(cp.abs(Config.fft_downs[2][newdown]).get(), cp.argmax(cp.abs(Config.fft_downs[2])), cp.max(cp.abs(Config.fft_downs[2])),  newdown)

        pidx = 0
        # print('fup fdown',fft_val_up, fft_val_down)

        f0 = ((fft_val_up + fft_val_down) / 2) % 1 - 0.5
        # print(f0)
        est_cfo_f = Config.bw * f0
        est_to_s = (f0 - fft_val_up) % 1 * Config.tsig
        draw_fit(0, pktdata_in, est_cfo_f, est_to_s)

        # Compute t
        # t = Config.tsig * ((fft_val_up - fft_val_down) * Config.fs / (2 * Config.bw)) % Config.tsig
        # print('estnew', est_cfo_f, est_to_s)

        # draw_fit(0, pktdata_in, est_cfo_f, est_to_s)
        detect_vals[detect_pkt] = (dvals, est_cfo_f, est_to_s)
        print(fft_val_up, fft_val_down, est_cfo_f, est_to_s, f0)
        sys.exit(0)
    detect_pkt_max = np.argmax(detect_vals[:, 0])
    est_cfo_f, est_to_s = detect_vals[detect_pkt_max, 1], detect_vals[detect_pkt_max, 2]
    est_to_s += detect_pkt_max * Config.nsamp
    return est_cfo_f, est_to_s


def fix_cfo_to(est_cfo_f, est_to_s, pktdata_in):
    est_cfo_slope = est_cfo_f / Config.sig_freq * Config.bw / Config.tsig * Config.fs
    sig_time = len(pktdata_in) / Config.fs
    # print(est_cfo_slope * sig_time)
    # logger.info(f'I03_3: SFO {est_cfo_f=} Hz, {est_cfo_slope=} Hz/s, {sig_time=} s')
    estt = cp.linspace(0, sig_time, len(pktdata_in) + 1)[:-1]
    est_cfo_symbol = mychirp(estt, f0=-est_cfo_f, f1=-est_cfo_f + est_cfo_slope * sig_time, t1=sig_time)
    pktdata_fix = pktdata_in * est_cfo_symbol
    return cp.roll(pktdata_fix, -round(est_to_s))


# read packets from file
if __name__ == "__main__":
    for file_path in Config.file_paths:
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
        logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')

        power_eval_len = 5000
        nmaxs = []
        for idx, rawdata in enumerate(read_large_file(file_path)):
            nmaxs.append(cp.max(cp.abs(rawdata)))
            if idx == power_eval_len - 1: break
        nmaxs = cp.array(nmaxs)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(tocpu(nmaxs.reshape(-1, 1)))
        if Config.thresh:
            thresh = Config.thresh
        else:
            thresh = cp.mean(kmeans.cluster_centers_)
        if False:
            counts, bins = cp.histogram(nmaxs, bins=100)
            # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
            logger.debug(f"cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
            threshpos = np.searchsorted(tocpu(bins), thresh).item()
            logger.debug(f"lower: {cp_str(counts[:threshpos])}")
            logger.debug(f"higher: {cp_str(counts[threshpos:])}")
            fig = px.line(nmaxs.get())
            fig.add_hline(y=thresh)
            fig.update_layout(
                title=f"{file_path} pow {len(nmaxs)}",
                legend=dict(x=0.1, y=1.1))
            fig.show()

        ps = []
        ps2 = []
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):
            if pkt_idx < 1: continue
            data1, data2 = pkt_data
            # rolx= 1761.6459319296052 + 100
            # data1 = cp.roll(data1, -round(rolx))
            # data1 = data1[Config.nsamp*2:] #!!!
            # if cp.max(cp.abs(data1)) > 0.072: continue
            logger.info(f"Prework {pkt_idx=} {len(data1)=}")
            # fine_work_new(pkt_idx, data1 / cp.mean(cp.abs(data1)))
            # est_cfo_f, est_to_s = coarse_work_fast(
            # draw_fit(0, data1, est_cfo_f, est_to_s)
            d1 = data1 #/ cp.mean(cp.abs(data1))#fix_cfo_to(est_cfo_f, est_to_s, data1)
            est_cfo_f = 0
            est_to_s = 0
            # print('f', objective_core(est_cfo_f, est_to_s, data1))
            for i in range(5):
                f, t = coarse_work_fast(d1)
                tchoice = [-1, 0, 1, 2]
                ychoice = [objective_core(est_cfo_f, est_to_s + Config.nsamp * x, data1) for x in tchoice]
                tbest = tchoice[np.argmax(np.array(ychoice))]
                t += tbest * Config.nsamp

                # if i>0: t = (t + Config.nsamp // 2) % Config.nsamp - Config.nsamp // 2 # [0, 1) to (-0.5, 0.5) TODO
                est_cfo_f += f
                est_to_s += t
                # d1 = fix_cfo_to(est_cfo_f, est_to_s, data1)
                d1 = cp.roll(data1, -round(est_to_s))
                # draw_fit(0, data1, est_cfo_f, est_to_s)
                print('f',f, t, est_cfo_f, est_to_s,  objective_core(est_cfo_f, est_to_s, data1))
            sys.exit(0)
            ps.extend(coarse_work_fast(d1, True))
            d2 = fix_cfo_to(est_cfo_f, est_to_s, data2)
            ps2.extend(coarse_work_fast(d2, True))
            plt.axvline(len(ps))
        plt.plot(np.angle(np.exp(1j * (np.array(ps) - np.array(ps2)))))
        # plt.plot(ps2)
        plt.show()
            # p, t, c = fine_work_new(pkt_idx, data1 / cp.mean(cp.abs(data1)))
            # pktdata_lst.append(p)
            # tstart_lst.append(t)
            # cfo_freq_est.append(c)
            # with open(f"dataout{parse_opts.searchphase_step}.pkl","wb") as f:
            #     pickle.dump((pktdata_lst, tstart_lst, cfo_freq_est),f)
