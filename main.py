import argparse
import logging
import os
import random
import sys
import time
import pickle
import cmath
import math
import matplotlib.pyplot as plt, mpld3
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
parser.add_argument('--searchphase_step', type=int, default=10000)
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
    bw = 406250
    fs = 1e6
    sig_freq = 2.4e9
    preamble_len = 16  # TODO

    thresh = None# 0.03
    scalef = 1000
    file_paths = ['/data/djl/temp/OptimalPkt/data1_test_0']

    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = round(n_classes * fs / bw)
    # f_lower, f_upper = -50000, -30000
    f_lower, f_upper = -41000, -38000
    t_lower, t_upper = 0, nsamp
    fguess = (f_lower + f_upper) / 2
    tguess = nsamp / 2
    code_len = 2

    # fguess = -39000
    # tguess = 4320

    gen_refchirp_deadzone = 0
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)


if use_gpu:
    cp.cuda.Device(0).use()
Config = Config()


def gen_upchirp(t0, td, f0, beta):
    # start from ceil(t0in), end
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t) / Config.fs
    sig = cp.exp(1j * phase)
    return sig


def objective(params, pktdata2a):
    cfofreq, time_error = params
    cfofreq *= Config.scalef
    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=Config.gen_refchirp_deadzone)
    res = cp.zeros(len(detect_symb), dtype=cp.complex64)
    ddx = 0 # TODO
    for sidx, ssymb in enumerate(detect_symb):
        ress = cp.conj(ssymb).dot(pktdata2a_roll[ddx : ddx + len(ssymb)])
        ddx += len(ssymb)
        res[sidx] = ress / len(ssymb)
    # print(cp.mean(cp.abs(pktdata2a[:ddx])))  # Negative because we use a minimizer
    # TODO qian zhui he plot
    # TODO remove **2 because res[sidx] is sum not sumofsquare
    # TODO phase consistency?
    return - tocpu(cp.abs(cp.sum(res)) / len(res))  # Negative because we use a minimizer

def fine_work_new(pktidx, pktdata2a):
    pktdata2a = togpu(pktdata2a)

    if parse_opts.pltphase:
        phase = cp.angle(pktdata2a)
        unwrapped_phase = cp.unwrap(phase)
        fig = px.line(y=tocpu(unwrapped_phase[:30 * Config.nsamp]), title=f"input data 15 symbol {pktidx=}")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} input_data.html"))

    # Perform optimization


    if parse_opts.searchphase:
        # Config.f_lower, Config.f_upper = Config.fguess - 3000, Config.fguess + 3000
        bestx = [Config.fguess/Config.scalef, Config.tguess]
        bestobj = objective(bestx, pktdata2a)
        logger.debug(
            f"trystart cfo_freq_est = {Config.fguess:.3f}, time_error = {Config.tguess:.3f} {bestobj=} {Config.f_lower=} {Config.f_upper=} {Config.t_lower=} {Config.t_upper=}")
        draw_fit(pktidx, pktdata2a, Config.fguess, Config.tguess)
        for tryidx in tqdm(range(parse_opts.searchphase_step)):
            start_t = random.uniform(Config.t_lower, Config.t_upper)
            start_f = random.uniform(Config.f_lower, Config.f_upper)
            # noinspection PyTypeChecker
            result = opt.minimize(objective, [start_f/Config.scalef, start_t], args=(pktdata2a,),
                                  bounds=[(Config.f_lower/Config.scalef, Config.f_upper/Config.scalef), (Config.t_lower, Config.t_upper)],
                                  method='L-BFGS-B',
                                  options={'gtol': 1e-12, 'disp': False}
                                  )

            if result.fun < bestobj:
                logger.debug(f"{tryidx=: 6d} cfo_freq_est = {result.x[0]*Config.scalef:.3f}, time_error = {result.x[1]:.3f} {result.fun=:.5f}")
                bestx = result.x
                # Config.fguess, Config.tguess = result.x
                bestobj = result.fun
                if tryidx > 100: draw_fit(pktidx, pktdata2a, result.x[0]*Config.scalef, result.x[1])
            if tryidx == 100: draw_fit(pktidx, pktdata2a, bestx[0] * Config.scalef, bestx[1])
        cfo_freq_est, time_error = bestx
        cfo_freq_est *= Config.scalef
        logger.info(f"Optimized parameters:\n{cfo_freq_est=}\n{time_error=}")
    else:
        cfo_freq_est = Config.fguess
        time_error = Config.tguess

    if parse_opts.refine:
        cfo_search = 200
        to_search = 10
        search_steps = 100
        for searchidx in range(5):
            xrange = cp.linspace(cfo_freq_est - cfo_search, cfo_freq_est + cfo_search, search_steps)
            yvals = [objective((x/Config.scalef, time_error), pktdata2a) for x in xrange]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tocpu(xrange), y=yvals, mode='markers', name='Input Data'))
            fig.add_vline(x=cfo_freq_est)
            fig.update_layout(title=f"pkt{pktidx} {searchidx=} {cfo_search=} {cfo_freq_est=} {time_error=} objective fit frequency")
            cfo_freq_est_new = xrange[np.argmin(np.array(yvals))]
            cfo_freq_est_delta = abs(cfo_freq_est_new - cfo_freq_est)
            cfo_freq_est = cfo_freq_est_new
            fig.add_vline(x=cfo_freq_est, line_dash="dash", line_color="red")
            if not parse_opts.noplot: fig.show()

            xrange = cp.linspace(time_error - to_search, time_error + to_search, 1000)
            yvals = [objective((cfo_freq_est/Config.scalef, x), pktdata2a) for x in xrange]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tocpu(xrange), y=yvals, mode='markers', name='Input Data'))
            fig.add_vline(x=time_error)
            fig.update_layout(title=f"pkt{pktidx} {searchidx=} {to_search=} objective fit timeerror")
            time_error_new = xrange[np.argmin(np.array(yvals))]
            time_error_delta = abs(time_error_new - time_error)
            time_error = time_error_new
            fig.add_vline(x=time_error, line_dash="dash", line_color="red")
            if not parse_opts.noplot: fig.show()
            if cfo_freq_est_delta > cfo_search / 2: cfo_search *= 3
            else: cfo_search = (cfo_search + cfo_freq_est_delta) / 2
            if time_error_delta > to_search / 2: to_search *= 3
            else: to_search = (to_search + time_error_delta) / 2
            logger.info(f"pkt{pktidx} {searchidx=} {cfo_freq_est=} {time_error=} objective={min(yvals)} fit")
            draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error)
    logger.info(f"FIN pkt{pktidx} {cfo_freq_est=} {time_error=}")



def draw_fit(pktidx, pktdata2a, cfo_freq_est, time_error):
    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    logger.info(f"{cfo_freq_est=:.3f}, time_error = {time_error:.3f}")
    detect_symb_plt = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    tsfd = sum([len(x) for x in detect_symb_plt[:-3]])
    print([len(x) for x in detect_symb_plt])
    print([max(abs(x)) for x in detect_symb_plt])
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
    fig = go.Figure()
    # view_len = 60
    fig.add_trace(go.Scatter(x=tocpu(xval2), y=tocpu(yval1[xval2]), mode='lines', name='input', line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=tocpu(xval), y=tocpu(yval2), mode='lines', name='fit', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(
        x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
        y=[0, Config.f_lower * 2 * np.pi / Config.fs * len(detect_symb_plt)],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
        y=[0, Config.f_upper * 2 * np.pi / Config.fs * len(detect_symb_plt)],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[math.ceil(time_error), math.ceil(time_error) + len(detect_symb_plt)],
        y=[0, tocpu(cfo_freq_est) * 2 * np.pi / Config.fs * len(detect_symb_plt)],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend = False
    ))
    fig.update_layout(title=f'{pktidx} f={cfo_freq_est:.3f} t={time_error:.3f} obj={objective((cfo_freq_est/Config.scalef, time_error), pktdata2a):.5f}', legend=dict(x=0.1, y=1.1))
    if not parse_opts.noplot: fig.show()


def gen_refchirp(cfofreq, tstart, deadzone=0):
    detect_symb = []
    sigt = Config.tsig * (1 + cfofreq / Config.sig_freq)
    beta = Config.bw / Config.tsig #* (1 + cfofreq / Config.sig_freq) / (1 - cfofreq / Config.sig_freq)  # TODO
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt, -Config.bw / 2 + cfofreq, beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tstart + sigt * (tid + 1)) - math.ceil(tstart + sigt * tid), dtype=cp.complex64))
    for tid in range(Config.sfdpos, Config.sfdend):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt if tid != Config.sfdend - 1 else sigt / 4,
                              Config.bw / 2 + cfofreq, - beta)
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


def read_pkt(file_path_in, threshold, min_length=20):
    current_sequence = []
    for rawdata in read_large_file(file_path_in):
        number = cp.max(cp.abs(rawdata))
        if number > threshold:
            current_sequence.append(rawdata)
        else:
            if len(current_sequence) > min_length:
                yield cp.concatenate(current_sequence)
            current_sequence = []


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

        pkt_totcnt = 0
        pktdata_lst = []
        tstart_lst = []
        cfo_freq_est = []
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=30)):
            if pkt_idx < 0: continue
            # if cp.max(cp.abs(pkt_data)) > 0.072: continue
            logger.info(f"Prework {pkt_idx=} {len(pkt_data)=}")
            fine_work_new(pkt_idx, pkt_data / cp.mean(cp.abs(pkt_data)))
            break
            # p, t, c = fine_work_new(pkt_idx, pkt_data / cp.mean(cp.abs(pkt_data)))
            # pktdata_lst.append(p)
            # tstart_lst.append(t)
            # cfo_freq_est.append(c)
            # with open(f"dataout{parse_opts.searchphase_step}.pkl","wb") as f:
            #     pickle.dump((pktdata_lst, tstart_lst, cfo_freq_est),f)
