import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import sys

from utils import *

def objective_linear(cfofreq, time_error, pktdata2a):
    if time_error < 0: return -cfofreq, -time_error
    detect_symb_concat = gen_refchirp(cfofreq, time_error, len(pktdata2a))
    tint = math.ceil(time_error)
    # logger.warning(f"ObjLinear {cfofreq=} {time_error=} {tint=} {len(pktdata2a)=}")
    # fig = FigureResampler(go.Figure(layout_title_text=f"OL fit {cfofreq=:.3f} {time_error=:.3f}"))
    # fig.add_trace(go.Scatter(y=tocpu(cp.unwrap(cp.angle(pktdata2a[tint:tint + len(detect_symb_concat)])))))
    # fig.add_trace(go.Scatter(y=tocpu(cp.unwrap(cp.angle(detect_symb_concat)))))
    # fig.show()
    ress2 = -cp.unwrap(cp.angle(pktdata2a)) + cp.unwrap(cp.angle(detect_symb_concat))


    phasediff = ress2#cp.unwrap(cp.angle(cp.array(ress2)))
    est_dfreqs = []
    for fit_symbidx in range(0, Config.preamble_len):
        x_values = np.arange(Config.nsamp * fit_symbidx + tint + 50, Config.nsamp * (fit_symbidx + 1) + tint - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_dfreq = coefficients[0] * Config.fs / 2 / np.pi
        est_dfreqs.append(est_dfreq)
        # print(f"fitted curve {est_dfreq=:.2f} Hz")

    est_ups = []
    for fit_symbidx in range(Config.sfdpos, Config.sfdpos + 2):
        x_values = np.arange(Config.nsamp * fit_symbidx + tint + 50, Config.nsamp * (fit_symbidx + 1) + tint - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_ufreq = coefficients[0] * Config.fs / 2 / np.pi
        est_ups.append(est_ufreq)
        # print(f"fitted curve {est_ufreq=:.2f} Hz")
    ret_ufreq = np.mean(est_ups)
    ret_dfreq = np.mean(est_dfreqs[Config.skip_preambles:])
    if abs(cfofreq - -41774.0) < 3000:
        fig = px.scatter(y=est_dfreqs, title=f"objlinear {cfofreq:.3f} {time_error:.3f}")
        fig.add_hline(y=ret_dfreq)
        fig.show()

    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    ret_freq = (ret_ufreq + ret_dfreq)/2
    ret_tdiff = (ret_ufreq - ret_dfreq)/2 / beta
    logger.warning(f"linear {ret_freq=} {ret_tdiff=}")

    return ret_freq, ret_tdiff


def gen_matrix2(dt, est_cfo_f):
    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs

    df = -(est_cfo_f + dt * beta)
    cfosymb = cp.exp(2j * cp.pi * df * cp.linspace(0, Config.nsamp / Config.fs, num=Config.nsamp, endpoint=False)).astype(cp.complex64)
    decode_matrix_a = Config.decode_matrix_a * cfosymb
    decode_matrix_b = Config.decode_matrix_b * cfosymb
    return decode_matrix_a, decode_matrix_b

# todo 斜率是否不受cfo sfo影响
def objective_decode(est_cfo_f, est_to_s, pktdata_in):
    tstandard = cp.arange(len(pktdata_in))
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    pstart = nsamp_small * (0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
    # pktdata_in *= np.exp(1j * np.pi * Config.cfo_change_rate * (tstandard - pstart) ** 2 / Config.fs)
    codes = []
    angdiffs = []
    amaxdfs = []
    # for pidx in range(Config.sfdpos + 2, round(Config.total_len)):
    dvx = []
    pidx_range = np.arange(Config.preamble_len)
    beta = Config.bw / ((2 ** Config.sf) / Config.bw) * np.pi /Config.fs/Config.fs
    estf = -43462.671492551664+2786#est_cfo_f #- 12000
    # est_to_s += 1004.44
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    x_data = (np.arange(len(pktdata_in))) #/ Config.fs #* (1 + estf / Config.sig_freq)
    pktdata_shift = add_freq(pktdata_in, -estf)
    y_data_all = np.unwrap(np.angle(tocpu(pktdata_shift.astype(np.complex128))))
    x_data_all = np.arange(0, len(pktdata_in), 1000)
    est_freq2 = []
    est_pow = []
    if False:
        fig = go.Figure(layout_title_text="y data all")
        fig.add_trace(go.Scatter(x=x_data_all, y=y_data_all[x_data_all]))
        fig.show()
    for pidx in pidx_range:
        start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        xv = np.arange(start_pos + 1000, start_pos + Config.nsamp - 1000)
        y_data = y_data_all[xv]
        # y_data = np.unwrap(np.angle(tocpu(pktdata_in[xv])))
        coefficients_2d = np.polyfit(x_data[xv], y_data, 2)
        y_data_1d = y_data - np.polyval((betanew, 0, 0), x_data[xv])
        coefficients_1d = np.polyfit(x_data[xv], y_data_1d, 1)
        # print(coefficients_2d, betanew, coefficients_1d )
        dvx.append((betanew, *coefficients_1d))


        symb_in = pktdata_in[xv]
        symb2 = symb_in * mychirp(np.arange(len(symb_in)) , 0, -betanew/np.pi, 1)

        coefficients_1d = np.polyfit(x_data[xv], np.unwrap(np.angle(tocpu(symb2))), 1)

        addpow = np.abs(tocpu(symb2).dot(np.exp(-1j * np.polyval((coefficients_1d[0], 0), x_data[xv]))))/len(symb_in)
        freq = coefficients_1d[0]/2/np.pi
        est_freq2.append(freq)
        est_pow.append(addpow)
        if pidx%100==0:#addpow < 0.5:
            ydata = np.unwrap(np.angle(tocpu(symb2)))
            coefficients_2d = np.polyfit(x_data[xv], ydata, 2)
            coefficients_1d = np.polyfit(x_data[xv], ydata, 1)
            print(pidx, coefficients_2d, coefficients_1d, betanew)
            freqreal = np.polyval((betanew / 2 / np.pi * Config.fs, coefficients_1d[0]/ 2 / np.pi*Config.fs - start_pos * betanew / 2 / np.pi* Config.fs), x_data[xv])
            fig = px.line(x=freqreal, y=ydata - np.polyval(coefficients_1d, x_data[xv]))
            fig.update_layout(title=f"{pidx=} diffline")
            fig.show()


    dvx = np.array(dvx)

    fig = go.Figure(layout_title_text="estfreq")
    fig.add_trace(go.Scatter(y=est_freq2))
    fig.show()
    fig = go.Figure(layout_title_text="addpow")
    fig.add_trace(go.Scatter(y=est_pow))
    fig.show()


    sys.exit(0)



    diffs = []
    for pidx in pidx_range[1:]:
        # coeffs1 = (betanew, *dvx[pidx - 1])
        # coeffs2 = (betanew, *dvx[pidx])
        coeffs1 = dvx[pidx - 1]
        coeffs2 = dvx[pidx]
        coeffs_diff = np.polysub(coeffs1, coeffs2)
        intersection_x_vals = np.roots(coeffs_diff)
        if len(intersection_x_vals) == 2:
            if abs(intersection_x_vals[0]) < abs(intersection_x_vals[1]):
                diffs.append(intersection_x_vals[0])
            else:
                diffs.append(intersection_x_vals[1])
        else:
            diffs.append(intersection_x_vals[0])
        if False:#pidx > pidx_range[1]:# True:#pidx % 100 == 0:
            start_pos_all_new = nsamp_small * (pidx-1) * (1 - estf / Config.sig_freq) + est_to_s
            start_pos = round(start_pos_all_new)
            xv = np.arange(start_pos - 1000, start_pos + Config.nsamp + 1000)
            fig = px.line(x=xv, y=np.unwrap(np.angle(tocpu(pktdata_in)))[xv])
            y_data1 = np.polyval(coeffs1,x_data[xv])
            y_data2 = np.polyval(coeffs2,x_data[xv])
            # fig.add_trace(go.Scatter(x=xv, y=y_data1))
            # fig.add_trace(go.Scatter(x=xv, y=y_data2))
            fig.add_vline(x=diffs[-2]*Config.fs)
            fig.add_vline(x=diffs[-1]*Config.fs)
            fig.update_layout(title=f"{pidx=}")
            fig.show()

    fig = go.Figure(layout_title_text="intersect points")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs))
    coefficients_1d = np.polyfit(pidx_range[1:], diffs, 1)
    print(coefficients_1d)
    print(f"estimated time:{coefficients_1d[0]} cfo ppm from time: {1-coefficients_1d[0]/Config.nsampf*Config.fs} cfo: {(1-coefficients_1d[0]/Config.nsampf*Config.fs)*Config.sig_freq}")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=np.polyval(coefficients_1d,pidx_range[1:])))
    fig.show()
    fig = go.Figure(layout_title_text="intersect points diff")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs-np.polyval(coefficients_1d,pidx_range[1:])))
    fig.show()

    fig=go.Figure(layout_title_text="cfo of each symb")
    fig.add_trace(go.Scatter(y=(np.diff(diffs)/Config.nsampf*Config.fs-1)*Config.sig_freq, mode='markers', marker=dict(size=2)))
    fig.show()

    coefficients_1dfit = np.polyfit(pidx_range, dvx[:, 0], 1)
    fig = go.Figure(layout_title_text="1d")
    # fig.add_trace(go.Scatter(x=pidx_range, y=- dvx / (2 * betanew), mode='lines',))
    # fig.add_trace(go.Scatter(x=pidx_range, y=- np.poly1d(coefficients_1dfit)(pidx_range) / (2 * betanew), mode='lines',))
    fig.add_trace(go.Scatter(x=pidx_range, y=dvx - np.poly1d(coefficients_1dfit)(pidx_range), mode='lines',))
    print(coefficients_1dfit)
    timediff = - coefficients_1dfit[0] / (2 * betanew)
    freqdiff = (timediff - Config.tsig / Config.fs) / Config.nsampf * Config.fs * Config.sig_freq
    print(timediff, Config.tsig / Config.fs, (timediff - Config.tsig / Config.fs) / Config.nsampf * Config.fs, freqdiff)
    time0 = - coefficients_1dfit[1] / (2 * betanew)
    print(time0 * Config.fs - 0.5 * nsamp_small)
    fig.show()
    sys.exit(0)



def gen_refchirp(est_to_s, estf, length):
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    x = (cp.arange(length) - est_to_s) * (1 + estf / Config.sig_freq)
    yi = cp.zeros_like(x, dtype=np.complex64)
    bwnew = Config.bw * (1 + estf / Config.sig_freq)
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    for i in range(Config.preamble_len):
        mask = (i * Config.nsampf < x) & (x <= (i + 1) * Config.nsampf)
        yi[mask] = cp.exp(2j * cp.pi * (
                betanew / 2 * (x[mask] - i * Config.nsampf) ** 2 / Config.fs ** 2 + (- bwnew / 2 + estf) * (
                x[mask] - i * Config.nsampf) / Config.fs))
    for i in range(Config.sfdpos, Config.sfdpos + 3):
        mask = (i * Config.nsampf < x) & (x <= (i + 1) * Config.nsampf) & (x <= (Config.sfdpos + 2.25) * Config.nsampf)
        yi[mask] = cp.exp(2j * cp.pi * (
                -betanew / 2 * (x[mask] - i * Config.nsampf) ** 2 / Config.fs ** 2 + (bwnew / 2 + estf) * (
                x[mask] - i * Config.nsampf) / Config.fs))
    return yi


def objective_core_new(est_cfo_f, est_to_s, pktdata_in):
    deltaf = 0
    estf = est_cfo_f + deltaf
    start_pos_all_new = est_to_s
    start_pos = round(start_pos_all_new)

    yi = gen_refchirp(est_to_s, estf, len(pktdata_in))

    retvals = 0
    for i in range(Config.preamble_len):
        xv = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i+1) * Config.nsamp))
        retvals += cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)
        # if i>=Config.preamble_len - 2:
        #     logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)}")
    for i in range(Config.sfdpos, Config.sfdpos + 3):
        xv = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i+1) * Config.nsamp))
        if i == Config.sfdpos + 2:
            xv = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i + 0.25) * Config.nsamp))
        retvals += cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)
        # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)}")
    # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {retvals=}")

    return retvals



