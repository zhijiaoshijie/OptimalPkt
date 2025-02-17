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
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * Config.fs * (1 + 2 * est_cfo_f / Config.sig_freq)

    df = -(est_cfo_f + dt * betai)
    cfosymb = cp.exp(2j * cp.pi * df * cp.linspace(0, Config.nsamp / Config.fs, num=Config.nsamp, endpoint=False)).astype(cp.complex64)
    decode_matrix_a = Config.decode_matrix_a * cfosymb
    decode_matrix_b = Config.decode_matrix_b * cfosymb
    return decode_matrix_a, decode_matrix_b
def objective_decode(est_cfo_f, est_to_s, pktdata_in):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    codes = []
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * Config.fs * (1 + 2 * est_cfo_f / Config.sig_freq)

    decode_matrix_a, decode_matrix_b = gen_matrix2(0, est_cfo_f)
    for pidx in range(Config.sfdpos + 2, Config.total_len):
        start_pos_all_new = nsamp_small * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        dt =  (start_pos - start_pos_all_new) / Config.fs
        # dataX = add_freq(pktdata_in[start_pos: start_pos + Config.nsamp], dt * betai + est_cfo_f)
        dataX = add_freq(pktdata_in[start_pos: start_pos + Config.nsamp], dt * betai)

        dataX = np.array(dataX).T
        data1 = np.matmul(decode_matrix_a, dataX)
        data2 = np.matmul(decode_matrix_b, dataX)
        vals = np.abs(data1) ** 2 + np.abs(data2) ** 2
        # fig = go.Figure(layout_title_text=f"decode")
        # fig.add_trace(go.Scatter(y=vals))
        # fig.show()
        # fig = go.Figure(layout_title_text=f"symb")
        # fig.add_trace(go.Scatter(y=tocpu(cp.unwrap(cp.angle(pktdata_in[start_pos: start_pos + Config.nsamp])))))
        # fig.show()
        est = np.argmax(vals).item()

        codes.append(est)
        # print(pidx, coderet)
    # fig = go.Figure(layout_title_text=f"decode angles")
    # fig.add_trace(go.Scatter(x=codes, y=angdiffs, mode="markers"))
    # fig.show()
    return codes

def objective_decode_baseline(est_cfo_f, est_to_s, pktdata_in):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    codes = []
    angdiffs = []
    tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    refchirp = mychirp(tstandard, f0=Config.bw * 0.5 - est_cfo_f, f1=Config.bw * -0.5 - est_cfo_f, t1=2 ** Config.sf / Config.bw)
    for pidx in range(Config.sfdpos + 2, Config.total_len):
        start_pos_all_new = nsamp_small * (pidx + 0.25) + est_to_s
        start_pos = round(start_pos_all_new)

        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * refchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        freq = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs)), y=cp.abs(data0)))
        # fig.show()
        if freq < 0: freq += Config.bw
        codex = freq / Config.bw * 2 ** Config.sf
        code = round(codex)
        code %= 2 ** Config.sf
        codes.append(code)
    return codes


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



