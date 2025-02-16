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
def objective_decode(est_cfo_f, est_to_s, pktdata_in):
    vals = np.zeros(Config.sfdpos + 2, dtype=np.complex64)
    yvals2 = np.zeros(Config.sfdpos + 2, dtype=int)
    vallen = Config.preamble_len - Config.skip_preambles + 2

    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    codes = []
    angdiffs = []
    for pidx in range(Config.sfdpos + 2, Config.total_len):
        # codes1 = []
        # codes2 = []
        codesv = []
        angdv = []
        start_pos_all_new = nsamp_small * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        cfoppm1 = (1 + est_cfo_f / Config.sig_freq)  # TODO!!!
        tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1] + (start_pos - start_pos_all_new) / Config.fs
        for code in range(Config.n_classes):
            nsamples = round(Config.nsamp / Config.n_classes * (Config.n_classes - code))
            refchirp = mychirp(tstandard, f0=Config.bw * (-0.5 + code / Config.n_classes) * cfoppm1 + est_cfo_f, f1=Config.bw * (0.5 + code / Config.n_classes) * cfoppm1 + est_cfo_f,
                                t1=2 ** Config.sf / Config.bw * cfoppm1)[:nsamples]
            sig1 = pktdata_in[start_pos: nsamples + start_pos]
            if len(sig1) > 0: sig2 = cp.abs(sig1.dot(cp.conj(refchirp))) #/ cp.sum(cp.abs(sig1))
            else: sig2 = cp.array(0)

            refchirp = mychirp(tstandard, f0=Config.bw * (-1.5 + code / Config.n_classes) * cfoppm1 + est_cfo_f, f1=Config.bw * (-0.5 + code / Config.n_classes) * cfoppm1 + est_cfo_f,
                               t1=2 ** Config.sf / Config.bw * cfoppm1)[nsamples:]
            sig1 = pktdata_in[start_pos + nsamples: start_pos + Config.nsamp]
            if len(sig1) > 0: sig3 = sig1.dot(cp.conj(refchirp)) #/ cp.sum(cp.abs(sig1))
            else: sig3 = cp.array(0)
            codesv.append(cp.abs(sig2).item() ** 2 + cp.abs(sig3).item() ** 2)
            angdv.append(cp.angle(sig3).item() - cp.angle(sig2).item())
            # codes1.append(sig2.item())
            # codes2.append(sig3.item())
        # fig = go.Figure(layout_title_text=f"decode {pidx=}")
        # fig.add_trace(go.Scatter(y=codes1, mode="lines"))
        # fig.add_trace(go.Scatter(y=codes2, mode="lines"))
        # fig.show()
        # plt.plot(np.unwrap(np.angle(pktdata_in[start_pos - Config.nsamp: start_pos + Config.nsamp].get())))
        # plt.show()
        coderet = np.argmax(np.array(codesv))
        codes.append(coderet)
        angdiffs.append(angdv[coderet])
        # print(pidx, coderet)
    # fig = go.Figure(layout_title_text=f"decode angles")
    # fig.add_trace(go.Scatter(x=codes, y=angdiffs, mode="markers"))
    # fig.show()
    return codes, angdiffs



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



