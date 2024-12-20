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
    for pidx in range(0, Config.preamble_len):
        # fig = FigureResampler(go.Figure(layout_title_text=f"OL fit {pidx=} {est_cfo_f=:.3f} {est_to_s=:.3f}"))
        if True:# for deltaf in np.arange(0, 200, 10):
            deltaf = 0
            estf = est_cfo_f + deltaf
            start_pos_all_new = nsamp_small * (pidx ) * (1 - estf / Config.sig_freq) + est_to_s
            start_pos = round(start_pos_all_new)

            tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
            refchirp = mychirp(tstandard, f0=Config.bw * -0.5 , f1=Config.bw * 0.5, t1=2 ** Config.sf / Config.bw)
            dt = (start_pos - start_pos_all_new) / Config.fs
            beta = Config.bw / ((2 ** Config.sf) / Config.bw)
            df = -estf #+ dt * beta)
            sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]

            x = (cp.arange(len(pktdata_in)) - est_to_s) * (1 + estf / Config.sig_freq)
            yi = cp.zeros_like(x, dtype=np.complex64)
            bwnew = Config.bw * (1 + estf / Config.sig_freq)
            betanew = beta * (1 + 2 * estf / Config.sig_freq)
            for i in range(Config.preamble_len):
                mask = (i * Config.nsampf < x) & (x <= (i + 1) * Config.nsampf)
                yi[mask] = cp.exp(2j * cp.pi * (betanew / 2 * (x[mask] - i * Config.nsampf) ** 2 / Config.fs**2 + (- bwnew / 2 + estf) * (x[mask] - i * Config.nsampf)/Config.fs))
            xv = cp.arange(start_pos , start_pos + Config.preamble_len * Config.nsamp)
            diffdata = np.diff(tocpu(cp.unwrap(cp.angle(pktdata_in[xv]*cp.conj(yi[xv])))), prepend=0)
            d2data = np.ones_like(diffdata)
            for i in range(1, Config.preamble_len):
                d2data[abs(x[xv] - i * Config.nsampf) < Config.gen_refchirp_deadzone] = 0
            p2 = pktdata_in[xv] * togpu(d2data)
            # val = cp.abs(cp.conj(togpu(yi[xv])).dot(p2))
            # val2 = cp.concatenate( [cp.abs(cp.cumsum(cp.conj(togpu(yi[xv][round(i * Config.nsampf): round((i+1)*Config.nsampf)])) * p2[round(i * Config.nsampf): round((i+1)*Config.nsampf)])) for i in range(0, Config.preamble_len)])
            val3 = cp.array( [cp.angle(cp.conj(togpu(yi[xv][round(i * Config.nsampf): round((i+1)*Config.nsampf)])) .dot( p2[round(i * Config.nsampf): round((i+1)*Config.nsampf)])) for i in range(0, Config.preamble_len)])

            # fig.add_trace(go.Scatter(x=x[xv], y=tocpu(cp.unwrap(cp.angle(yi[xv])))))
            # fig.add_trace(go.Scatter(x=x[xv], y=tocpu(cp.unwrap(cp.angle(pktdata_in[xv])))))
            # fig.add_trace(go.Scatter(x=x[xv], y=tocpu(val2), name=f"{estf=:.3f}"))
        # fig.show()
        x_data = np.arange(Config.skip_preambles*4, Config.preamble_len-1)
        y_data = np.unwrap(np.diff(tocpu(val3)))[x_data]
        coefficients = np.polyfit(x_data, y_data, 1)
        # print(coefficients)
        # return (est_cfo_f, coefficients[0])
        # if abs(coefficients[0] + 0.025)>0.001:
        if True:
            logger.warning(f"Fitted parameters:  {coefficients} {est_cfo_f=}")
            x_data = np.arange(0, Config.preamble_len)
            fig=px.scatter(y=tocpu(cp.unwrap(cp.diff(val3))))
            fig.add_trace(go.Scatter(x=x_data, y=np.polyval(coefficients, x_data), mode="lines", name="Fitted Curve"))
            fig.show()
            return None
        else: return (est_cfo_f, coefficients[0])
        # coefficients[0] = est_cfo_f/Config.sig_freq * Config.bw/Config.fs/np.pi

        sys.exit(0)

        # logger.warning(f"standard a1={np.pi * beta / Config.fs**2} b1={2*np.pi*(Config.bw*-0.5-df)/Config.fs} c1=0")
        estc1 = (1 + est_cfo_f/Config.sig_freq * 2) * np.pi * beta / Config.fs**2
        estc2 = est_cfo_f/Config.sig_freq * Config.bw/Config.fs/np.pi*pidx + (-Config.bw*0.5+est_cfo_f)*2*np.pi/Config.fs
        print(estc1, estc2)
        x_val = np.arange(Config.nsamp) + dt * Config.fs #- pidx
        y_val = tocpu(cp.unwrap(cp.angle(sig1))) - np.polyval((estc1, estc2, 0), x_val)
        coefficients = np.polyfit(x_val, y_val, 0)
        # logger.warning(f"ours a={coefficients})")
        if False:#pidx % 10 == 0:
            fig = FigureResampler(go.Figure(layout_title_text=f"OL fit {pidx=} {est_cfo_f=:.3f} {est_to_s=:.3f}"))
            # fig.add_trace(go.Scatter(y=y_val))
            # fig.add_trace(go.Scatter(y=np.polyval(coefficients, x_val)))
            fig.add_trace(go.Scatter(y=y_val - np.polyval(coefficients, x_val), mode="markers"))
            fig.update_layout(yaxis=dict(range=[-0.2, 0.2]))
            fig.show()
        dvx.append(coefficients)
    dvx = np.array(dvx)
    fig = FigureResampler(go.Figure(layout_title_text=f"OL coeffs1 {est_cfo_f=:.3f} {est_to_s=:.3f}"))
    fig.add_trace(go.Scatter(y=np.unwrap(dvx[:, 0])))
    # fig.add_hline(y=estc1)
    # fig.add_hline(y=2*np.pi*(Config.bw*-0.5-df)/Config.fs)
    # fig.show()
    # x_data = np.arange(Config.skip_preambles, Config.preamble_len)
    x_data = np.arange(0, Config.preamble_len)
    from scipy.optimize import curve_fit

    # def log_func(x, a, b, c, d):
    #     return a * np.log(b * x + c) + d
    # # Fit the curve
    # initial_guess = [ 8.90091896e-05, 2.28698507e+00 , 4.78538888e-01, -1.53961710e+00] # Initial guess for the parameters [a, b, c, d]
    # params, covariance = curve_fit(log_func, x_data, y_data, p0=initial_guess)
    # print("Fitted parameters:", params)
    # y_fit = log_func(x_data, *params)
    # fig.add_trace(go.Scatter(x=x_data, y=y_fit, mode="lines", name="Fitted Curve"))
    # fig.show()

    x_data = np.arange(Config.skip_preambles * 3, Config.preamble_len)
    y_data = np.unwrap(dvx[:, 0])[x_data]
    coefficients = np.polyfit(x_data, y_data, 2)
    print("Fitted parameters:", coefficients,  est_cfo_f/Config.sig_freq * Config.bw/Config.fs/np.pi)
    x_data = np.arange(0, Config.preamble_len)
    # coefficients[0] = est_cfo_f/Config.sig_freq * Config.bw/Config.fs/np.pi
    fig.add_trace(go.Scatter(x=x_data, y=np.polyval(coefficients, x_data), mode="lines", name="Fitted Curve"))
    fig.show()




    # fig = go.Figure(layout_title_text=f"decode angles")
    # fig.add_trace(go.Scatter(y=amaxdfs, mode="markers"))
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



