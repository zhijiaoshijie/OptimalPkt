import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.ma.extras import polyfit
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
    beta = Config.bw / ((2 ** Config.sf) / Config.bw) * np.pi
    estf = -40454.52914447023#-43462.671492551664+2786-2871.857651918567+3332-238#est_cfo_f #- 12000
    est_to_s -= 0.5
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    x_data = (np.arange(len(pktdata_in))) / Config.fs #* (1 + estf / Config.sig_freq)
    pktdata_shift = add_freq(pktdata_in, -estf)
    y_data_all = np.unwrap(np.angle(tocpu(pktdata_shift.astype(np.complex128))))
    x_data_all = np.arange(0, len(pktdata_in), 1000)
    est_freq2 = []
    est_pow = []


    # accurately compute coeff2d
    def obj(xdata, ydata, coeff2d):
        return np.abs(ydata.dot(np.exp(-1j * np.polyval(coeff2d, xdata))))
    coeflist = []
    for pidx in range(100, 150):
        start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        xv = np.arange(start_pos + 1000, start_pos + Config.nsamp - 1000)
        y_data = np.unwrap(np.angle(tocpu(pktdata_in[xv])))

        coefficients_2d = np.polyfit(x_data[xv], y_data, 2)
        coefficients_2d[2] = 0
        val = obj(x_data[xv], y_data, coefficients_2d)
        import copy
        rangeval = 0.005
        for i in range(20):
            for j in range(2):
                xvals = np.linspace(coefficients_2d[j]*(1-rangeval), coefficients_2d[j]*(1+rangeval), 101)
                yvals = []
                for x in xvals:
                    coef2 = copy.deepcopy(coefficients_2d)
                    coef2[j] = x
                    yvals.append(obj(x_data[xv], y_data, coef2))
                oldv = coefficients_2d[j]
                coefficients_2d[j] = xvals[np.argmax(yvals)]
                valnew = obj(x_data[xv], y_data, coefficients_2d)
                if valnew < val*(1-1e-7):
                    fig = go.Figure(layout_title_text=f"{pidx=} {i=} {j=} {val=} {valnew=}")
                    fig.add_trace(go.Scatter(x=xvals, y=yvals))
                    fig.add_vline(x=oldv)
                    fig.show()
                assert valnew >= val*(1-1e-7), f"{val=} {valnew=} {i=} {j=} {coefficients_2d=} {val-valnew=}"
                if abs(valnew - val) < 1e-7: rangeval /= 2
                val = valnew
        coeflist.append(coefficients_2d[0])
        print(pidx, coefficients_2d[0], val, betanew, beta, (beta+betanew)/2)
    fig=px.line(y=coeflist)
    fig.show()
    sys.exit(0)



    diffs = []
    for pidx in pidx_range[:-1]:
        start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        xv = np.arange(start_pos - 1000, start_pos + Config.nsamp * 2 + 1000)
        y_data = tocpu(cp.unwrap(cp.angle(pktdata_in[xv])))
        # y_data = np.unwrap(np.angle(tocpu(pktdata_in[xv])))
        y_data_1d = y_data - np.polyval((betanew, 0, 0), x_data[xv])
        xv1 = np.arange(1000, Config.nsamp - 1000) + 1000
        coefficients_1da = np.polyfit(x_data[xv][xv1], y_data_1d[xv1], 1)
        xv2 = np.arange(Config.nsamp + 1000, Config.nsamp * 2 - 1000) + 1000
        coefficients_1db = np.polyfit(x_data[xv][xv2], y_data_1d[xv2], 1)

        coeffs_diff = np.polysub(coefficients_1da, coefficients_1db)
        intersection_x_vals = np.roots(coeffs_diff)
        if len(intersection_x_vals) == 2:
            if abs(intersection_x_vals[0]) < abs(intersection_x_vals[1]):
                diffs.append(intersection_x_vals[0])
            else:
                diffs.append(intersection_x_vals[1])
        else:
            diffs.append(intersection_x_vals[0])
    fig = go.Figure(layout_title_text="intersect points")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs))
    coefficients_1d2 = np.polyfit(pidx_range[1:], diffs, 1)
    print(coefficients_1d2)
    print(
        f"estimated time:{coefficients_1d2[0]} cfo ppm from time: {1 - coefficients_1d2[0] / Config.nsampf * Config.fs} cfo: {(1 - coefficients_1d2[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=np.polyval(coefficients_1d2, pidx_range[1:])))
    fig.show()
    fig = go.Figure(layout_title_text="intersect points diff")
    fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs - np.polyval(coefficients_1d2, pidx_range[1:])))
    fig.show()

    c2d = []
    for pidx in pidx_range:
        start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        xv = np.arange(start_pos + 1000, start_pos + Config.nsamp - 1000)
        y_data = y_data_all[xv]
        coefficients_2d = np.polyfit(x_data[xv], y_data, 2)
        c2d.append(coefficients_2d[0])

    fig=px.line(y=c2d)
    fig.add_hline(y=betanew)
    fig.add_hline(y=beta)
    fig.show()


    if True:
        if False:
            fig = go.Figure(layout_title_text="y data all")
            fig.add_trace(go.Scatter(x=x_data_all, y=y_data_all[x_data_all]))
            fig.show()
        pidxs = []
        for pidx in pidx_range:
            # start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
            start_pos_all_new = np.polyval(coefficients_1d2, pidx - 1)*Config.fs + est_to_s
            start_pos = round(start_pos_all_new)
            xv = np.arange(start_pos + 1000, start_pos + Config.nsamp - 1000)
            y_data = y_data_all[xv]
            # y_data = np.unwrap(np.angle(tocpu(pktdata_in[xv])))
            coefficients_2d = np.polyfit(x_data[xv], y_data, 2)
            y_data_1d = y_data - np.polyval((betanew, 0, 0), x_data[xv])
            coefficients_1dx = np.polyfit(x_data[xv], y_data_1d, 1)
            dvx.append((betanew, *coefficients_1dx))


            symb_in = pktdata_in[xv]
            t = x_data[xv] - start_pos_all_new/Config.fs
            phase = 2 * cp.pi * (0 * t - 0.5 * betanew/np.pi * t * t)
            newchirp = cp.exp(1j * togpu(phase))
            symb2 = symb_in * newchirp

            coefficients_1d = np.polyfit(x_data[xv], np.unwrap(np.angle(tocpu(symb2))), 1)

            addpow = np.abs(tocpu(symb2).dot(np.exp(-1j * np.polyval((coefficients_1d[0], 0), x_data[xv]))))/len(symb_in)
            freq = coefficients_1d[0]/2/np.pi
            if pidx==199: print(pidx, coefficients_2d, betanew, coefficients_1dx, coefficients_1d, addpow, freq)
            if addpow > 0.5:
                pidxs.append(pidx)
                est_freq2.append(freq)
                est_pow.append(addpow)
            if pidx%50==0 or pidx<3 or pidx>237:#addpow < 0.5:
                ydata = np.unwrap(np.angle(tocpu(symb_in)))
                coefficients_2d = np.polyfit(x_data[xv], ydata, 2)
                fig=go.Figure()
                xv2 = np.arange(start_pos - 1000, start_pos + Config.nsamp + 1000)
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.unwrap(np.angle(tocpu(pktdata_in[xv2]))), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.polyval(coefficients_2d, x_data[xv2]), mode='lines+markers'))
                fig.add_vline(x=start_pos_all_new/Config.fs)
                fig.add_vline(x=np.polyval(coefficients_1d2, pidx)  + est_to_s/Config.fs)
                fig.update_layout(title=f"{pidx=} diffline")
                fig.show()





        pidx_range2 = np.arange(Config.preamble_len+2, Config.preamble_len + 4)
        for pidx in pidx_range2:
            start_pos_all_new = nsamp_small * pidx * (1 - estf / Config.sig_freq) + est_to_s
            start_pos = round(start_pos_all_new)
            xv = np.arange(start_pos + 1000, start_pos + Config.nsamp - 1000)
            y_data = y_data_all[xv]
            # y_data = np.unwrap(np.angle(tocpu(pktdata_in[xv])))
            coefficients_2d = np.polyfit(x_data[xv], y_data, 2)
            print(pidx, coefficients_2d, betanew, beta)
            y_data_1d = y_data - np.polyval((-betanew, 0, 0), x_data[xv])
            coefficients_1d = np.polyfit(x_data[xv], y_data_1d, 1)
            # print(coefficients_2d, betanew, coefficients_1d )
            print(betanew, coefficients_1d)

            symb_in = pktdata_in[xv]
            t = x_data[xv] - start_pos_all_new/Config.fs
            phase = 2 * cp.pi * (0 * t + 0.5 * betanew/np.pi * t * t)
            newchirp = cp.exp(1j * togpu(phase))
            symb2 = symb_in * newchirp

            coefficients_1d = np.polyfit(x_data[xv], np.unwrap(np.angle(tocpu(symb2))), 1)
            print(coefficients_1d)

            addpow = np.abs(tocpu(symb2).dot(np.exp(-1j * np.polyval((coefficients_1d[0], 0), x_data[xv]))))/len(symb_in)
            freq = coefficients_1d[0]/2/np.pi
            print(addpow, freq, 2 * betanew * freq/Config.fs/Config.fs, start_pos_all_new/Config.fs)
            pidxs.append(pidx)
            est_freq2.append(freq - Config.bw*(1 - 2 * estf / Config.sig_freq))
            if True:
                ydata = np.unwrap(np.angle(tocpu(symb_in)))
                coefficients_2d = np.polyfit(x_data[xv], ydata, 2)
                # coefficients_1d = np.polyfit(x_data[xv], ydata, 1)
                # print(pidx, coefficients_2d, coefficients_1d, betanew)
                # freqreal = np.polyval((betanew / 2 / np.pi * Config.fs, coefficients_1d[0]/ 2 / np.pi*Config.fs - start_pos * betanew / 2 / np.pi* Config.fs), x_data[xv])
                # fig = px.line(x=freqreal, y=ydata - np.polyval(coefficients_1d, x_data[xv]))
                fig=go.Figure()
                xv2 = np.arange(start_pos - 1000, start_pos + Config.nsamp + 1000)
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.unwrap(np.angle(tocpu(pktdata_in[xv2]))), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.polyval(coefficients_2d, x_data[xv2]), mode='lines+markers'))
                fig.add_vline(x=start_pos_all_new/Config.fs)
                fig.add_vline(x=(start_pos_all_new + nsamp_small * (1 - estf / Config.sig_freq)) /Config.fs)
                fig.update_layout(title=f"{pidx=} diffline")
                fig.show()

        dvx = np.array(dvx)

        co_freq = np.polyfit(pidxs[100:-2], est_freq2[100:-2], 1)
        fig = go.Figure(layout_title_text="estfreq")
        fig.add_trace(go.Scatter(x=pidxs,y=est_freq2))
        fig.add_trace(go.Scatter(x=pidxs,y=np.polyval(co_freq, pidxs)))
        fig.show()
        f0 = co_freq[1]+(Config.bw*(1+estf/Config.sig_freq)/2)
        fdiff = co_freq[0]/Config.bw*Config.sig_freq
        print(fdiff, (f0 + estf + fdiff)/beta/np.pi/2*Config.fs)
        fig = go.Figure(layout_title_text="addpow")
        fig.add_trace(go.Scatter(x=pidxs,y=est_pow))
        fig.show()
        print(est_freq2[-2] - np.polyval(co_freq, pidxs[-2]))
        print(est_freq2[-1] - np.polyval(co_freq, pidxs[-1]))


    # find intersections of all symbols





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



