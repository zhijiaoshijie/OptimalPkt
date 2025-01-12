import matplotlib.pyplot as plt
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.ma.extras import polyfit
from plotly_resampler import FigureResampler
import sys
from find_intersections import find_intersections

from utils import *
from newwork import *




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

    betai = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    ret_freq = (ret_ufreq + ret_dfreq)/2
    ret_tdiff = (ret_ufreq - ret_dfreq)/2 / betai
    logger.warning(f"linear {ret_freq=} {ret_tdiff=}")

    return ret_freq, ret_tdiff


def gen_matrix2(dt, est_cfo_f):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs

    df = -(est_cfo_f + dt * betai)
    cfosymb = cp.exp(2j * cp.pi * df * cp.linspace(0, Config.nsamp / Config.fs, num=Config.nsamp, endpoint=False)).astype(cp.complex64)
    decode_matrix_a = Config.decode_matrix_a * cfosymb
    decode_matrix_b = Config.decode_matrix_b * cfosymb
    return decode_matrix_a, decode_matrix_b

# todo 斜率是否不受cfo sfo影响
def objective_decode(estf, est_to_s, pktdata_in):
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs
    dvx = []
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * np.pi
    betanew = betai * (1 + 2 * estf / Config.sig_freq)
    betat = betai * (1 + estf / Config.sig_freq)
    x_data = (np.arange(len(pktdata_in))) / Config.fs #* (1 + estf / Config.sig_freq)
    pktdata_shift = add_freq(pktdata_in, -estf)
    y_data_all = np.unwrap(np.angle(tocpu(pktdata_shift.astype(np.complex128))))
    est_freq2 = []
    est_pow = []
    # plt.plot(np.unwrap(np.angle(tocpu(pktdata_in[20000:40000]))))
    # plt.show()
    # sys.exit(0)


    # accurately compute coeff2d


    if False:
        with open("coefout3.pkl", "rb") as fl: coeflist = pickle.load(fl)
        assert isinstance(coeflist, np.ndarray)


        if False:
            pidx = 150
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart)-1000,round(nstart) + Config.nsamp+1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tsymbr, y=np.abs(pktdata_in[nsymbr])))
            fig.show()
            dif2 = []
            newsig = np.zeros_like(pktdata_in)
            newsig[nsymbr] = np.exp(1j * np.polyval(coeflist[pidx], tsymbr))
            print(coeflist[pidx])
            for xi in nsymbr[:-2]:
                # dif2.append(wrap(np.angle(newsig[xi]) + np.angle(newsig[xi + 2]) - 2 * np.angle(newsig[xi + 1])))
                # dif2.append(wrap(np.angle(pktdata_in[xi]) + np.angle(pktdata_in[xi + 2]) - 2 * np.angle(pktdata_in[xi + 1])))
                dif2.append(wrap(np.angle(pktdata_in[xi + 1]) - np.angle(pktdata_in[xi])))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data[nsymbr[:-2]], y=dif2, mode='markers'))
            fig.show()
            sys.exit(0)

        fig = go.Figure(layout_title_text="coef0")
        fig.add_trace(go.Scatter(y=coeflist[:, 0]))
        fig.add_hline(y=betai)
        fig.add_hline(y=betai * (1 + 2 * estf / Config.sig_freq))
        fig.add_hline(y=betai * (1 + estf / Config.sig_freq))
        fig.show()




        dd = []
        for pidx in range(240):
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            coeflist[pidx, 2] += np.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coeflist[pidx], tsymbr))))
            dd.append(np.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coeflist[pidx], tsymbr)))))
        # lst = lst[abs(lst - 126.58e6) < 0.018e6]
        fig=px.line(dd, title="coef 2 fix phase, phase diff")
        fig.show()



        # time diff
        diffs = []
        dd2 = []
        coeflist2 = coeflist.copy()
        dd31 = []
        for pidx in range(239):
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            xva = np.arange(round(nstart) - 1000, round(nsymblen * (pidx+2) * (1 - estf / Config.sig_freq) + est_to_s) + 1000)
            yplt = np.zeros_like(pktdata_in, dtype=np.float64)
            yplt[xva] = np.unwrap(np.angle(tocpu(pktdata_in[xva])))
            val = (yplt[round(nstart)+1000] - np.angle(pktdata_in[round(nstart)+1000])) / 2 / np.pi
            # val = (yplt[round(nstart)+1000] - np.polyval(coeflist[pidx], x_data[round(nstart) + 1000])) / 2 / np.pi
            logger.warning(f"{val=}, {np.polyval(coeflist[pidx], x_data[round(nstart) + 1000])}")
            assert abs(val-round(val))<0.1
            dd2.append(val - round(val))
            coeflist[pidx, 2] += round(val) * 2 * np.pi

            start_pos_all_new2 = nsymblen * (pidx + 1) * (1 - estf / Config.sig_freq) + est_to_s
            start_pos2 = round(start_pos_all_new2)
            val2 = (yplt[start_pos2+1000] - np.angle(pktdata_in[start_pos2+1000])) / 2 / np.pi
            # val = (yplt[start_pos2+1000] - np.polyval(coeflist[pidx+1], x_data[start_pos2 + 1000])) / 2 / np.pi
            dd2.append(val2 - round(val2))
            logger.warning(f"{val2=}")
            assert abs(val2-round(val2))<0.1
            coeflist[pidx + 1, 2] += round(val2) * 2 * np.pi
            dd31.append(round(val2) - round(val))

            coeffs_diff = np.polysub(coeflist[pidx], coeflist[pidx + 1])
            intersection_x_vals = np.roots(coeffs_diff)
            if len(intersection_x_vals) == 2:
                if abs(intersection_x_vals[0]) < abs(intersection_x_vals[1]):
                    diffs.append(intersection_x_vals[0])
                else:
                    diffs.append(intersection_x_vals[1])
            else:
                diffs.append(intersection_x_vals[0])

            coplt = [2 * coeflist[pidx, 0] * x_data[round(nstart)] + coeflist[pidx, 1], 0]
            coplt[1] = yplt[round(nstart)] - np.polyval(coplt, x_data[round(nstart)])
            xvb = np.arange(round(nstart) - 1000, round(nstart) + 1000)

            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            coef2d1 = np.polyfit(tsymbr, np.unwrap(np.angle(yplt[nsymbr])), 2)
            xvp = np.arange(round(nstart) - 100, round(nstart) + Config.nsamp + 100)

            xv2 = np.arange(start_pos2 + 1000, start_pos2 + Config.nsamp - 1000)
            coef2d2 = np.polyfit(x_data[xv2], np.unwrap(np.angle(yplt[xv2])), 2)
            logger.warning(
                f"res:{(coef2d1[2] - coeflist[pidx, 2]) / 2 / np.pi}  {(coef2d2[2] - coeflist[pidx + 1, 2]) / 2 / np.pi}")

            xvp2 = np.arange(start_pos2 - 100, start_pos2 + Config.nsamp + 100)
            coeflist = coeflist2.copy()
        plt.plot(dd2)
        plt.show()
        fig = go.Figure(layout_title_text="intersect points")
        pidx_range = np.arange(240)
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
        fig=px.line(dd, title="coef2 middle time")
        fig.show()










        anslst = []
        for pidx in range(1, 240):
            nstart = np.polyval(coeff_time, pidx)*Config.fs


            ans, c1, c2 = find_intersections(coeflist[pidx - 1: pidx + 1], nstart/Config.fs, 1e-6)

            if False:
                pxv1 = np.linspace(np.polyval(coeff_time, pidx-1), np.polyval(coeff_time, pidx) , 100000)
                pxv3 = np.arange(round(np.polyval(coeff_time, pidx-1) * Config.fs), round(np.polyval(coeff_time, pidx)* Config.fs))
                fig = go.Figure(layout_title_text=f"{pidx=} packet view")
                fig.add_trace(go.Scatter(x=pxv1, y=wrap(np.polyval(c1, pxv1))))
                fig.add_trace(go.Scatter(x=pxv3 / Config.fs, y=np.angle(pktdata_in[pxv3]), mode='markers'))
                fig.show()

            anslst.extend(ans)
            val = ans[0]
            pxv1 = np.linspace(val - 1e-4, val, 1000)
            pxv2 = np.linspace(val, val + 1e-4, 1000)
            pxv3 = np.arange(round((val - 1e-4) * Config.fs), round((val + 1e-4) * Config.fs))
            if False:
                fig = go.Figure(layout_title_text=f"{pidx=} intersection view")
                fig.add_trace(go.Scatter(x=pxv1, y=wrap(np.polyval(c1, pxv1))))
                fig.add_trace(go.Scatter(x=pxv2, y=wrap(np.polyval(c2, pxv2))))
                fig.add_trace(go.Scatter(x=pxv3/Config.fs, y=np.angle(pktdata_in[pxv3]), mode='markers'))
                fig.show()
                sys.exit(0)

            if len(ans) != 1: logger.error(f"ANSLEN {pidx=}, {ans=}")
        anslst = np.array(anslst)
        fig = go.Figure(layout_title_text="intersection points")
        fig.add_trace(go.Scatter(x=pidx_range[1:], y=anslst))
        fig.show()
        coeff_time = np.polyfit(pidx_range[1:], anslst, 1)
        print(coeff_time)
        print(
            f"estimated time:{coeff_time[0]} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
        fig.add_trace(go.Scatter(x=pidx_range[1:], y=np.polyval(coeff_time, pidx_range[1:])))
        fig.show()
        fig = go.Figure(layout_title_text="intersect points diff")
        fig.add_trace(go.Scatter(x=pidx_range[1:], y=anslst - np.polyval(coeff_time, pidx_range[1:])))
        fig.show()

        pidx_range = np.arange(240)
        f0list = []
        for pidx in pidx_range:
            f0 = np.polyval((2 * coeflist[pidx, 0], coeflist[pidx, 1]), np.polyval(coeff_time, pidx))
            f0list.append(f0 / 2 / np.pi)
        fig = go.Figure(layout_title_text="f0 from coeflist[:, 1] and coeftime")
        fig.add_trace(go.Scatter(x=pidx_range, y=f0list))
        # avgf0 = np.mean(f0list[32:])
        coeff1 = np.polyfit(pidx_range[32:], f0list[32:], 1)
        avgf0 = np.polyval(coeff1, 0)
        fig.add_hline(y=avgf0)
        print(f"{avgf0=} cfo ppm : {avgf0 + Config.bw/2=}")

        fig.show()

        f0list = []
        for pidx in pidx_range:
            f0 = np.polyval((2 * coeflist[pidx, 0], coeflist[pidx, 1]), np.polyval(coeff_time, pidx + 1))
            f0list.append(f0 / 2 / np.pi)
        fig = go.Figure(layout_title_text="f1 from coeflist[:, 1] and coeftime")
        fig.add_trace(go.Scatter(x=pidx_range, y=f0list))
        # avgf0 = np.mean(f0list[32:])
        coeff1 = np.polyfit(pidx_range[32:], f0list[32:], 1)
        avgf0 = np.polyval(coeff1, 0)
        fig.add_hline(y=avgf0)
        print(f"{avgf0=} cfo ppm : {avgf0 - Config.bw/2=}")

        fig.show()





        sys.exit(0)


    # accurately search for
    if True:
        nsymblen = 2 ** Config.sf / Config.bw * Config.fs * (1 - estf / Config.sig_freq)
        # coeff_time = [0.01008263, 0.01015366]
        coeflist = []
        for pidx in range(0, Config.preamble_len):
            nstart = pidx * nsymblen
            # nstart = np.polyval(coeff_time, pidx) * Config.fs

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            tsymbr = nsymbr / Config.fs
            fitmethod = "2dfit"
            if fitmethod == "2dfit":
                coefficients_2d = np.polyfit(tsymbr, np.unwrap(np.angle(pktdata_in[nsymbr])), 2)
            elif fitmethod == "1dfit":
                coefficients_2d = np.polyfit(
                    tsymbr,
                    np.unwrap(np.angle(pktdata_in[nsymbr])) - np.polyval((betat, 0, 0), tsymbr),
                    1
                )
                coefficients_2d = [betat, *coefficients_2d]
            logger.warning(f"{fitmethod=} {coefficients_2d=} {betat=} {betai=}")

            
            if True:
                plot_fit2d(coefficients_2d, estf, pidx, pktdata_in)

            coefficients_2d[-1] -= cp.angle(
                pktdata_in[nsymbr].dot(cp.exp(-1j * cp.polyval(coefficients_2d, tsymbr))))

            val = obj(tsymbr, pktdata_in[nsymbr], coefficients_2d)
            logger.warning(f'direct linear fit {val=}')
            rangeval = 0.005
            for i in range(20):
                for j in range(2):
                    xvals = np.linspace(coefficients_2d[j]*(1-rangeval), coefficients_2d[j]*(1+rangeval), 1001)
                    yvals = []
                    for x in xvals:
                        coef2 = coefficients_2d.copy()
                        coef2[j] = x
                        yvals.append(obj(tsymbr, y_data, coef2))
                    oldv = coefficients_2d[j]
                    coefficients_2d[j] = xvals[np.argmax(yvals)]
                    valnew = obj(tsymbr, y_data, coefficients_2d)
                    if True:#valnew < val*(1-1e-7):
                        fig = go.Figure(layout_title_text=f"{pidx=} {i=} {j=} {val=} {valnew=}")
                        fig.add_trace(go.Scatter(x=xvals, y=yvals))
                        fig.add_vline(x=oldv)
                        fig.show()
                    assert valnew >= val*(1-1e-7), f"{val=} {valnew=} {i=} {j=} {coefficients_2d=} {val-valnew=}"
                    if abs(valnew - val) < 1e-7: rangeval /= 2
                    val = valnew
            coefficients_2d[-1] -= np.angle(y_data.dot(np.exp(-1j * np.polyval(coefficients_2d, tsymbr))))
            coeflist.append(coefficients_2d)
            if True:
                # plot diff here
                fig = go.Figure(layout_title_text=f"{pidx=} fitdiff")
                ydatap = np.unwrap(np.angle(tocpu(pktdata_in[xvp])))
                coefficients_2d_old[-1] += np.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coefficients_2d_old, tsymbr))))
                coefficients_2d[-1] += np.angle(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coefficients_2d, tsymbr))))
                fig.add_trace( go.Scatter(x=x_data[xvp], y=np.angle(pktdata_in[xvp] * np.exp(-1j * np.polyval(coefficients_2d_old, x_data[xvp]))), mode='markers', marker=dict(size=1)))
                fig.add_trace( go.Scatter(x=x_data[xvp], y=np.angle(pktdata_in[xvp] * np.exp(-1j * np.polyval(coefficients_2d, x_data[xvp]))), mode='markers', marker=dict(size=1)))
                fig.update_layout(yaxis=dict(range=[-2, 2]), )
                fig.show()
                val1 = np.abs(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coefficients_2d_old, tsymbr))))
                val2 = np.abs(pktdata_in[nsymbr].dot(np.exp(-1j * np.polyval(coefficients_2d, tsymbr))))
                print(val1, val2)

                print(pidx, coefficients_2d[0], val, betanew, betai, (betai+betanew)/2)
                fig=go.Figure(layout_title_text=f"{pidx=} fit")
                xvp = np.arange(round(nstart) - 1000, round(nstart) + Config.nsamp + 1000)
                vdiff =  - (np.polyval(coefficients_2d, x_data[xvp[1000]]) - ydatap[1000])
                coefficients_2d[-1] += vdiff
                fig.add_trace(go.Scatter(x=x_data[xvp], y=ydatap))
                fig.add_trace(go.Scatter(x=x_data[xvp], y=np.polyval(coefficients_2d, x_data[xvp])))
                fig.add_vline(x=np.polyval(coeff_time, pidx))
                fig.add_vline(x=np.polyval(coeff_time, pidx+1))
                fig.show()
                coefficients_2d[-1] -= vdiff
                fig=go.Figure(layout_title_text=f"{pidx=} fitdiff")
                fig.add_trace(go.Scatter(x=x_data[xvp], y=ydatap - np.polyval(coefficients_2d_old, x_data[xvp]),  mode='markers', marker=dict(size=1)))
                fig.add_trace(go.Scatter(x=x_data[xvp], y=ydatap - np.polyval(coefficients_2d, x_data[xvp]), mode='markers', marker=dict(size=1)))
                fig.add_vline(x=np.polyval(coeff_time, pidx))
                fig.add_vline(x=np.polyval(coeff_time, pidx+1))
                fig.update_layout(yaxis=dict(range=[-2, 2]),)
                fig.show()


                coefficients_2d[-1] -= np.angle(y_data.dot(np.exp(-1j * np.polyval(coefficients_2d, tsymbr))))
                fig = go.Figure(layout_title_text=f"{pidx=} fitnwrap")
                xvp = np.arange(round(nstart) - 1000, round(nstart) + Config.nsamp + 1000)

                fig.add_trace(go.Scatter(x=x_data[xvp], y=np.angle(tocpu(pktdata_in[xvp]))))
                fig.add_trace(go.Scatter(x=x_data[xvp], y=wrap(np.polyval(coefficients_2d, x_data[xvp]))))
                fig.show()
            print(coefficients_2d)
        coeflist = np.array(coeflist)
        with open("coeftest.pkl", "wb") as fl: pickle.dump(coeflist, fl)
        fig=px.line(y=coeflist[:,0])
        fig.show()
        sys.exit(0)


    # suppose 2d[0] = betanew

    if True:
        diffs = []
        for pidx in pidx_range[:-1]:
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart) - 1000, round(nstart) + Config.nsamp * 2 + 1000)
            y_data = tocpu(cp.unwrap(cp.angle(pktdata_in[nsymbr])))
            y_data_1d = y_data - np.polyval((betanew, 0, 0), tsymbr)
            xv1 = np.arange(1000, Config.nsamp - 1000) + 1000
            coefficients_1da = np.polyfit(tsymbr[xv1], y_data_1d[xv1], 1)
            xv2 = np.arange(Config.nsamp + 1000, Config.nsamp * 2 - 1000) + 1000
            coefficients_1db = np.polyfit(tsymbr[xv2], y_data_1d[xv2], 1)

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
        coeff_time = np.polyfit(pidx_range[1:], diffs, 1)
        print(coeff_time)
        print(
            f"estimated time:{coeff_time[0]} cfo ppm from time: {1 - coeff_time[0] / Config.nsampf * Config.fs} cfo: {(1 - coeff_time[0] / Config.nsampf * Config.fs) * Config.sig_freq}")
        fig.add_trace(go.Scatter(x=pidx_range[1:], y=np.polyval(coeff_time, pidx_range[1:])))
        fig.show()
        fig = go.Figure(layout_title_text="intersect points diff")
        fig.add_trace(go.Scatter(x=pidx_range[1:], y=diffs - np.polyval(coeff_time, pidx_range[1:])))
        fig.show()

        c2d = []
        for pidx in pidx_range:
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            y_data = y_data_all[nsymbr]
            coefficients_2d = np.polyfit(tsymbr, y_data, 2)
            c2d.append(coefficients_2d[0])

        fig=px.line(y=c2d)
        fig.add_hline(y=betanew)
        fig.add_hline(y=betai)
        fig.show()


    if True:
        if False:
            fig = go.Figure(layout_title_text="y data all")
            fig.add_trace(go.Scatter(x=x_data_all, y=y_data_all[x_data_all]))
            fig.show()
        pidxs = []
        for pidx in pidx_range:
            # nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s
            nstart = np.polyval(coeff_time, pidx - 1)*Config.fs + est_to_s

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            y_data = y_data_all[nsymbr]
            # y_data = np.unwrap(np.angle(tocpu(pktdata_in[nsymbr])))
            coefficients_2d = np.polyfit(tsymbr, y_data, 2)
            y_data_1d = y_data - np.polyval((betanew, 0, 0), tsymbr)
            coefficients_1dx = np.polyfit(tsymbr, y_data_1d, 1)
            dvx.append((betanew, *coefficients_1dx))


            symb_in = pktdata_in[nsymbr]
            t = tsymbr - nstart/Config.fs
            phase = 2 * cp.pi * (0 * t - 0.5 * betanew/np.pi * t * t)
            newchirp = cp.exp(1j * togpu(phase))
            symb2 = symb_in * newchirp

            coefficients_1d = np.polyfit(tsymbr, np.unwrap(np.angle(tocpu(symb2))), 1)

            addpow = np.abs(tocpu(symb2).dot(np.exp(-1j * np.polyval((coefficients_1d[0], 0), tsymbr))))/len(symb_in)
            freq = coefficients_1d[0]/2/np.pi
            if pidx==199: print(pidx, coefficients_2d, betanew, coefficients_1dx, coefficients_1d, addpow, freq)
            if addpow > 0.5:
                pidxs.append(pidx)
                est_freq2.append(freq)
                est_pow.append(addpow)
            if pidx%50==0 or pidx<3 or pidx>237:#addpow < 0.5:
                ydata = np.unwrap(np.angle(tocpu(symb_in)))
                coefficients_2d = np.polyfit(tsymbr, ydata, 2)
                fig=go.Figure()
                xv2 = np.arange(round(nstart) - 1000, round(nstart) + Config.nsamp + 1000)
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.unwrap(np.angle(tocpu(pktdata_in[xv2]))), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.polyval(coefficients_2d, x_data[xv2]), mode='lines+markers'))
                fig.add_vline(x=nstart/Config.fs)
                fig.add_vline(x=np.polyval(coeff_time, pidx)  + est_to_s/Config.fs)
                fig.update_layout(title=f"{pidx=} diffline")
                fig.show()





        pidx_range2 = np.arange(Config.preamble_len+2, Config.preamble_len + 4)
        for pidx in pidx_range2:
            nstart = nsymblen * pidx * (1 - estf / Config.sig_freq) + est_to_s

            nsymbr = np.arange(round(nstart) + 1000, round(nstart) + Config.nsamp - 1000)
            y_data = y_data_all[nsymbr]
            # y_data = np.unwrap(np.angle(tocpu(pktdata_in[nsymbr])))
            coefficients_2d = np.polyfit(tsymbr, y_data, 2)
            print(pidx, coefficients_2d, betanew, betai)
            y_data_1d = y_data - np.polyval((-betanew, 0, 0), tsymbr)
            coefficients_1d = np.polyfit(tsymbr, y_data_1d, 1)
            # print(coefficients_2d, betanew, coefficients_1d )
            print(betanew, coefficients_1d)

            symb_in = pktdata_in[nsymbr]
            t = tsymbr - nstart/Config.fs
            phase = 2 * cp.pi * (0 * t + 0.5 * betanew/np.pi * t * t)
            newchirp = cp.exp(1j * togpu(phase))
            symb2 = symb_in * newchirp

            coefficients_1d = np.polyfit(tsymbr, np.unwrap(np.angle(tocpu(symb2))), 1)
            print(coefficients_1d)

            addpow = np.abs(tocpu(symb2).dot(np.exp(-1j * np.polyval((coefficients_1d[0], 0), tsymbr))))/len(symb_in)
            freq = coefficients_1d[0]/2/np.pi
            print(addpow, freq, 2 * betanew * freq/Config.fs/Config.fs, nstart/Config.fs)
            pidxs.append(pidx)
            est_freq2.append(freq - Config.bw*(1 - 2 * estf / Config.sig_freq))
            if True:
                ydata = np.unwrap(np.angle(tocpu(symb_in)))
                coefficients_2d = np.polyfit(tsymbr, ydata, 2)
                # coefficients_1d = np.polyfit(tsymbr, ydata, 1)
                # print(pidx, coefficients_2d, coefficients_1d, betanew)
                # freqreal = np.polyval((betanew / 2 / np.pi * Config.fs, coefficients_1d[0]/ 2 / np.pi*Config.fs - round(nstart) * betanew / 2 / np.pi* Config.fs), tsymbr)
                # fig = px.line(x=freqreal, y=ydata - np.polyval(coefficients_1d, tsymbr))
                fig=go.Figure()
                xv2 = np.arange(round(nstart) - 1000, round(nstart) + Config.nsamp + 1000)
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.unwrap(np.angle(tocpu(pktdata_in[xv2]))), mode='lines+markers'))
                fig.add_trace(go.Scatter(x=x_data[xv2], y=np.polyval(coefficients_2d, x_data[xv2]), mode='lines+markers'))
                fig.add_vline(x=nstart/Config.fs)
                fig.add_vline(x=(nstart + nsymblen * (1 - estf / Config.sig_freq)) /Config.fs)
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
        print(fdiff, (f0 + estf + fdiff)/betai/np.pi/2*Config.fs)
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
    print(time0 * Config.fs - 0.5 * nsymblen)
    fig.show()
    sys.exit(0)


def gen_refchirp(est_to_s, estf, length):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw)
    x = (cp.arange(length) - est_to_s) * (1 + estf / Config.sig_freq)
    yi = cp.zeros_like(x, dtype=np.complex64)
    bwnew = Config.bw * (1 + estf / Config.sig_freq)
    betanew = betai * (1 + 2 * estf / Config.sig_freq)
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
    nstart = est_to_s


    yi = gen_refchirp(est_to_s, estf, len(pktdata_in))

    retvals = 0
    for i in range(Config.preamble_len):
        nsymbr = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i+1) * Config.nsamp))
        retvals += cp.abs(pktdata_in[nsymbr].dot(cp.conj(yi[nsymbr]))) / len(nsymbr)
        # if i>=Config.preamble_len - 2:
        #     logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[nsymbr].dot(cp.conj(yi[nsymbr]))) / len(nsymbr)}")
    for i in range(Config.sfdpos, Config.sfdpos + 3):
        nsymbr = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i+1) * Config.nsamp))
        if i == Config.sfdpos + 2:
            nsymbr = cp.arange(round(est_to_s + i * Config.nsamp), round(est_to_s + (i + 0.25) * Config.nsamp))
        retvals += cp.abs(pktdata_in[nsymbr].dot(cp.conj(yi[nsymbr]))) / len(nsymbr)
        # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[nsymbr].dot(cp.conj(yi[nsymbr]))) / len(nsymbr)}")
    # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {retvals=}")

    return retvals



