# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

from utils import *

def objective_linear(cfofreq, time_error, pktdata2a):
    if time_error < 0: return -cfofreq, -time_error
    detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=Config.gen_refchirp_deadzone,
                               calctime=0)
    detect_symb_concat = cp.concatenate(detect_symb, axis=0)
    tint = math.ceil(time_error)
    logger.info(f"ObjLinear {time_error=} {tint=} {len(pktdata2a)=}")
    ress2 = -cp.unwrap(cp.angle(pktdata2a[tint:tint + len(detect_symb_concat)])) + cp.unwrap(cp.angle(detect_symb_concat))


    phasediff = ress2#cp.unwrap(cp.angle(cp.array(ress2)))
    est_dfreqs = []
    for fit_symbidx in range(0, Config.preamble_len):
        x_values = np.arange(Config.nsamp * fit_symbidx + 50, Config.nsamp * (fit_symbidx + 1) - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_dfreq = coefficients[0] * Config.fs / 2 / np.pi
        est_dfreqs.append(est_dfreq)
        # print(f"fitted curve {est_dfreq=:.2f} Hz")

    est_ups = []
    for fit_symbidx in range(Config.sfdpos, Config.sfdpos + 2):
        x_values = np.arange(Config.nsamp * fit_symbidx + 50, Config.nsamp * (fit_symbidx + 1) - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_ufreq = coefficients[0] * Config.fs / 2 / np.pi
        est_ups.append(est_ufreq)
        # print(f"fitted curve {est_ufreq=:.2f} Hz")
    ret_ufreq = np.mean(est_ups)
    # fig = px.scatter(y=est_dfreqs)
    ret_dfreq = np.mean(est_dfreqs[Config.skip_preambles:])
    # fig.add_hline(y=ret_dfreq)
    # fig.show()

    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    ret_freq = (ret_ufreq + ret_dfreq)/2
    ret_tdiff = (ret_ufreq - ret_dfreq)/2 / beta

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
    pktdata_in *= np.exp(1j * np.pi * Config.cfo_change_rate * (tstandard - pstart) ** 2 / Config.fs)
    codes = []
    angdiffs = []
    amaxdfs = []
    for pidx in range(Config.sfdpos + 2, round(Config.total_len)):
        dv1 = []
        dv2 = []
        freqrange = np.arange(0, 90, 0.1)
        start_pos_all_new = nsamp_small * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        # plt.plot(np.unwrap(np.angle(pktdata_in[start_pos - Config.nsamp: start_pos + Config.nsamp].get())))
        # plt.show()
        for debugfreq in freqrange:

            dt = (start_pos - start_pos_all_new) / Config.fs
            decode_matrix_a, decode_matrix_b = Config.decode_matrix_a, Config.decode_matrix_b

            beta = Config.bw / ((2 ** Config.sf) / Config.bw)

            df = -(est_cfo_f + dt * beta)
            # cfosymb = cp.exp(
            #     2j * cp.pi * df * cp.linspace(0, Config.nsamp / Config.fs, num=Config.nsamp, endpoint=False)).astype(
            #     cp.complex64)


            sig1 = add_freq(pktdata_in[start_pos: Config.nsamp + start_pos], debugfreq + df)
            sig2 = sig1.dot(decode_matrix_a.T)
            sig3 = sig1.dot(decode_matrix_b.T)

            codesv = cp.abs(sig2) ** 2 + cp.abs(sig3) ** 2
            angdv = cp.angle(sig3) - cp.angle(sig2)
            # fig = go.Figure(layout_title_text=f"decode {pidx=}")
            # fig.add_trace(go.Scatter(y=cp.abs(sig2).get(), mode="lines"))
            # fig.add_trace(go.Scatter(y=cp.abs(sig3).get(), mode="lines"))
            # fig.show()
            coderet = cp.argmax(cp.array(codesv))
            # coderet = cp.array(2732)
            # if abs(debugfreq)<1e-2: print("coderet", coderet)
            codes.append(coderet.item())
            angdiffs.append(angdv[coderet].item())

            debugval2 = cp.abs(sig2[coderet])
            debugval3 = cp.abs(sig3[coderet])
            dv1.append(debugval2.item())
            dv2.append(debugval3.item())
            # logger.warning(f"WO106 {debugfreq=} {pidx=}, {coderet=} {debugval2=} {debugval3=} {df=}")
        # fig = go.Figure(layout_title_text=f"decode angles")
        # fig.add_trace(go.Scatter(x=freqrange, y=dv1, mode="lines"))
        # fig.add_trace(go.Scatter(x=freqrange, y=dv2, mode="lines"))
        # fig.show()
        amaxdf = freqrange[np.argmax(np.array(dv1) ** 2 + np.array(dv2)**2)]
        print(f"{pidx=} {dt=} {dt*beta=} {est_cfo_f=} amax_df={amaxdf}")
        amaxdfs.append(amaxdf)
        # print(max(dv1), max(dv2), max(dv2)/(max(dv1)+max(dv2))*Config.n_classes)

    # fig = go.Figure(layout_title_text=f"decode angles")
    # fig.add_trace(go.Scatter(y=amaxdfs, mode="markers"))
    # fig.show()
    return codes, angdiffs




def objective_core_new(est_cfo_f, est_to_s, pktdata_in):
    vals = np.zeros(Config.sfdpos + 2, dtype=np.complex64)
    yvals2 = np.zeros(Config.sfdpos + 2, dtype=int)
    vallen = Config.preamble_len - Config.skip_preambles + 2

    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    for pidx in range(Config.sfdpos + 2):
        if pidx < Config.skip_preambles: continue
        if pidx >= Config.preamble_len and pidx < Config.sfdpos: continue

        start_pos_all_new = nsamp_small * pidx * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = round(start_pos_all_new)
        cfoppm1 = (1 + est_cfo_f / Config.sig_freq)  # TODO!!!
        tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1] + (start_pos - start_pos_all_new) / Config.fs
        if pidx <= Config.preamble_len:
            refchirp = mychirp(tstandard, f0=-Config.bw / 2 * cfoppm1 + est_cfo_f, f1=Config.bw / 2 * cfoppm1 + est_cfo_f,
                                t1=2 ** Config.sf / Config.bw * cfoppm1)
        else:
            refchirp = mychirp(tstandard, f0=Config.bw / 2 * cfoppm1 + est_cfo_f,
                               f1=-Config.bw / 2 * cfoppm1 + est_cfo_f,
                               t1=2 ** Config.sf / Config.bw * cfoppm1)
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * cp.conj(refchirp) / cp.sum(cp.abs(sig1))

        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        yval2 = cp.argmax(cp.abs(data0)).item()

        # print("objres", pidx, yval2 - Config.fft_n // 2, cp.max(cp.abs(data0)).item(), np.abs(data0[Config.fft_n // 2]))
        vals[pidx] = data0[yval2].item()
        yvals2[pidx] = yval2 - Config.fft_n // 2

    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    yup = np.mean(yvals2[Config.skip_preambles:Config.preamble_len])
    ydown = np.mean(yvals2[Config.sfdpos:Config.sfdpos + 2])
    est_cfo_f += (yup + ydown) / 2
    est_to_s -= (yup - ydown) / 2 / beta
    # print(yup, ydown, (yup + ydown) / 2, (yup - ydown) / 2)

    if False:
        freq = np.linspace(0, 2 * np.pi, 1000)
        vals2 = vals[Config.skip_preambles : Config.preamble_len]
        res2 = np.array([sum(vals2 * np.exp(np.arange(len(vals2)) * 1j * x)) for x in freq])
        retval = np.max(np.abs(res2)) / (Config.preamble_len - Config.skip_preambles)
        retvala = np.angle(res2[np.argmax(np.abs(res2))])
        vals2 = vals[Config.sfdpos : Config.sfdpos + 2]
        res2 = np.array([sum(vals2 * np.exp(np.arange(len(vals2)) * 1j * x)) for x in freq])
        retval2 = np.max(np.abs(res2)) / 2
        retvala2 = np.angle(res2[np.argmax(np.abs(res2))])
        print(f"newcore {retval=} {retvala=} {retval2=} {retvala2=}")
        print(f"newcore angles {np.diff(np.angle(vals[Config.skip_preambles : Config.preamble_len]))=} {np.diff(np.angle(vals[Config.sfdpos : Config.sfdpos + 2]))=}")

    freq = np.linspace(0, 2 * np.pi, 10000)
    res = np.array([vals.dot(np.exp(np.arange(len(vals)) * -1j * x)) for x in freq])
    retval = np.max(np.abs(res)) / vallen
    if False:
        vals2 = vals.copy()
        vals2[Config.preamble_len:] = 0
        res0 = np.array([vals2.dot(np.exp(np.arange(len(vals2)) * -1j * x)) for x in freq])
        # plt.plot(freq, np.abs(res)/vallen)
        # plt.show()
        # lst = np.concatenate((vals[Config.skip_preambles:Config.preamble_len], vals[Config.sfdpos:Config.sfdpos + 2]))
        lst = vals[:]#[Config.skip_preambles:]
        lst[Config.preamble_len] = lst[Config.preamble_len - 1] * (lst[Config.preamble_len - 1] / lst[Config.preamble_len - 2])
        lst[Config.preamble_len + 1] = lst[Config.preamble_len - 1] * (lst[Config.preamble_len - 1] / lst[Config.preamble_len - 2])**2
       # for plotting only
        lst = np.unwrap(np.angle(lst))
        # lst[Config.sfdpos:] += 2 * np.pi # for plotting only
        x_values = np.arange(Config.skip_preambles, Config.sfdpos + 2)
        plt.plot(x_values, lst[x_values])
        x_values = np.arange(Config.skip_preambles, Config.preamble_len)
        coefficients = np.polyfit(x_values, lst[x_values], 1)
        # coefficients[0] = 2.6164215092960035
        # coefficients[1] += (2.87-2.61)*8
        x_values = np.arange(Config.skip_preambles, Config.sfdpos + 2)
        plt.plot(x_values, np.poly1d(coefficients)(x_values), '--')
        plt.title(f"{retval=:.5f} fitup {coefficients[0]:.5f},{freq[np.argmax(np.abs(res0))]:.5f} fitall={freq[np.argmax(np.abs(res))]:.5f}")
        plt.show()
        print(f"{retval=} fitup {coefficients[0]},{freq[np.argmax(np.abs(res0))]} fitall={freq[np.argmax(np.abs(res))]} f+{(yup + ydown) / 2}  t+{-(yup - ydown) / 2 / beta}")
    return retval, est_cfo_f, est_to_s



def gen_refchirp(cfofreq, tstart, deadzone=0, calctime=0):
    detect_symb = []
    bw = Config.bw * (1 + cfofreq / Config.sig_freq)
    sigt = 2 ** Config.sf / bw * Config.fs #* (1 - cfofreq / Config.sig_freq)
    beta = Config.bw / sigt
    # sigt *= (1 + 1e-6 * calctime)# (1 + cfofreq / Config.sig_freq)
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



