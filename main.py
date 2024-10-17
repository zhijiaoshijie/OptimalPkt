import plotly.express as px
import plotly.graph_objects as go

import scipy.optimize as opt
import cupy as cp

# Define the negative of the function for maximization


import cupy as cp
cp.cuda.Device(0).use()


class Config:
    sf = 11
    bw = 125e3
    fs = 1e6
    sig_freq = 470e6

    n_classes = 2 ** sf
    tsig = n_classes / bw * fs
    nsamp = round(n_classes * fs / bw)


Config = Config()


def test():
    cfo = 20000
    sfo = cfo * Config.fs / Config.sig_freq
    to = -1 / Config.fs * 10 # to < 0: we started recving early
    # assert to <= 0, "Time Offset must be <= 0"
    
    # their symbol: standard time
    # our sampling: offset time
    tsamp = cp.arange(Config.nsamp, dtype=cp.float32) * 1 / (Config.fs + sfo) + to

    # they sent a standard symbol
    for symb_idx in range(200, 2048, 200):
        phase0 = phase1 = 0

        beta = Config.bw / (2 ** Config.sf / Config.bw)
        f0R = -Config.bw / 2  + cfo + Config.bw * (symb_idx / Config.n_classes)
        tjump = 2 ** Config.sf / Config.bw * (1 - symb_idx / Config.n_classes)

        phaseA = 2 * cp.pi * (f0R * tsamp + 0.5 * beta * tsamp ** 2) + phase0
        phaseA[tsamp >= tjump] = 0
        phaseB = 2 * cp.pi * ((f0R - Config.bw) * tsamp + 0.5 * beta * tsamp ** 2) + phase1
        phaseB[tsamp < tjump] = 0
        data = phaseA + phaseB

        # sig power, out of symb = -
        power = cp.ones(Config.nsamp)
        power[tsamp < 0] = 0
        power[tsamp >= 2 ** Config.sf / Config.bw] = 0

        # data is generated on asynced tsamp, it has cfo and sfo and to

        # generate standard downchirp
        tsamp2 = cp.arange(Config.nsamp, dtype=cp.float32) * 1 / Config.fs
        beta = Config.bw / (2 ** Config.sf / Config.bw)
        downchirp = 2 * cp.pi * (Config.bw / 2 * tsamp2 - 0.5 * beta * tsamp2 ** 2)

        # dechirp
        dataX = data + downchirp

        # fig=go.Figure()
        # fig.add_trace(go.Scatter(y=dataX.get(), mode='lines', name='fit', line=dict(color='blue')))
        # fig.show()
        # break

        # fit with codes
        # coarse search
        resolution = 10000
        dataX2C = cp.zeros(resolution, dtype=cp.float32)
        freq_range = cp.linspace(-Config.bw, Config.bw, resolution)

        def func(freq):
            freqsymb = 2 * cp.pi * (cp.array(freq) * tsamp2)
            dataX2 = dataX - freqsymb
            return cp.abs(cp.sum(cp.exp(1j * dataX2.astype(complex)) * power))


        for fid, freq in enumerate(freq_range):
            dataX2C[fid] = func(freq)
        start_freq = freq_range[cp.argmax(dataX2C)].get()

        def neg_func(freq):
            return -func(freq).get()  # Convert to numpy for scipy compatibility
        result = opt.minimize(neg_func, start_freq, method='L-BFGS-B')
        max_freq = result.x[0]
        # print("Maximum frequency:", max_freq)
        # print("Function value at maximum frequency:", -result.fun)

        # fig = px.line(x = freq_range.get(), y = dataX2C.get())
        # fig.add_shape( type="line", y0=dataX2C.get().min(), x0=max_freq, y1=dataX2C.get().max(), x1=max_freq, line=dict(color="Red", width=2) )
        # fig.show()
        sig_freqmax = 2 * cp.pi * (cp.array(max_freq) * tsamp2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tsamp2.get(), y=dataX.get(), mode='lines', name='dataX'))
        fig.add_trace(go.Scatter(x=tsamp2.get(), y=sig_freqmax.get(), mode='lines', name='sig_freqmax', line=dict(dash='dot')))
        fig.show()
        break


        code = symb_idx
        kk = int(Config.fs / Config.bw)
        dataX1 = dataX[:-code * kk]
        dataX2 = dataX[-code * kk:]

        freq = code * Config.bw / Config.n_classes
        fsig = tsamp2 * 2 * cp.pi * freq
        fsig2 = tsamp2 * 2 * cp.pi * (freq - Config.bw)

        # fig.add_trace(go.Scatter(y=dataX.get(), mode='lines', name='fit', line=dict(color='blue')))
        # fig.add_trace(go.Scatter(y=fsig.get(), mode='lines', name='fit', line=dict(color='red')))
        # fig.add_trace(go.Scatter(y=fsig2.get(), mode='lines', name='fit2', line=dict(color='red')))
        d1 = dataX1 - fsig[:-code * kk]
        d2 = dataX2 - fsig2[-code * kk:]
        print(f"d1avg: {cp.mean(d1)} d2avg: {cp.mean(d2)} symb:{symb_idx}")
    # fig.update_layout(title="dataX, fsig, fsig2")
    # fig.show()



if __name__ == "__main__":
    test()
