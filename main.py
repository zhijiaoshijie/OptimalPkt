import plotly.express as px
import plotly.graph_objects as go
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
    cfo = 0
    sfo = cfo * Config.fs / Config.sig_freq
    to = -1 / Config.fs * 0
    assert to <= 0, "Time Offset must be <= 0"

    t_all = cp.arange(Config.nsamp, dtype=cp.float32) * 1 / (Config.fs + sfo) + to

    symb_idx = 1000
    istart = 0 #Config.nsamp

    f0 = -Config.bw / 2 + cfo
    f1 = Config.bw / 2 + cfo
    t1 = Config.nsamp / Config.fs
    t0 = 0
    t = t_all[:Config.nsamp]
    phase0 = phase1 = 0

    beta = (f1 - f0) / (t1 - t0)
    f0R = f0 + (f1 - f0) * (symb_idx / Config.n_classes)
    tjump = t0 + (t1 - t0) * (1 - symb_idx / Config.n_classes)

    phaseA = 2 * cp.pi * (f0R * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase0
    phaseA[t >= tjump] = 0
    phaseB = 2 * cp.pi * ((f0R - (f1 - f0)) * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase1
    phaseB[t < tjump] = 0
    data = phaseA + phaseB

    data[t_all < istart / Config.fs] = 0
    data[t_all >= (istart + Config.nsamp) / Config.fs] = 0

    phase3 = 0
    t = cp.arange(Config.nsamp, dtype=cp.float32) * 1 / Config.fs
    fig = go.Figure()
    for t0 in (0, t[1] * 10):
        # t = cp.linspace(0, (Config.nsamp + 1) / Config.fs, Config.nsamp + 1)[:-1]
        f0 = Config.bw / 2
        f1 = -Config.bw / 2
        beta = (f1 - f0) / (2 ** Config.sf / Config.bw)
        downchirp = 2 * cp.pi * (f0 * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase3

        dataX = data + downchirp

        code = symb_idx
        kk = int(Config.fs / Config.bw)
        dataX1 = dataX[:-code * kk]
        dataX2 = dataX[-code * kk:]

        freq = code * Config.bw / Config.n_classes
        fsig = t * 2 * cp.pi * freq
        fsig2 = t * 2 * cp.pi * (freq - Config.bw)

        fig.add_trace(go.Scatter(y=dataX.get(), mode='lines', name='fit', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=fsig.get(), mode='lines', name='fit', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=fsig2.get(), mode='lines', name='fit2', line=dict(color='red')))
        d1 = dataX1 - fsig[:-code * kk]
        d2 = dataX2 - fsig2[-code * kk:]
        print(f"d1avg: {cp.mean(d1)} d2avg: {cp.mean(d2)}")
    fig.update_layout(title="dataX, fsig, fsig2")
    fig.show()



if __name__ == "__main__":
    test()
