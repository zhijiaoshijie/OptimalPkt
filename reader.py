from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
import plotly.express as px

from utils import *
from pltfig import pltfig1
import scipy.signal as signal
from mainwork import mainwork

def preprocess_file(file_path, fftflag = False, draw=False, thresh_manual = None):
    #  read file and count size
    logger.info(f"FILEPATH {file_path}")
    pkt_cnt = 0
    pktdata = []
    fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
    logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')
    # read max power of first 5000 windows, for envelope detection

    power_eval_len = 5000
    power_skip_len = 51

    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    tstandard = cp.arange(Config.nsamp) / Config.fs
    refchirp = cp.exp(-1j * 2 * cp.pi * (-Config.bw * 0.5 * tstandard + 0.5 * beta * tstandard * tstandard))

    nmaxs = []
    data2s = cp.zeros((Config.preamble_len, Config.nsamp), dtype=cp.float32)
    for idx, rawdata in enumerate(read_large_file(file_path)):
        if fftflag:
            data2 = rawdata * refchirp
            data3 = cp.abs(myfft(data2, Config.nsamp, Config.plan2))
            data3a = cp.convolve(data3, cp.array([1, 1, 1]), mode='same')
            data2s[idx % Config.preamble_len] = data3a
            datav = cp.sum(data2s, axis=0)
            nmaxs.append(cp.max(datav).item())
        else:
            if idx >= power_skip_len:
                nmaxs.append(cp.max(cp.abs(rawdata)))
        if idx == power_eval_len - 1: break
    nmaxs = tocpu(cp.array(nmaxs))
    peaks, properties = signal.find_peaks(nmaxs, prominence=0.5, distance=Config.total_len)  # Detect peaks above height 0
    print(peaks)

    # Plot result
    # pltfig1(None, nmaxs, addvline=peaks).show()
    # pltfig1(None, peaks, title="peak positions").show()
    peaks = cp.array(peaks) - Config.preamble_len - 2
    for peak in peaks[1:5]:
        with open(file_path, 'rb') as file:
            # Move the file pointer to the desired position (e.g., 100 bytes from the start)
            file.seek(around(max(peak, 0) * Config.nsamp * 4 * 2))
            rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp * (Config.total_len + 10))
            rawdata.tofile(f"test{peak}.sigdat")
    sys.exit(0)
    # clustering


    data = nmaxs.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data)
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    sorted_indices = np.argsort(means)
    mean1, mean2 = means[sorted_indices]
    covariance1, covariance2 = covariances[sorted_indices]
    weight1, weight2 = weights[sorted_indices]
    # threshold to divide the noise power from signal power
    thresh = (mean1 * covariance2 + mean2 * covariance1) / (covariance1 + covariance2)
    if thresh < 0.01:
        logger.error(f"ERR too small thresh check {thresh=} {mean1=} {mean2=} {file_path=}")
    # # <<< PLOTFIG FOR POWER ENVELOPE DETECTION >>>
    if draw or thresh_manual:
        counts, bins = cp.histogram(togpu(nmaxs), bins=100)
        # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
        threshpos = np.searchsorted(tocpu(bins), thresh).item()
        logger.warning(f"lower: {cp_str(counts[:threshpos])}")
        logger.warning(f"higher: {cp_str(counts[threshpos:])}")
        fig = px.line(nmaxs)
        if thresh_manual is not None: fig.add_hline(y=thresh_manual, line_color='Red')
        fig.add_hline(y=thresh, line_color='Black')
        fig.update_layout(title=f"powermap of {file_path} length {len(nmaxs)}")
        fig.show()
        # plt.plot(nmaxs)
        # plt.axhline(y=thresh, color='black', linestyle='-', label=f'Threshold (Auto): {thresh}')
        # if thresh_manual is not None:
        #     plt.axhline(y=thresh_manual, color='red', linestyle='-', label=f'Threshold (Manual): {thresh_manual}')
        # plt.title(f"Powermap of {file_path} length {len(nmaxs)} {thresh=} {thresh_manual=}")
        # plt.show()

    # thresh = max(thresh, 0.01)
    # if threshold may not work set this to True
    # plot the power map

    if thresh_manual is not None: thresh = thresh_manual
    return thresh

def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        # t = 1.45e6
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
                # t-=len(rawdata)
            except EOFError:
                logger.info(f"file complete with EOF {file_path_in=}")
                break
            if len(rawdata) < Config.nsamp:
                logger.info(f"file complete{file_path_in=}, {len(rawdata)=}")
                break
            # if t<0:
            #     plt.scatter(x=np.arange(Config.nsamp - 1),
            #                 y=cp.diff(cp.unwrap(cp.angle(rawdata[:Config.nsamp]))).get(), s=0.2)
            #     plt.show()
            yield rawdata



def read_pkt(file_path_in1, threshold, min_length=15):
    current_sequence1 = []

    read_idx = -1
    for rawdata1 in read_large_file(file_path_in1):
        read_idx += 1

        number1 = cp.max(cp.abs(rawdata1))
        # if read_idx > 12564: logger.warning(f"{read_idx=} {number1=}")

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
        else:
            if len(current_sequence1) > min_length:
                # if read_idx > 12564:
                     #logger.warning(f"end {read_idx=} {threshold=} {number1=}")
                     # pltfig1(None, cp.unwrap(cp.angle(rawdata1)), title="read_pkt ending code").show()
                current_sequence1.append(rawdata1) # end +1 window
                yield read_idx + 1 - len(current_sequence1), cp.concatenate(current_sequence1)
            current_sequence1 = [rawdata1,] # previous +1 window

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1)


