from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
import plotly.express as px

from utils import *



def preprocess_file(file_path):
    #  read file and count size
    logger.info(f"FILEPATH {file_path}")
    pkt_cnt = 0
    pktdata = []
    fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
    logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')
    # read max power of first 5000 windows, for envelope detection
    power_eval_len = 1000
    nmaxs = []
    for idx, rawdata in enumerate(read_large_file(file_path)):
        nmaxs.append(cp.max(cp.abs(rawdata)))
        if idx == power_eval_len - 1: break
    nmaxs = tocpu(cp.array(nmaxs))
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
    thresh_manual = None
    if "_farest" in file_path:
        thresh_manual = 0.02
    if thresh < 0.01:
        logger.error(f"ERR too small thresh check {thresh=} {mean1=} {mean2=} {file_path=}")
    # # <<< PLOTFIG FOR POWER ENVELOPE DETECTION >>>
    if False:
        counts, bins = cp.histogram(togpu(nmaxs), bins=100)
        # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
        threshpos = np.searchsorted(tocpu(bins), thresh).item()
        logger.debug(f"lower: {cp_str(counts[:threshpos])}")
        logger.debug(f"higher: {cp_str(counts[threshpos:])}")
        # fig = px.line(nmaxs)
        # fig.add_hline(y=thresh, line_color='Black')
        # if thresh_manual is not None: fig.add_hline(y=thresh_manual, line_color='Red')
        # fig.update_layout(title=f"powermap of {file_path} length {len(nmaxs)}")
        # fig.show()
        plt.plot(nmaxs)
        plt.axhline(y=thresh, color='black', linestyle='-', label=f'Threshold (Auto): {thresh}')
        if thresh_manual is not None:
            plt.axhline(y=thresh_manual, color='red', linestyle='-', label=f'Threshold (Manual): {thresh_manual}')
        plt.title(f"Powermap of {file_path} length {len(nmaxs)} {thresh=} {thresh_manual=}")
        plt.show()

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

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
        else:
            if len(current_sequence1) > min_length:
                current_sequence1.append(rawdata1) # end +1 window
                yield read_idx + 1 - len(current_sequence1), cp.concatenate(current_sequence1)
            current_sequence1 = [rawdata1,] # previous +1 window

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1)


