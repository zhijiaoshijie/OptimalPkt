import time
import csv

from work import *
from reader import *

if __name__ == "__main__":

    script_path = __file__
    mod_time = os.path.getmtime(script_path)
    readable_time = time.ctime(mod_time)
    logger.warning(f"Last modified time of the script: {readable_time}")

    fulldata = []
    # Main loop read files
    for file_path, file_path_id in Config.file_paths_zip:
        thresh = preprocess_file(file_path)

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        codescl = []
        canglescl = []
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):
            read_idx, data1, data2 = pkt_data
            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == len(data1) // Config.nsamp: continue

            # normalization
            data1 /= cp.mean(cp.abs(data1))
            data2 /= cp.mean(cp.abs(data1))

            logger.info(f"Prework {pkt_idx=} {len(data1)=}")
            est_cfo_f = 0
            est_to_s = 0
            # iterate 2 times to detect, each time based on estimations of the last time
            for tryi in range(2):
                f, t, retval = work(data1, est_cfo_f, est_to_s)
                if t <= 0:
                    logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                    break

            if t > 0 and abs(f + 38000) < 4000:
                est_cfo_f, est_to_s = f, t
                est_to_s_full = est_to_s + (read_idx * Config.nsamp)
                logger.warning(f"est f{file_path_id:3d} {est_cfo_f=:.6f} {est_to_s=:.6f} {pkt_idx=:3d} {read_idx=:5d} tot {est_to_s_full:15.2f} {retval=:.6f}")
                fulldata.append([file_path_id, pkt_idx, est_cfo_f, est_to_s_full , retval])
                if False: # code for writing symbol data
                    sig1 = data1[round(est_to_s): Config.nsamp * (Config.total_len ) + round(est_to_s)]
                    sig2 = data2[round(est_to_s): Config.nsamp * (Config.total_len ) + round(est_to_s)]
                    sig1.tofile(f"fout/data0_test_{file_path_id}_pkt_{pkt_idx}")
                    sig2.tofile(f"fout/data1_test_{file_path_id}_pkt_{pkt_idx}")
            else:
                est_cfo_f, est_to_s = f, t
                logger.error(f"ERR f{file_path_id} {est_cfo_f=} {est_to_s=} {pkt_idx=} {read_idx=} tot {est_to_s + read_idx * Config.nsamp} {retval=}")

            if False:  # code for writing CSV file
                header = ["fileID", "pktID", "CFO", "Time offset", "Power"]
                # header.extend([f"Angle{x}" for x in range(Config.total_len)])
                # header.extend([f"Abs{x}" for x in range(Config.total_len)])
                csv_file_path = 'data_out.csv'
                with open(csv_file_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(header)  # Write the header
                    for row in fulldata:
                        csvwriter.writerow(row)
        if False: # code for reading
            bytes_per_complex = 8
            byte_offset = round(est_to_s_full) * bytes_per_complex
            L = Config.nsamp * Config.total_len

            with open(file_path, 'rb') as f:
                # Move the file pointer to the desired byte offset
                f.seek(byte_offset)

                # Read L complex64 numbers (L * 8 bytes)
                data = f.read(L * bytes_per_complex)

                # Ensure that the correct amount of data was read
                if len(data) != L * bytes_per_complex:
                    raise ValueError(f"Expected to read {L * bytes_per_complex} bytes, but got {len(data)} bytes.")

            # Convert the byte data to a CuPy array of type complex64
            rawdata = cp.frombuffer(data, dtype=cp.complex64)
            rawdata.tofile(f"fout/dataZ_test_{file_path_id}_pkt_{pkt_idx}")


