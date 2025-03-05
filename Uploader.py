import subprocess
import json
import re
import sys
import os
from tqdm import tqdm
import threading
import requests
from pprint import pprint
import requests
import urllib3
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

wFlag = True
while wFlag:
    try:
        wFlag = False
        # No Proxy
        session = requests.Session()
        session.trust_env = False

        # REPO URL
        # generate share link with upload and download
        repo_url = 'https://cloud.tsinghua.edu.cn/d/248a8c6a122748fa8ec8/'
        share_url = 'https://cloud.tsinghua.edu.cn/u/d/8f19b4142f9c4a0c9132/'
        # upload_fpath = r'/data/djl/datasets/Dataset_50Nodes'
        # fnames_raw = [os.path.join(root, filename) for root, dirs, file_in_dir in os.walk(upload_fpath) for filename in file_in_dir]
        # fnames = sorted(fnames_raw)
        fnames = ["/data/djl/OptimalPkt/farm_cover_sf10_1.tar.gz", "/data/djl/OptimalPkt/lot_cover_sf10_1.tar.gz"]

        print(f"upload filenames: {len(fnames)=}")
        print('\n'.join(fnames))

        cookies = {
            'sessionid': '7dw70y8ti02g0ywh99zorg1ilitzwnry',
            'sfcsrftoken': 'trkSr7urBzHPu6rZSSOC27O4IGB72697MvmLQXOWUC3n2v7eWay3cXVe3Ii3FdLC',
            'serverid': '3',
        }

        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en,zh;q=0.9,zh-CN;q=0.8',
            'cache-control': 'max-age=0',
            # 'cookie': 'sessionid=e8qgwv2rdtpflywmdrfjhsz8v4fomu5a; sfcsrftoken=t1KMw7CpW9eBMMzTRPGyGFn4dx830TsAXoH2sUqTsHNOQhUDiIwFvoHdvhDVEdzI; serverid=6',
            '^sec-ch-ua': '^\\^Google',
            'sec-ch-ua-mobile': '?0',
            '^sec-ch-ua-platform': '^\\^Windows^^^',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        }

        response = requests.get(repo_url, cookies=cookies, headers=headers)

        print('requesting', repo_url, response.status_code)
        assert (response.status_code == 200)
        response = response.content.decode('utf-8').split('\n')
        line_token = list(filter(lambda x: ('token' in x), response))
        token = re.compile(r"token: [\"\'](?P<url>[-\w]+)[\"\']").search(line_token[0]).groupdict()['url']

        newurl = f'https://cloud.tsinghua.edu.cn/api/v2.1/upload-links/{token}/upload/'
        response = requests.get(newurl, cookies=cookies, headers=headers)
        print('requesting', newurl, response.status_code)
        assert (response.status_code == 200)
        upload_link = response.json()["upload_link"]

        print('upload_link', upload_link)

        # Get the file size

        from downloader import *

        share_key = get_share_key(share_url)
        verify_password(share_key)

        # search files
        logging.info("Searching for files to be downloaded, Wait a moment...")
        filelist = dfs_search_files(share_key, pattern=None)
        for file in filelist:
            fnamex = os.path.basename(file["file_path"])
            for fname in fnames:
                if os.path.basename(fname) == fnamex: fnames.remove(fname)

        print(f"filtered upload filenames: {len(fnames)=}")
        print('\n'.join(fnames))

        # Create and start a thread for each command
        for idx, fname in enumerate(fnames):
            file_size = os.path.getsize(fname)


            def create_callback(pbar):
                # Define a callback function to update the progress bar
                def callback(monitor):
                    pbar.update(monitor.bytes_read - pbar.n)

                return callback


            # Open the file in binary mode
            with open(fname, 'rb') as f:
                encoder = MultipartEncoder(
                    fields={
                        'file': (fname, f),
                        'parent_dir': ('', '/')
                    }
                )

                print(f'[{idx}/{len(fname)}] Uploading {fname}')
                with tqdm(total=file_size, unit='B', unit_scale=True, smoothing=0) as pbar:
                    monitor = MultipartEncoderMonitor(encoder, create_callback(pbar))

                    response = requests.post(upload_link, data=monitor, cookies=cookies,
                                             headers={'Content-Type': monitor.content_type})
                print(response.status_code, response.text)
                if response.status_code != 200:
                    wFlag = True
                    print('restarting...')
                    break

    except urllib3.exceptions.SSLError as e:
        print(e)
        wFlag = True

        '''
        threads = []
        for fname in fnames: 
            if os.path.basename(fname)[:3] in ('sf8', 'sf9', 'sf1') and os.path.isfile(fname):
                thread = threading.Thread(target=execute_command, args=(fname,))
                thread.start()
                threads.append(thread)
        for thread in threads: thread.join()'''

