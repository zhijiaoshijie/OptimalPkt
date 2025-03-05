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
from tabulate import tabulate
import urllib.parse
from datetime import datetime

def get_file_details(file_path):
    """Get file details like size, last modified, and whether it's a directory."""
    if os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        is_dir = os.path.isdir(file_path)
        return [file_name, size, last_modified, is_dir, file_path]
    else:
        return [file_path, "N/A", "N/A", "N/A", "N/A"]  # Handle non-existent paths

def is_match(file_path: str, pattern: str) -> bool:
    # judge if the file path matches the regex provided by the user
    file_path = file_path[1:] # remove the first '/'
    return pattern is None or fnmatch.fnmatch(file_path, pattern)

def dfs_search_files(sess, token: str,
                     path: str = "/", 
                     pattern: str = None) -> list:
    filelist = []
    encoded_path = urllib.parse.quote(path)
    r = sess.get(f'https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{token}/dirents/?path={encoded_path}')
    objects = r.json()['dirent_list']
    for obj in objects:
        if obj["is_dir"]:
            filelist.extend(
                dfs_search_files(sess, token, obj['folder_path'], pattern))
        elif is_match(obj["file_path"], pattern):
            filelist.append(obj)
    return filelist

def human_readable_size(size_bytes):
    """Convert bytes to a human-readable format (KB, MB, GB, etc.)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"  # For sizes larger than TB

def upload_files(share_url, repo_url, fnames, quiteFlag=False):
    if not quiteFlag:
        print(f"upload filenames: {len(fnames)=}")
        table_data = [get_file_details(file_path) for file_path in fnames]

        # Define headers
        headers_table = ["File Name", "Size (Bytes)", "Last Modified", "Is Directory", "File Path"]

        # Print the table
        print(tabulate(table_data, headers=headers_table, tablefmt="grid"))
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
    with open('1.html', 'wb') as f:
        f.write(response.content)
    response = response.content.decode('utf-8').split('\n')
    line_token = list(filter(lambda x: ('token' in x), response))
    token = re.compile(r"token: [\"\'](?P<url>[-\w]+)[\"\']").search(line_token[0]).groupdict()['url']
    print(f"Get token: {token}")
    newurl = f'https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{token}/dirents'
    response = requests.get(newurl, cookies=cookies, headers=headers)
    print('requesting', newurl, response.status_code)
    data = response.json()
    dirent_list = data['dirent_list']

    # Prepare the table data
    table_data = []
    for dirent in dirent_list:
        table_data.append([
            dirent['file_name'],
            dirent['size'],
            dirent['last_modified'],
            dirent['is_dir'],
            dirent['file_path']
        ])
    for row in table_data:
        row[1] = human_readable_size(row[1])
    # Define table headers
    table_headers = ["File Name", "Size (Bytes)", "Last Modified", "Is Directory", "File Path"]

    # Print the table
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

    newurl = f'https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{token}/upload/'
    response = requests.get(newurl, cookies=cookies, headers=headers)
    print('requesting', newurl, response.status_code)
    # n2 = f'https://cloud.tsinghua.edu.cn/api2/repos/{token}/dir/?p=%2F2'
    # print('requesting', n2, response.status_code)
    assert (response.status_code == 200)
    upload_link = response.json()["upload_link"]
    print('upload_link', upload_link)
    # Get the file size
    # from downloader import *
    #
    # token = get_share_key(share_url)
    # verify_password(token)
    #
    # # search files
    # logging.info("Searching for files to be downloaded, Wait a moment...")
    filelist = dfs_search_files(session, token, pattern=None)
    for file in filelist:
        fnamex = os.path.basename(file["file_path"])
        for fname in fnames:
            if os.path.basename(fname) == fnamex: fnames.remove(fname)

    if not quiteFlag:
        print(f"filtered upload filenames: {len(fnames)=}")
        table_data = [get_file_details(file_path) for file_path in fnames]

        # Define headers
        headers_table = ["File Name", "Size (Bytes)", "Last Modified", "Is Directory", "File Path"]

        # Print the table
        print(tabulate(table_data, headers=headers_table, tablefmt="grid"))
    else:
        print(f"start uploading...")
    if len(fnames) == 0:
        print(f"No files to upload.")
        return True


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
                    # 'parent_dir': (''+os.path.dirname(fname), '/'+os.path.dirname(fname))
                    'parent_dir': ('', '/')
                }
            )

            if not quiteFlag:
                print(f'[{idx}/{len(fnames)}] Uploading {fname}')
            with tqdm(total=file_size, unit='B', unit_scale=True, smoothing=0, disable=quiteFlag) as pbar:
                monitor = MultipartEncoderMonitor(encoder, create_callback(pbar))

                response = requests.post(upload_link, data=monitor, cookies=cookies,
                                         headers={'Content-Type': monitor.content_type})
            if not quiteFlag:
                print(response.status_code, response.text)
            if response.status_code != 200:
                print('restarting...')
                return False
            else:
                return True

while True:
    try:
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

        if upload_files(share_url, repo_url, fnames, quiteFlag=False): break

    except urllib3.exceptions.SSLError as e:
        print(e)

        # threads = []
        # for fname in fnames:
        #     if os.path.basename(fname)[:3] in ('sf8', 'sf9', 'sf1') and os.path.isfile(fname):
        #         thread = threading.Thread(target=execute_command, args=(fname,))
        #         thread.start()
        #         threads.append(thread)
        # for thread in threads: thread.join()

