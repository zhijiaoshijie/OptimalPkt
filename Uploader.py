import subprocess
import json
import re
import sys
import os
from tqdm import tqdm
import threading
import requests
from pprint import pprint
from requests_toolbelt.multipart.encoder import MultipartEncoder
# No Proxy
session = requests.Session()
session.trust_env = False

# REPO URL
# generate share link with upload and download
repo_url = 'https://cloud.tsinghua.edu.cn/u/d/e00b1713bbaa47b68266/'

cookies = {
    'sessionid': 'e8qgwv2rdtpflywmdrfjhsz8v4fomu5a',
    'sfcsrftoken': 't1KMw7CpW9eBMMzTRPGyGFn4dx830TsAXoH2sUqTsHNOQhUDiIwFvoHdvhDVEdzI',
    'serverid': '6',
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
assert(response.status_code == 200)
with open('1.html', 'wb') as f: f.write(response.content)
response = response.content.decode('utf-8').split('\n')
line_token = list(filter(lambda x:('token' in x), response))
token = re.compile(r"token: [\"\'](?P<url>[-\w]+)[\"\']").search(line_token[0]).groupdict()['url']

newurl = f'https://cloud.tsinghua.edu.cn/api/v2.1/upload-links/{token}/upload/'
response = requests.get(newurl, cookies=cookies, headers=headers)
print('requesting', newurl, response.status_code)
assert(response.status_code == 200)
upload_link =  response.json()["upload_link"]

print('upload_link', upload_link)

def execute_command(fname):
    multipart_encoder = MultipartEncoder(
        fields={
            'file': (fname, open(fname, 'rb'), 'application/octet-stream'),
            'parent_dir': ('/', 'application/octet-stream')
        }
    )
    headers.update({'Content-Type': multipart_encoder.content_type})
    response = requests.post(upload_link, data=multipart_encoder, headers=headers, cookies=cookies)
    
    '''
    files = {
        'file': open(fname, 'rb'),
        'parent_dir': (None, '/'),
    }

    response = requests.post(upload_link, files=files, cookies=cookies, headers=headers)'''
    assert(response.status_code == 200)

    """Execute a single command and print its output in real time."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Print stdout in real time
    with tqdm(total=os.path.getsize(fname), unit='B', unit_scale=True, smoothing=0.01, desc=fname) as pbar:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and len(output.strip()) > 0:
                match = re.search(r'\s*([\.\d]+)%', output)
                if match:
                    progress = float(match.group(1))
                    pbar.update(progress * os.path.getsize(fname) / 100  - pbar.n)
                else: print('>>>', output.strip(), '<<<')
        
# List to store threads
threads = []

# Create and start a thread for each command
fnames = [os.path.join('D:\\', x) for x in os.listdir('D:')]
fnames.append(r'C:\Users\d\Desktop\sf10-490-out-4.bin')
fnames.append(r'C:\Users\d\Desktop\sf10-490-out-3.bin')
for fname in fnames: 
    if os.path.basename(fname)[:3] in ('sf8', 'sf9', 'sf1') and os.path.isfile(fname):
        print(fname)
    # fname = f'E:\\data2\\{i}.dat'
        thread = threading.Thread(target=execute_command, args=(fname,))
        thread.start()
        threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

# for i in range(2): upload_with_progress(upload_link, f'E:\\sf11\\sf11_{i}.bin')
# for i in range(3,4): upload_with_progress(f'https://cloud.tsinghua.edu.cn/seafhttp/upload-api/{rescode}', f'E:\\sf11\\sf11_{i}.bin')
