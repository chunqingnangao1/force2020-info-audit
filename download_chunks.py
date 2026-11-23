# -*- coding: utf-8 -*-
"""并行分段下载 Zenodo 大文件：8线程 x 5MB块，逐块校验重试，最后拼装"""
import sys, io, os, urllib.request, concurrent.futures, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

URL = ('https://zenodo.org/records/4351156/files/'
       'LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip?download=1')
TOTAL = 169785172
CHUNK = 5 * 1024 * 1024
THREADS = 8
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
PARTS = os.path.join(DATA, 'parts')
OUT = os.path.join(DATA, 'LAS_files_Force_2020.zip')
os.makedirs(PARTS, exist_ok=True)

N = (TOTAL + CHUNK - 1) // CHUNK
print('total %d bytes -> %d chunks' % (TOTAL, N))

def fetch_range(args):
    idx, lo, hi = args
    path = os.path.join(PARTS, 'part_%04d' % idx)
    want = hi - lo + 1
    for attempt in range(6):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == want:
            return idx
        req = urllib.request.Request(URL)
        req.add_header('Range', 'bytes=%d-%d' % (lo + have, hi))
        req.add_header('User-Agent', 'Mozilla/5.0 chunkdl')
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(path, 'ab') as f:
                while True:
                    buf = r.read(1 << 18)
                    if not buf:
                        break
                    f.write(buf)
        except Exception:
            time.sleep(2 + attempt)
    raise RuntimeError('chunk %d failed' % idx)

jobs = [(i, i * CHUNK, min((i + 1) * CHUNK, TOTAL) - 1) for i in range(N)]
t0 = time.time()
done = 0
with concurrent.futures.ThreadPoolExecutor(THREADS) as ex:
    futs = {ex.submit(fetch_range, j): j[0] for j in jobs}
    for fut in concurrent.futures.as_completed(futs):
        fut.result()
        done += 1
        if done % 8 == 0 or done == N:
            print('chunks %d/%d %.0fs' % (done, N, time.time() - t0))

with open(OUT, 'wb') as out:
    for i in range(N):
        with open(os.path.join(PARTS, 'part_%04d' % i), 'rb') as f:
            out.write(f.read())
size = os.path.getsize(OUT)
print('assembled:', size, 'OK' if size == TOTAL else 'MISMATCH!')
import zipfile
zf = zipfile.ZipFile(OUT)
print('zip entries:', len(zf.namelist()), '| integrity:', 'PASS' if zf.testzip() is None else 'FAIL')
