#!/usr/bin/env python3
"""Fetch the two exact UCI files without changing a different local file."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL='https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip'
HASHES={
 'optdigits.tra':'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
 'optdigits.tes':'6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    archive=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL,timeout=60).read()))
    for name,want in HASHES.items():
        blob=archive.read(name)
        if hashlib.sha256(blob).hexdigest()!=want: raise ValueError('Source hash differs: '+name)
        target=out/name
        if target.exists() and target.read_bytes()!=blob: raise FileExistsError(target)
        target.write_bytes(blob)
        print(name,want)
if __name__=='__main__': main()
