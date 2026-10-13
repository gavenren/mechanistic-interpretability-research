#!/usr/bin/env python3
"""Fetch the six pinned public files. No account or inference API is needed."""
import argparse
import ast
import hashlib
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
source = ast.parse(Path(__file__).with_name('run.py').read_text())
values = {}
for node in source.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {'MODEL_ID', 'REVISION', 'MODEL_HASHES'}:
                values[target.id] = ast.literal_eval(node.value)
args.output.mkdir(parents=True, exist_ok=True)
for name, expected in values['MODEL_HASHES'].items():
    path = args.output / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        url = f"https://huggingface.co/{values['MODEL_ID']}/resolve/{values['REVISION']}/{name}"
        request = urllib.request.Request(url, headers={'User-Agent': 'Pythia-patch-memory-research/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Downloaded file does not match pinned hash: ' + name)
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_bytes(data)
        temporary.replace(path)
    print(name, expected)
