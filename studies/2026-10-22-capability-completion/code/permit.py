"""Write a permit atomically. The caller must first read live account usage."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def replace_permit(path, allowed, lifetime=120):
    if not 0 < lifetime <= 120:
        raise ValueError('Permit lifetime must be at most 120 seconds')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'allowed': bool(allowed), 'expires_at': time.time() + lifetime}
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(document, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--allowed', action='store_true')
    parser.add_argument('--lifetime', type=int, default=120)
    args = parser.parse_args()
    replace_permit(args.path, args.allowed, args.lifetime)
