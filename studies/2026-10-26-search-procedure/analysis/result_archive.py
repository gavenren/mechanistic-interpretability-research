"""Store exact completed result files in small ZIP bundles for publication.

This tool does not run or change the scientific procedure. Run it only after
the scientific process has finished. The original result manifest stays public.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

TASKS = ('calibration/subsets', 'results/subsets', 'results/selection')
LIMIT = 24 * 1024**2  # Each ZIP must be strictly smaller than this limit.
GROUP_TARGET = 20 * 1024**2
CHUNK = 1024**2
DEFAULT_STUDY = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def member_path(name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Invalid member path')
    parts = name.split('/')
    if len(parts) < 3 or '/'.join(parts[:2]) not in TASKS:
        raise ValueError('Member is outside an allowed task directory')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(name).is_absolute():
        raise ValueError('Unsafe member path')
    return PurePosixPath(name)


def safe_target(study, relative, make_parents=False):
    """Reject symlinks in the destination and every existing parent."""
    path = study
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        if part in ('', '.', '..'):
            raise ValueError('Unsafe destination path')
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink destination is not allowed')
        if index < len(parts) - 1:
            if path.exists() and not path.is_dir():
                raise ValueError('Destination parent is not a directory')
            if make_parents:
                path.mkdir(exist_ok=True)
    if not path.resolve().is_relative_to(study.resolve()):
        raise ValueError('Destination leaves the study directory')
    return path


def result_record(study):
    expected, source_hashes = {}, []
    for area in ('calibration', 'results'):
        status_path = safe_target(study, area + '/status.json')
        if json.loads(status_path.read_text())['status'] != 'complete':
            raise RuntimeError('Packing requires both complete scientific phases')
        manifest_path = safe_target(study, area + '/manifest.json')
        source_hashes.append(digest(manifest_path))
        original = json.loads(manifest_path.read_text())
        if digest(status_path) != original['status.json']:
            raise ValueError('Direct status differs from its manifest')
        for name, sha in original.items():
            public_name = area + '/' + name
            if '/'.join(public_name.split('/')[:2]) in TASKS:
                member_path(public_name); expected[public_name] = sha
    if { '/'.join(name.split('/')[:2]) for name in expected } != set(TASKS):
        raise ValueError('The manifests do not cover all three result groups')
    combined = hashlib.sha256(':'.join(source_hashes).encode()).hexdigest()
    return expected, combined


def groups(paths):
    group, size = [], 0
    for path in paths:
        length = path.stat().st_size
        if group and size + length > GROUP_TARGET:
            yield group
            group, size = [], 0
        group.append(path)
        size += length
    if group:
        yield group


def pack(study):
    expected, source_hash = result_record(study)
    artifacts = safe_target(study, 'artifacts')
    if artifacts.exists() and (not artifacts.is_dir() or any(artifacts.iterdir())):
        raise FileExistsError('The artifacts directory must be absent or empty')
    task_files = {}
    seen = set()
    for task in TASKS:
        folder = safe_target(study, task)
        paths = []
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('Source symlinks are not allowed')
            if path.is_file():
                name = path.relative_to(study).as_posix()
                member_path(name)
                paths.append(path)
                seen.add(name)
            elif not path.is_dir():
                raise ValueError('A source entry is not a regular file or directory')
        task_files[task] = paths
    if seen != set(expected):
        raise ValueError('Task files differ from the original result manifest')
    receipt = dict(schema_version=1, study=study.name, result_status='complete',
                   source_manifests_sha256=source_hash, archive_size_limit_bytes=LIMIT,
                   preservation='All task file bytes are preserved; file times and permissions are not preserved.',
                   archives=[])
    published = []
    with tempfile.TemporaryDirectory(prefix='.result-archive-', dir=study) as temporary:
        staging = Path(temporary)
        for task in TASKS:
            for index, paths in enumerate(groups(task_files[task]), start=1):
                name = f"{task.replace('/', '-')}-{index:03d}.zip"
                archive = staging / name
                members = []
                with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=6, allowZip64=True) as bundle:
                    for path in paths:
                        relative = path.relative_to(study).as_posix()
                        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = (stat.S_IFREG | 0o644) << 16
                        h, size = hashlib.sha256(), 0
                        with path.open('rb') as source, bundle.open(info, 'w', force_zip64=True) as target:
                            for block in iter(lambda: source.read(CHUNK), b''):
                                target.write(block)
                                h.update(block)
                                size += len(block)
                        if h.hexdigest() != expected[relative]:
                            raise ValueError('A source file differs from the original result manifest: ' + relative)
                        members.append(dict(path=relative, sha256=h.hexdigest(), size=size))
                size = archive.stat().st_size
                if size >= LIMIT:
                    raise ValueError('A bundle is too large; no files were published: ' + name)
                receipt['archives'].append(dict(path='artifacts/' + name, task=task,
                                                sha256=digest(archive), size=size, members=members))
        (staging / 'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        if result_record(study)[1] != source_hash:
            raise ValueError('The source result manifest changed during packing')
        artifacts.mkdir(exist_ok=True)
        try:
            # Hard links create names exclusively. They cannot replace a file.
            # The manifest is installed last, after all bundles are present.
            for source in sorted(staging.glob('*.zip')) + [staging / 'manifest.json']:
                target = artifacts / source.name
                os.link(source, target)
                published.append(target)
            return verify(study)
        except BaseException:
            for target in reversed(published):
                target.unlink()
            raise


def archive_record(study):
    expected, source_hash = result_record(study)
    receipt = json.loads(safe_target(study, 'artifacts/manifest.json').read_text())
    if receipt.get('schema_version') != 1 or receipt.get('source_manifests_sha256') != source_hash:
        raise ValueError('Archive manifest does not match the original result manifest')
    names, archives = set(), set()
    for item in receipt['archives']:
        if not re.fullmatch(r'artifacts/(calibration-subsets|results-subsets|results-selection)-[0-9]{3,}\.zip', item['path']):
            raise ValueError('Unsafe archive path')
        if item['path'] in archives or item['task'] not in TASKS:
            raise ValueError('Duplicate archive or invalid task')
        archives.add(item['path'])
        for member in item['members']:
            name = member['path']
            member_path(name)
            if name in names or '/'.join(name.split('/')[:2]) != item['task']:
                raise ValueError('Duplicate member or wrong task')
            if member['sha256'] != expected.get(name) or type(member['size']) is not int or member['size'] < 0:
                raise ValueError('Member does not match the original result manifest')
            names.add(name)
    if names != set(expected):
        raise ValueError('Archive manifest omits or adds task files')
    return receipt


def verified_members(study, item):
    archive = safe_target(study, item['path'])
    size = archive.stat().st_size
    if size >= LIMIT or size != item['size'] or digest(archive) != item['sha256']:
        raise ValueError('Archive size or hash differs: ' + item['path'])
    expected = {member['path']: member for member in item['members']}
    with zipfile.ZipFile(archive, 'r') as bundle:
        seen = set()
        for info in bundle.infolist():
            name = info.filename
            member_path(name)
            mode = stat.S_IFMT(info.external_attr >> 16)
            if name in seen or name not in expected or info.is_dir() or mode not in (0, stat.S_IFREG):
                raise ValueError('Duplicate, unexpected, or non-file ZIP member')
            if info.flag_bits & 1 or info.file_size != expected[name]['size']:
                raise ValueError('Encrypted member or incorrect member size')
            seen.add(name)
            h, length = hashlib.sha256(), 0
            with bundle.open(info) as source:
                for block in iter(lambda: source.read(CHUNK), b''):
                    h.update(block)
                    length += len(block)
            if length != expected[name]['size'] or h.hexdigest() != expected[name]['sha256']:
                raise ValueError('Member size or hash differs: ' + name)
        if seen != set(expected):
            raise ValueError('A ZIP is missing expected members')


def verify(study):
    receipt = archive_record(study)
    for item in receipt['archives']:
        verified_members(study, item)
    return dict(status='verified', archives=len(receipt['archives']),
                files=sum(len(item['members']) for item in receipt['archives']))


def unpack(study):
    result = verify(study)  # Verify every bundle before any extraction.
    receipt = archive_record(study)
    for item in receipt['archives']:
        for member in item['members']:
            target = safe_target(study, member['path'])
            if target.exists() and (not target.is_file() or digest(target) != member['sha256']):
                raise FileExistsError('A conflicting result file exists: ' + member['path'])
    added = 0
    for item in receipt['archives']:
        with zipfile.ZipFile(safe_target(study, item['path']), 'r') as bundle:
            for member in item['members']:
                target = safe_target(study, member['path'], make_parents=True)
                if target.exists():
                    if not target.is_file() or digest(target) != member['sha256']:
                        raise FileExistsError('A conflicting result file exists: ' + member['path'])
                    continue
                fd, temporary = tempfile.mkstemp(prefix='.restore-', dir=target.parent)
                try:
                    h = hashlib.sha256()
                    with os.fdopen(fd, 'wb') as output, bundle.open(member['path']) as source:
                        for block in iter(lambda: source.read(CHUNK), b''):
                            output.write(block)
                            h.update(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if h.hexdigest() != member['sha256']:
                        raise ValueError('Member changed during extraction')
                    os.link(temporary, target)  # Exclusive creation; never overwrite.
                    added += 1
                finally:
                    Path(temporary).unlink(missing_ok=True)
    return dict(result, status='restored', files_added=added)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pack', 'verify', 'unpack'))
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    arguments = parser.parse_args()
    root = arguments.study.resolve()
    result = {'pack': pack, 'verify': verify, 'unpack': unpack}[arguments.action](root)
    print(json.dumps(result, sort_keys=True))
