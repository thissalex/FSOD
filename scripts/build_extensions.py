#!/usr/bin/env python3
"""Build native upstream extensions inside the active fsod environment only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component', choices=['sam2', 'upn'], required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prefix = Path(sys.prefix)
    if prefix.name != 'fsod':
        raise RuntimeError('Run using the fsod environment Python')
    nvcc = prefix/'bin/nvcc'
    if not nvcc.is_file():raise RuntimeError('CUDA 11.8 compiler not installed in fsod')
    nvcc_version = subprocess.check_output([str(nvcc), '--version'], text=True)
    if 'release 11.8' not in nvcc_version:raise RuntimeError('Expected CUDA 11.8 nvcc')
    env = os.environ.copy()
    env.update(CUDA_HOME=str(prefix), MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='8.9',
               SAM2_BUILD_CUDA='1', SAM2_BUILD_ALLOW_ERRORS='0')
    env['PATH'] = str(prefix/'bin') + os.pathsep + env.get('PATH', '')
    nvidia = prefix/'lib/python3.10/site-packages/nvidia'
    includes = [str(p) for p in nvidia.glob('*/include')]
    env['CPATH'] = os.pathsep.join(includes + [str(prefix/'include'), env.get('CPATH','')])
    env['LIBRARY_PATH'] = str(prefix/'lib') + os.pathsep + env.get('LIBRARY_PATH','')
    target = root/('external/sam2' if args.component == 'sam2' else 'external/FSOD-VFM/chatrex/upn/ops')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                    '-v', '-e', str(target)], env=env, check=True)


if __name__ == '__main__':main()
