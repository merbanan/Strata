#!/usr/bin/env python3
"""Build the local CUDA snapshot diagnostic from an existing Unix Makefiles build."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('build', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--record', type=Path, help='save exact command and build directory')
a = p.parse_args()
build = a.build.resolve()
output = a.output.resolve()
repo = Path(__file__).resolve().parents[2]
link = build / 'CMakeFiles/strata.dir/link.txt'
if not link.is_file():
    p.error('requires a completed CUDA engine build using CMake Unix Makefiles')
original = shlex.split(link.read_text())
index = next((i for i, v in enumerate(original) if v.endswith('/generate.cpp.o')), None)
if index is None or original[index + 1:index + 3] != ['-o', 'strata']:
    p.error('unrecognized engine link command; inspect link.txt before adapting')
command = [original[0], '-O3', '-std=c++20', '-fno-access-control',
           '-I' + str(repo / 'include'), str(repo / 'bench/volta-turing/mtp-snapshot-parity.cpp'),
           '-o', str(output), '-Wl,--wrap=cudaMemcpy', '-Wl,--wrap=cudaMallocHost',
           '-Wl,--wrap=cudaStreamSynchronize', *original[index + 3:]]
if a.record:
    a.record.write_text(json.dumps({'cwd': str(build), 'command': command}, indent=2) + '\n')
subprocess.run(command, cwd=build, check=True)
print(output)
