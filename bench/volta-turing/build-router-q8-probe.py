#!/usr/bin/env python3
"""Build the isolated C02 fusion probe without modifying an engine build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('build', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--warps', type=int, choices=(4, 8), default=8)
p.add_argument('--selection', choices=('original', 'sorted', 'winner-exp'), default='original')
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
build = a.build.resolve()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
source = repo / 'src/kernels/cuda/native_router.cu'
code = source.read_text()

def body(name):
    start = code.index('{', code.index(name + '(')) + 1
    depth = 1
    end = start
    while depth:
        depth += (code[end] == '{') - (code[end] == '}')
        end += 1
    return code[start:end - 1]

# Reuse the current routing arithmetic verbatim, changing only its row index.
# This is a standalone experiment, not an alternate production router.
route = body('route_multi')
geometry = '    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;\n'
assert geometry in route
route = route.replace(geometry, '')
header = 'namespace router_probe {\n'
for name in ('warp_sum', 'warp_max'):
    header += '__device__ __forceinline__ float ' + name + '(float value) {' + body(name) + '}\n'
extra_sources = []
if a.selection == 'sorted':
    selection = repo / 'bench/volta-turing/router-sorted-selection.cuh'
    extra_sources.append(selection)
    header += selection.read_text() + '\n'
    start = route.index('    float selected = 0.0f, selected_sum = 0.0f;')
    assert route[start:].rstrip().endswith('if (lane < 10) weights[lane] = selected * inverse_selected_sum;')
    route = route[:start] + '    sorted_selection(values, ids, weights, lane);\n'
row_name = 'row_original' if a.selection == 'winner-exp' else 'row'
header += ('__device__ __forceinline__ void ' + row_name + '(const float* logits, int32_t* ids, float* weights, '
           'int n_tok, int tk) {' + route + '}\n')
if a.selection == 'winner-exp':
    selection = repo / 'bench/volta-turing/router-winner-exp.cuh'
    extra_sources.append(selection)
    header += selection.read_text() + '\n'
header += '}\n'
(out / 'router-row.cuh').write_text(header)
probe = repo / 'bench/volta-turing/router-q8-probe.cu'
(out / probe.name).write_bytes(probe.read_bytes())
(out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
original = shlex.split((build / 'CMakeFiles/strata.dir/link.txt').read_text())
index = next(i for i, value in enumerate(original) if value.endswith('/generate.cpp.o'))
assert original[index + 1:index + 3] == ['-o', 'strata']
production_api = 'native_router_top10_q8_multi(' in (repo / 'include/strata/kernels/native_router.hpp').read_text()
compile_cmd = ['nvcc', '-O3', '-std=c++20', '-arch=sm_75', '--use_fast_math',
               '-DROUTER_Q8_WARPS=' + str(a.warps),
               *(['-DROUTER_Q8_PRODUCTION_API=1'] if production_api else []),
               '-I' + str(repo / 'include'), '-I' + str(out), '-Xptxas=-v',
               '-c', str(out / probe.name), '-o', str(out / 'probe.o')]
link_cmd = [original[0], '-O3', str(out / 'probe.o'), '-o', str(out / 'probe'), *original[index + 3:]]
manifest = {'scope': 'standalone C02 CUDA probe; no engine integration',
            'fused_warps': a.warps,
            'selection': a.selection,
            'production_api_declared': production_api,
            'compile': compile_cmd, 'link': link_cmd, 'link_cwd': str(build),
            'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in (source, probe, Path(__file__), out / 'router-row.cuh', *extra_sources,
                                  repo / 'include/strata/kernels/native_router.hpp',
                                  repo / 'include/strata/kernels/iq_kernels.hpp',
                                  repo / 'include/strata/kernels/q8_1_finite.hpp',
                                  repo / 'src/kernels/cuda/iq_kernels.cu')},
            'reference_library_sha256': hashlib.sha256((build / 'libstrata_kernels.a').read_bytes()).hexdigest()}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(compile_cmd, check=True)
subprocess.run(link_cmd, cwd=build, check=True)
print(out / 'probe')
