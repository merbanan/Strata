#!/usr/bin/env python3
"""Record host pressure and GPU conditions for the lifetime of a benchmark PID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('pid', type=int)
p.add_argument('out', type=Path)
a = p.parse_args()
stat = Path(f'/proc/{a.pid}/stat')

def identity():
    try:
        # Skip the parenthesized comm, which can contain spaces.
        fields = stat.read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None

started = identity()
with a.out.open('x') as output:
    while started is not None and identity() == started:
        row = dict(time=time.time(), load=os.getloadavg())
        for name, path in [('memory', '/proc/meminfo'),
                           ('memory_pressure', '/proc/pressure/memory'),
                           ('cpu_pressure', '/proc/pressure/cpu')]:
            row[name] = Path(path).read_text()
        try:
            row['gpu'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        except (subprocess.SubprocessError, OSError) as error:
            row['gpu_error'] = str(error)
        output.write(json.dumps(row) + '\n'); output.flush()
        time.sleep(2)
