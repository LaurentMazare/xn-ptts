"""Compile the graphs as fp32 DLCs for QNN's CPU backend.

    python package_cpu.py [prefill,step]

The HTP graphs take and return fp16 and cast inside; the CPU backend cannot run
those casts, and it crashes on Mimi's transposed convolutions. These are the
same graphs with fp32 I/O, no fp16 at all, and the transposed convs as an exact
overlap-add (load(io_half=False)), compiled separately: the CPU backend loads
DLCs, not a linked context binary. They land in out/cpu/<graph>.dlc, job ids in
out/cpu/package.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch

from load import load
from package import example_inputs

OUT = Path(__file__).parent / "out" / "cpu"
GRAPHS = ["prefill", "step"]
DEVICE = "Samsung Galaxy S25 (Family)"  # any device with a CPU: a DLC is not tied to an SoC


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else GRAPHS
    ph = load(io_half=False)
    examples = example_inputs(ph, torch.float32)
    device = hub.Device(DEVICE)
    saved = OUT / "package.json"
    jobs = json.loads(saved.read_text()) if saved.exists() else {}
    with torch.no_grad():
        programs = {name: torch.export.export(ph.graphs[name], examples[name]) for name in names}
    for name in names:
        job = hub.submit_compile_job(programs[name], device=device, name=f"phonon cpu {name}",
                                     options="--target_runtime qnn_dlc")
        jobs[name] = job.job_id
        print(name, "compile", job.job_id, job.url, flush=True)
        saved.write_text(json.dumps(jobs, indent=2))
    for name in names:
        job = hub.get_job(jobs[name])
        status = job.wait()
        print(name, "compile", status.code, status.message or "", flush=True)
        if status.success:
            specs = job.get_target_shapes()
            print(name, "compiled inputs", {k: (tuple(s), str(np.dtype(t))) for k, (s, t) in specs.items()}, flush=True)
            job.get_target_model().download(str(OUT / f"{name}.dlc"))
            print(name, "->", OUT / f"{name}.dlc", (OUT / f"{name}.dlc").stat().st_size / 2**20, "MiB", flush=True)


if __name__ == "__main__":
    main()
