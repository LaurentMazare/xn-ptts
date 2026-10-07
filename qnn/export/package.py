"""Export prefill and step, compile and link them into one QNN context binary, and profile each graph.

    python package.py [--device "Samsung Galaxy S25 (Family)"] [--resume]

The NPU variant: fp16 float inputs and outputs, fp32 inside, compiled with
default_graph_htp_precision=FLOAT16. The context binary holds both graphs and
stores the shared backbone and tables once. Job ids go to out/package.json; the
binary to out/phonon.bin, and each graph's DLC to out/<graph>.dlc.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch

from host import CACHE, PREFILL, int32, mimi_zero_states
from load import load

OUT = Path(__file__).parent / "out"
GRAPHS = ["prefill", "step"]
NAME = "phonon"


def example_inputs(ph, dtype=torch.float16):
    d = ph.dims
    kv = torch.zeros(2 * d.layers, d.heads, CACHE, d.head_dim, dtype=dtype)
    return {
        "prefill": (torch.zeros(1, PREFILL, dtype=torch.int32), kv, int32(0)),
        "step": (ph.bos_emb.to(dtype), torch.zeros(1, d.ldim, dtype=dtype), kv, int32(0), int32(0),
                 *mimi_zero_states(d, dtype)),
    }


def save(jobs):
    (OUT / "package.json").write_text(json.dumps(jobs, indent=2))


def check_dtypes(compile_jobs):
    """Every float input must stay fp16: AI Hub widens them to fp32 when a graph mixes the two."""
    ok = True
    for n, j in compile_jobs.items():
        specs = j.get_target_shapes()
        print(n, "compiled inputs", {k: (tuple(s), str(np.dtype(t))) for k, (s, t) in specs.items()}, flush=True)
        wide = [k for k, (_, t) in specs.items() if np.dtype(t) == np.float32]
        if wide:
            print(n, "WARNING: fp32 inputs", wide, flush=True)
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Samsung Galaxy S25 (Family)")
    ap.add_argument("--compile-options", default="--qnn_options default_graph_htp_precision=FLOAT16")
    ap.add_argument("--resume", action="store_true", help="finish the jobs in out/package.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = hub.Device(args.device)

    if args.resume:
        # Pick up jobs a previous run submitted, without exporting or uploading again.
        jobs = json.loads((OUT / "package.json").read_text())
        compile_jobs = {n: hub.get_job(jobs["compile"][n]) for n in GRAPHS}
        link_job = hub.get_job(jobs["link"])
        device = hub.Device(jobs["device"])
    else:
        ph = load(io_half=True)
        examples = example_inputs(ph)
        with torch.no_grad():
            programs = [torch.export.export(ph.graphs[name], examples[name]) for name in GRAPHS]
        jobs_list, link_job = hub.submit_compile_and_link_jobs(
            programs,
            device=device,
            name=NAME,
            graph_names=GRAPHS,
            compile_options=args.compile_options,
        )
        compile_jobs = dict(zip(GRAPHS, jobs_list))
        jobs = {"device": args.device, "compile": {n: j.job_id for n, j in compile_jobs.items()}}
        for n, j in compile_jobs.items():
            print(n, "compile", j.job_id, j.url, flush=True)
        print("link", link_job.job_id, link_job.url, flush=True)
        jobs["link"] = link_job.job_id
        save(jobs)

    status = link_job.wait()
    print("link", status.code, status.message or "", flush=True)
    if not status.success:
        for n, j in compile_jobs.items():
            s = j.get_status()
            print(n, "compile", s.code, s.message or "", flush=True)
        return
    check_dtypes(compile_jobs)
    # The per-graph DLCs too: QNN's CPU and GPU backends cannot load an HTP context binary.
    for n, j in compile_jobs.items():
        j.get_target_model().download(str(OUT / f"{n}.dlc"))
    model = link_job.get_target_model()
    jobs["model"] = model.model_id
    binary = OUT / f"{NAME}.bin"
    model.download(str(binary))
    jobs["binary_bytes"] = binary.stat().st_size
    print("context binary", binary, binary.stat().st_size / 2**20, "MiB", flush=True)
    save(jobs)

    profiles = {}
    for n in GRAPHS:
        if n in jobs.get("profile", {}):
            profiles[n] = hub.get_job(jobs["profile"][n])
            continue
        profiles[n] = hub.submit_profile_job(
            model=model, device=device, name=f"{NAME} {n}",
            options=f"--qnn_options context_enable_graphs={n}",
        )
        print(n, "profile", profiles[n].job_id, profiles[n].url, flush=True)
        jobs.setdefault("profile", {})[n] = profiles[n].job_id
        save(jobs)

    for n, p in profiles.items():
        status = p.wait()
        print(n, "profile", status.code, status.message or "", flush=True)
        if status.success:
            summary = p.download_profile()["execution_summary"]
            ms = summary["estimated_inference_time"] / 1000
            mem = summary.get("estimated_inference_peak_memory", 0) / 2**20
            print(f"{n}: {ms:.2f} ms, peak memory {mem:.0f} MiB", flush=True)
            jobs.setdefault("ms", {})[n] = ms
            jobs.setdefault("peak_mib", {})[n] = round(mem, 1)
            save(jobs)


if __name__ == "__main__":
    main()
