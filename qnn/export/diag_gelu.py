"""The exact (erf) GELU on the HTP in isolation, against other formulations.

    python diag_gelu.py

One tiny graph, fp16 in and out, compiled with FLOAT16 precision and run on the
device. Each output is a GELU formulation of the same input; all are compared
with the fp32 exact GELU on the Mac, next to what fp16 rounding alone costs.
"""
import math

import numpy as np
import qai_hub as hub
import torch
from torch import nn
from torch.nn import functional as F

from graphs import gelu_variants

N = 4096
torch.manual_seed(0)
x = torch.cat([torch.linspace(-8, 8, N // 2), torch.randn(N // 2) * 2])[None, None]


class Diag(nn.Module):
    def forward(self, x):
        xf = x.float()
        return tuple(f(xf).half() for f in gelu_variants().values())


names = list(gelu_variants())
xh = x.half()
ep = torch.export.export(Diag().eval(), (xh,))
want = F.gelu(xh.float()).numpy().reshape(-1)


def report(tag, outs):
    for name, got in zip(names, outs):
        got = np.asarray(got, dtype=np.float32).reshape(-1)
        err = np.abs(got - want)
        print(f"{tag:6s} {name:10s} max abs err {err.max():.2e}  rel err {np.linalg.norm(got - want) / np.linalg.norm(want):.2e}", flush=True)


with torch.no_grad():
    report("mac", [t.float().numpy() for t in Diag()(xh)])

device = hub.Device("Samsung Galaxy S25 (Family)")
cj = hub.submit_compile_job(ep, device=device, name="phonon diag gelu",
                            options="--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16")
st = cj.wait()
print("compile", cj.job_id, st.code, st.message or "", flush=True)
if not st.success:
    raise SystemExit(1)
specs = cj.get_target_shapes()
print("inputs", specs, flush=True)
(k, (_, dtype)), = specs.items()
ij = hub.submit_inference_job(cj.get_target_model(), device=device, inputs={k: [xh.numpy().astype(dtype)]},
                              name="phonon diag gelu")
st = ij.wait()
print("inference", ij.job_id, st.code, st.message or "", flush=True)
out = ij.download_output_data()
keys = sorted(out, key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
print("output keys", keys, flush=True)
report("device", [out[k][0] for k in keys])
