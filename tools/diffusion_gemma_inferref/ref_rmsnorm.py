"""Reference RMSNorm (with and without learned weight) for InferRef traces."""

import torch
from torch.nn import functional as F

torch.manual_seed(0)

x = torch.randn(4, 8, dtype=torch.float32)
w = torch.randn(8, dtype=torch.float32)

y_weighted = F.rms_norm(x, (8,), w, 1e-6)
y_unweighted = F.rms_norm(x, (8,), None, 1e-6)

print("weighted_sum", float(y_weighted.sum()))
print("unweighted_sum", float(y_unweighted.sum()))
