---
name: gpu-cost-optimization
description: Use for reducing GPU memory and runtime by adapting resolution, frames, precision, quantization and model choice.
---

# GPU Cost Optimization

Prefer the cheapest configuration that satisfies the user's quality target.

Levers:

- reduce resolution for drafts
- reduce frames for previews
- use distilled or fp8 variants when available
- select LTX for fast iteration
- reserve Wan I2V for final quality passes
