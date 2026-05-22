---
name: image-to-video-routing
description: Use for selecting LTX-Video, Wan I2V or fallback mock generation according to quality, speed and VRAM constraints.
---

# Image-to-Video Routing

## Model policy

- Use `mock` for tests and non-GPU environments.
- Use `ltx-video` for fast MVP generation and iteration.
- Use `wan-i2v` for high-quality cinematic output when high VRAM is available.

## Approval policy

Ask for human approval before jobs expected to consume expensive GPU resources or run longer than the configured threshold.
