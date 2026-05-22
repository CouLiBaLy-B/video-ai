---
name: cinematic-prompting
description: Use for transforming short user prompts into rich image-to-video prompts with camera motion, lighting, style and negative prompts.
---

# Cinematic Prompting

## Instructions

1. Preserve the user's subject and the identity constraints from the input image.
2. Add camera motion only when it is physically plausible.
3. Prefer concise but descriptive prompts for video diffusion models.
4. Include lighting, composition and motion cues.
5. Always produce a negative prompt addressing blur, flicker, warping and identity drift.

## Output fields

- `positive`
- `negative`
- `camera_motion`
- `style`
