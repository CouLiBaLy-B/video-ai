---
name: safety-and-policy
description: Use for validating user inputs and generated outputs against product safety constraints before generation or delivery.
---

# Safety and Policy

Reject or require review for requests involving illegal, harmful or disallowed content.

Protect infrastructure:

- do not fetch arbitrary internal URLs
- validate MIME types and file sizes
- strip unsafe metadata when possible
- never expose secrets in logs, prompts or artifacts
