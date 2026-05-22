"""DeepAgents integration factory.

The production orchestrator will use LangChain DeepAgents as the high-level
planner/supervisor. This module isolates the optional dependency so the rest of
our application remains testable without GPU or agent packages installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeepAgentConfig:
    """Configuration for the DeepAgents video supervisor."""

    model: str
    skills_path: str = "skills"
    system_prompt: str = (
        "You are an agentic AI film director for text+image-to-video workflows. "
        "Use write_todos for complex requests. Delegate with the task tool to "
        "specialized subagents. Use available skills before specialized work. "
        "Use provided tools for image analysis, prompt enhancement, model routing "
        "and parameter validation. Preserve image identity, control GPU cost, and "
        "request human approval before expensive GPU work."
    )


class DeepAgentsUnavailableError(RuntimeError):
    """Raised when DeepAgents is required but not installed."""


def create_video_deep_agent(config: DeepAgentConfig, tools: list[Any]) -> Any:
    """Create a DeepAgents supervisor when the optional dependency is available."""
    try:
        from deepagents import create_deep_agent  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise DeepAgentsUnavailableError(
            "Install the 'agents' extra to enable DeepAgents orchestration."
        ) from exc

    subagents = [
        {
            "name": "vision-analysis-agent",
            "description": "Analyze input images for image-to-video generation.",
            "system_prompt": "Return concise visual analysis and identity constraints.",
        },
        {
            "name": "cinematic-prompt-agent",
            "description": "Create cinematic prompts and negative prompts for video models.",
            "system_prompt": "Write model-ready prompts with camera motion and style.",
        },
        {
            "name": "model-routing-agent",
            "description": "Select the best open-source video model and parameters.",
            "system_prompt": "Balance quality, speed, VRAM and user intent.",
        },
        {
            "name": "generation-supervisor-agent",
            "description": "Prepare generation parameters and supervise worker execution.",
            "system_prompt": (
                "Validate parameters, respect approval gates and avoid duplicate jobs."
            ),
        },
        {
            "name": "quality-review-agent",
            "description": "Review generated videos for quality and prompt adherence.",
            "system_prompt": "Identify artifacts, drift and iteration suggestions.",
        },
        {
            "name": "safety-agent",
            "description": "Review prompts and generation requests against product safety policy.",
            "system_prompt": (
                "Identify blocked content, review-required content and safe alternatives."
            ),
        },
    ]
    return create_deep_agent(
        model=config.model,
        tools=tools,
        system_prompt=config.system_prompt,
        subagents=subagents,
        skills=[config.skills_path],
    )
