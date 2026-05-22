"""Workflow planners for video generation."""

from __future__ import annotations

import json
from typing import Any

from video_ai.agents.deepagents_factory import (
    DeepAgentConfig,
    DeepAgentsUnavailableError,
    create_video_deep_agent,
)
from video_ai.agents.tools import VideoAgentToolbelt, build_deepagents_tools
from video_ai.domain.models import AgentPlan, AgentPlanStep, VideoGenerationJob


class SimpleWorkflowPlanner:
    """Deterministic planner used by default and in tests."""

    async def plan(self, job: VideoGenerationJob) -> AgentPlan:
        """Return the standard video generation plan."""
        _ = job
        return AgentPlan(
            summary=(
                "Standard image-to-video workflow using analysis, prompt enhancement, "
                "routing, generation and review."
            ),
            steps=[
                AgentPlanStep(
                    name="analyze-image",
                    description=(
                        "Understand the uploaded image and extract preservation "
                        "constraints."
                    ),
                    agent="vision-analysis-agent",
                ),
                AgentPlanStep(
                    name="enhance-prompt",
                    description="Create a cinematic video prompt and negative prompt.",
                    agent="cinematic-prompt-agent",
                ),
                AgentPlanStep(
                    name="route-model",
                    description="Select the best configured video backend and parameters.",
                    agent="model-routing-agent",
                ),
                AgentPlanStep(
                    name="generate-video",
                    description="Run the selected image-to-video generator.",
                    agent="generation-supervisor-agent",
                ),
                AgentPlanStep(
                    name="review-quality",
                    description="Review temporal quality and prompt adherence.",
                    agent="quality-review-agent",
                ),
            ],
        )


class DeepAgentsWorkflowPlanner:
    """Planner backed by LangChain DeepAgents.

    The planner only creates a plan. Execution remains controlled by our typed
    application service and ports, preserving SOLID boundaries and testability.
    """

    def __init__(self, config: DeepAgentConfig, toolbelt: VideoAgentToolbelt | None = None) -> None:
        self._config = config
        self._toolbelt = toolbelt

    async def plan(self, job: VideoGenerationJob) -> AgentPlan:
        """Ask DeepAgents to produce a structured workflow plan."""
        tools = build_deepagents_tools(self._toolbelt, job) if self._toolbelt else []
        agent = create_video_deep_agent(self._config, tools=tools)
        prompt = (
            "Create a JSON plan for this text+image to video request. "
            "Use available tools when useful to inspect image context, route model, "
            "and validate parameters. "
            "Return keys: summary, requires_human_approval, steps. "
            "Each step has name, description, agent.\n"
            f"Prompt: {job.request.prompt}\n"
            f"Image path: {job.request.image.path}\n"
            f"Image mime type: {job.request.image.mime_type}"
        )
        result = await _invoke_agent(agent, prompt)
        return _parse_plan(result)


async def _invoke_agent(agent: Any, prompt: str) -> str:
    payload = {"messages": [{"role": "user", "content": prompt}]}
    if hasattr(agent, "ainvoke"):
        response = await agent.ainvoke(payload)
    else:  # pragma: no cover - depends on optional DeepAgents runtime
        response = agent.invoke(payload)
    return _extract_content(response)


def _extract_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(getattr(last, "content", ""))
    return str(response)


def _parse_plan(content: str) -> AgentPlan:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DeepAgentsUnavailableError("DeepAgents planner did not return valid JSON") from exc
    steps = [
        AgentPlanStep(
            name=str(step.get("name", "step")),
            description=str(step.get("description", "")),
            agent=str(step.get("agent", "general-purpose")),
        )
        for step in data.get("steps", [])
        if isinstance(step, dict)
    ]
    return AgentPlan(
        summary=str(data.get("summary", "DeepAgents workflow plan")),
        steps=steps,
        requires_human_approval=bool(data.get("requires_human_approval", False)),
    )
