"""Agentic video generation orchestration use case."""

from __future__ import annotations

from video_ai.domain.enums import JobStatus
from video_ai.domain.models import GenerationParameters, VideoGenerationJob
from video_ai.domain.ports import (
    JobRepository,
    PromptEnhancer,
    VideoGenerator,
    VideoModelRouter,
    VideoQualityReviewer,
    VisionAnalyzer,
    WorkflowPlanner,
)


class VideoGenerationOrchestrator:
    """Coordinate the complete text+image to video workflow.

    This class is deliberately framework-independent. DeepAgents-specific code can
    plug into the same ports, while tests can use deterministic adapters.
    """

    def __init__(
        self,
        *,
        repository: JobRepository,
        workflow_planner: WorkflowPlanner,
        vision_analyzer: VisionAnalyzer,
        prompt_enhancer: PromptEnhancer,
        model_router: VideoModelRouter,
        video_generator: VideoGenerator,
        quality_reviewer: VideoQualityReviewer,
    ) -> None:
        self._repository = repository
        self._workflow_planner = workflow_planner
        self._vision_analyzer = vision_analyzer
        self._prompt_enhancer = prompt_enhancer
        self._model_router = model_router
        self._video_generator = video_generator
        self._quality_reviewer = quality_reviewer

    async def run(self, job: VideoGenerationJob) -> VideoGenerationJob:
        """Run planning, analysis, generation and review for a job."""
        current = await self._transition(job, JobStatus.PLANNING, "Creating agentic workflow plan")
        plan = await self._workflow_planner.plan(current)
        current = current.model_copy(update={"plan": plan})
        await self._repository.save(current)

        current = await self._transition(current, JobStatus.ANALYZING, "Analyzing input image")
        analysis = await self._vision_analyzer.analyze(
            current.request.image, current.request.prompt
        )

        current = await self._transition(
            current, JobStatus.PLANNING, "Enhancing prompt and routing model"
        )
        prompt = await self._prompt_enhancer.enhance(current.request.prompt, analysis)
        model = await self._model_router.select_model(current.request)
        parameters = GenerationParameters(
            prompt=prompt,
            image=current.request.image,
            model=model,
            width=model.default_width,
            height=model.default_height,
            fps=model.default_fps,
        )

        current = await self._transition(
            current, JobStatus.GENERATING, f"Generating with {model.id}"
        )
        video = await self._video_generator.generate(parameters)

        current = current.model_copy(update={"video": video})
        await self._repository.save(current)
        current = await self._transition(current, JobStatus.REVIEWING, "Reviewing generated video")
        report = await self._quality_reviewer.review(current.request, video, parameters)

        final_status = JobStatus.COMPLETED if report.accepted else JobStatus.FAILED
        current = current.model_copy(update={"quality_report": report})
        current = await self._transition(current, final_status, report.summary)
        return current

    async def _transition(
        self,
        job: VideoGenerationJob,
        status: JobStatus,
        reason: str,
    ) -> VideoGenerationJob:
        updated = job.transition(status, reason)
        await self._repository.save(updated)
        return updated
