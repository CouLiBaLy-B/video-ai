"""Agentic video generation orchestration use case."""

from __future__ import annotations

from video_ai.domain.enums import JobStatus, VideoBackend
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
        runtime_profile: str = "ai=mock;video=mock",
    ) -> None:
        self._repository = repository
        self._workflow_planner = workflow_planner
        self._vision_analyzer = vision_analyzer
        self._prompt_enhancer = prompt_enhancer
        self._model_router = model_router
        self._video_generator = video_generator
        self._quality_reviewer = quality_reviewer
        self._runtime_profile = runtime_profile

    async def run(self, job: VideoGenerationJob) -> VideoGenerationJob:
        """Run planning and either complete generation or wait for approval."""
        current = await self._prepare(job)
        if current.pending_parameters is None:
            raise RuntimeError("Prepared job is missing generation parameters")
        if self._requires_approval(current.pending_parameters):
            return await self._transition(
                current,
                JobStatus.WAITING_FOR_APPROVAL,
                "Human approval required before GPU video generation",
            )
        return await self.run_approved(current)

    async def run_approved(self, job: VideoGenerationJob) -> VideoGenerationJob:
        """Continue a prepared job after human approval."""
        if job.pending_parameters is None:
            raise RuntimeError("Cannot approve a job without pending generation parameters")
        parameters = job.pending_parameters
        current = await self._transition(
            job, JobStatus.GENERATING, f"Generating with {parameters.model.id}"
        )
        video = await self._video_generator.generate(parameters)

        current = current.model_copy(update={"video": video, "pending_parameters": None})
        await self._repository.save(current)
        current = await self._transition(current, JobStatus.REVIEWING, "Reviewing generated video")
        report = await self._quality_reviewer.review(current.request, video, parameters)

        final_status = JobStatus.COMPLETED if report.accepted else JobStatus.FAILED
        current = current.model_copy(update={"quality_report": report})
        current = await self._transition(current, final_status, report.summary)
        return current

    async def reject(
        self, job: VideoGenerationJob, reason: str = "Human rejected generation"
    ) -> VideoGenerationJob:
        """Cancel a job that is waiting for human approval."""
        updated = job.model_copy(update={"pending_parameters": None})
        return await self._transition(updated, JobStatus.CANCELLED, reason)

    async def _prepare(self, job: VideoGenerationJob) -> VideoGenerationJob:
        """Plan, analyze and build generation parameters without generating video."""
        current = await self._transition(
            job,
            JobStatus.PLANNING,
            f"Creating agentic workflow plan ({self._runtime_profile})",
        )
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
        preferences = current.request.preferences
        parameters = GenerationParameters(
            prompt=prompt,
            image=current.request.image,
            model=model,
            width=preferences.width or model.default_width,
            height=preferences.height or model.default_height,
            num_frames=preferences.num_frames or 121,
            fps=preferences.fps or model.default_fps,
            seed=preferences.seed,
            guidance_scale=preferences.guidance_scale or 3.5,
            inference_steps=preferences.inference_steps or 30,
        )
        current = current.model_copy(update={"pending_parameters": parameters})
        await self._repository.save(current)
        return current

    @staticmethod
    def _requires_approval(parameters: GenerationParameters) -> bool:
        """Require approval for real GPU-oriented backends."""
        return parameters.model.backend != VideoBackend.MOCK

    async def _transition(
        self,
        job: VideoGenerationJob,
        status: JobStatus,
        reason: str,
    ) -> VideoGenerationJob:
        updated = job.transition(status, reason)
        await self._repository.save(updated)
        return updated
