from video_ai.workers import jobs


def test_worker_entrypoints_are_importable() -> None:
    assert callable(jobs.run_job_task)
    assert callable(jobs.run_approved_job_task)
