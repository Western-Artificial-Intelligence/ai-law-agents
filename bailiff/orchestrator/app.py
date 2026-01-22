import asyncio
import uuid
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager

from bailiff.orchestrator.schemas import PilotTrialRequest, Job, JobStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None

    async def submit_job(self, request: PilotTrialRequest) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            submitted_at=datetime.now(),
            request=request
        )
        self.jobs[job_id] = job
        await self.queue.put(job_id)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        return list(self.jobs.values())

    async def start_worker(self):
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Job worker started")

    async def stop_worker(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Job worker stopped")

    async def _worker(self):
        while True:
            job_id = await self.queue.get()
            job = self.jobs[job_id]
            
            try:
                await self._run_job(job)
            except Exception as e:
                logger.error(f"Error running job {job_id}: {e}")
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
            finally:
                self.queue.task_done()

    async def _run_job(self, job: Job):
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        # Construct command line arguments
        cmd = [sys.executable, "scripts/run_pilot_trial.py"]
        
        req = job.request
        if req.case:
            cmd.extend(["--case", req.case])
        if req.config:
            cmd.extend(["--config", req.config])
        
        cmd.extend(["--seed", str(req.seed)])
        cmd.extend(["--backend", req.backend.value])
        
        if req.model:
            cmd.extend(["--model", req.model])
        if req.out:
            cmd.extend(["--out", req.out])
        if req.manifest:
            cmd.extend(["--manifest", req.manifest])
            
        for placebo in req.placebos:
            cmd.extend(["--placebo", placebo])
            
        cmd.extend(["--timeout-seconds", str(req.timeout_seconds)])
        cmd.extend(["--max-retries", str(req.max_retries)])
        cmd.extend(["--backoff-seconds", str(req.backoff_seconds)])
        cmd.extend(["--backoff-multiplier", str(req.backoff_multiplier)])
        cmd.extend(["--rate-limit-seconds", str(req.rate_limit_seconds)])
        
        for k, v in req.backend_params.items():
            cmd.extend(["--backend-param", f"{k}={v}"])

        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.cwd() # Ensure we run from the project root
        )

        stdout, stderr = await process.communicate()
        
        if stdout:
            job.logs.append(stdout.decode())
        if stderr:
            job.logs.append(stderr.decode())

        job.completed_at = datetime.now()
        if process.returncode == 0:
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.FAILED
            job.error = f"Process exited with code {process.returncode}"

job_manager = JobManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_manager.start_worker()
    yield
    await job_manager.stop_worker()

app = FastAPI(title="Bailiff Orchestrator", lifespan=lifespan)

@app.post("/jobs", response_model=Job)
async def submit_job(request: PilotTrialRequest):
    return await job_manager.submit_job(request)

@app.get("/jobs", response_model=List[Job])
async def list_jobs():
    return job_manager.list_jobs()

@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
def health():
    return {"status": "ok"}
    