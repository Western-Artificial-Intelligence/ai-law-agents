import unittest
from fastapi.testclient import TestClient
from bailiff.orchestrator.app import app, job_manager
from bailiff.orchestrator.schemas import JobStatus, PilotTrialRequest, Backend
from unittest.mock import AsyncMock, patch
import asyncio
import sys

client = TestClient(app)

class TestOrchestratorAPI(unittest.TestCase):
    def test_submit_job(self):
        response = client.post("/jobs", json={
            "backend": "echo",
            "seed": 123
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["request"]["backend"], "echo")
        self.assertEqual(data["request"]["seed"], 123)
        return data["id"]

    def test_list_jobs(self):
        response = client.get("/jobs")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_get_job(self):
        # First submit a job
        job_id = self.test_submit_job()
        
        response = client.get(f"/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], job_id)

class TestAsyncJobExecution(unittest.IsolatedAsyncioTestCase):
    async def test_job_execution_flow(self):
        # Mock subprocess to avoid actual execution
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"stdout", b"stderr")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            # Start the worker manually for the test
            await job_manager.start_worker()
            
            try:
                # Submit a job via the manager directly to await it easily in test
                req = PilotTrialRequest(backend=Backend.ECHO, seed=999)
                job = await job_manager.submit_job(req)
                
                # Wait for job to complete (worker is running in background)
                # Since we mocked subprocess, it should be fast.
                # We poll a few times.
                for _ in range(10):
                    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)
                    
                self.assertEqual(job.status, JobStatus.COMPLETED)
                self.assertIn("stdout", job.logs[0])
                
            finally:
                await job_manager.stop_worker()

if __name__ == "__main__":
    unittest.main()
