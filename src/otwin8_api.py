from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import subprocess
import os
import json
import asyncio
import uuid
import hashlib
from datetime import datetime
import tempfile
import logging
import sys
import pandas as pd
from dotenv import load_dotenv
import shutil

load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OncoTwin Simplified API", version="1.0.0")

job_status_memory = {}
cache_memory = {}

# --- Pydantic Models ---

# Represents a single request within a batch
class SingleRequest(BaseModel):
    doctor_id: int
    patient_ids: List[str]

# The main request body for the batch endpoint
class BatchPipelineRequest(BaseModel):
    requests: List[SingleRequest]
    refresh: bool = False

# The simplified status response sent to the user
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str
    completed_at: Optional[str] = None
    doctor_ids_total: List[int]
    doctor_ids_success: List[int] = []
    doctor_ids_failed: List[int] = []

# Internal model for storing detailed job status in Redis/memory
class InternalJobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str
    completed_at: Optional[str] = None
    tasks: Dict[int, Dict] = {} # doctor_id -> {status, output_dir, error}
    final_output_files: Optional[Dict[str, str]] = None

def get_cache_key(request: SingleRequest) -> str:
    """Generate a cache key based on request parameters"""
    content = f"{request.doctor_id}_{sorted(request.patient_ids)}"
    return hashlib.md5(content.encode()).hexdigest()

def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status from in-memory storage."""
    return job_status_memory.get(job_id)

def set_job_status(job_id: str, status_data: Dict):
    """Set job status in in-memory storage."""
    job_status_memory[job_id] = status_data

def get_cache(cache_key: str) -> Optional[Dict]:
    """Get cached result from in-memory storage."""
    return cache_memory.get(cache_key)

def set_cache(cache_key: str, data: Dict, ttl: int = 7200):  # TTL is noted but not enforced
    """Set cached result in in-memory storage."""
    cache_memory[cache_key] = data

def create_samples_file(patient_ids: List[str], job_id: str) -> str:
    """Create a temporary samples file for the pipeline"""
    samples_file = f"temp_samples_{job_id}.txt"
    with open(samples_file, 'w') as f:
        for patient_id in patient_ids:
            f.write(f"{patient_id}\n")
    return samples_file

def generate_excel_output(json_data: Dict, job_id: str) -> str:
    """Convert JSON results to Excel format"""
    excel_file = f"results_{job_id}.xlsx"
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Process each data type
            for key, data in json_data.items():
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    sheet_name = key[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(data, dict):
                    # Convert dict to DataFrame
                    df = pd.DataFrame([data])
                    sheet_name = key[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return excel_file
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}")
        return None

# --- Helper functions for the new batch process ---

async def run_single_pipeline_task(job_id: str, request: SingleRequest, refresh: bool) -> Dict:
    """
    Runs the pipeline for a single doctor's list and returns the output directory.
    This function is designed to be run concurrently and supports caching.
    """
    doctor_id = request.doctor_id

    # --- Caching Logic ---
    cache_key = get_cache_key(request)
    if not refresh:
        cached_result = get_cache(cache_key)
        if cached_result:
            # Check if the cached files still exist on disk
            cached_files = cached_result.get("output_files", {})
            if all(os.path.exists(p) for p in cached_files.values()):
                logger.info(f"Using cached result for doctor_id {doctor_id}")
                return {
                    "doctor_id": doctor_id,
                    "status": "completed_from_cache",
                    "json_path": cached_files["json"],
                    "excel_path": cached_files["excel"],
                    "error": None,
                }
            else:
                logger.warning(f"Cache hit for doctor_id {doctor_id}, but output files are missing. Re-running.")

    # Create a dedicated directory for this sub-task's intermediate files
    sub_task_dir = os.path.join(os.getcwd(), "api_outputs", job_id, str(doctor_id))
    os.makedirs(sub_task_dir, exist_ok=True)
    
    samples_file = None
    try:
        samples_file = create_samples_file(request.patient_ids, f"{job_id}_{doctor_id}")
        
        cmd = [
            sys.executable, "src/run_pipeline_pq.py",
            "--samples", samples_file,
            "--output_dir", sub_task_dir,
            "--doctor_id", str(doctor_id),
            "--json_output",
        ]
        if refresh:
            cmd.append("--refresh")

        # Use asyncio.create_subprocess_exec for non-blocking execution
        # Run from root directory so all paths work correctly
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Pipeline failed for doctor_id {doctor_id}: {stderr.decode().strip()}")
            
        # Verify output files exist
        json_path = os.path.join(sub_task_dir, "matches_consolidated.json")
        excel_path = os.path.join(sub_task_dir, "matches_scoring_consolidated.xlsx")

        if not os.path.exists(json_path) or not os.path.exists(excel_path):
             raise FileNotFoundError(f"Expected output files not found in {sub_task_dir}")

        # --- Cache the successful result ---
        set_cache(cache_key, {
            "output_files": { "json": json_path, "excel": excel_path },
            "cached_at": datetime.now().isoformat()
        }, ttl=7200) # Cache for 2 hours

        return {
            "doctor_id": doctor_id,
            "status": "completed",
            "json_path": json_path,
            "excel_path": excel_path,
            "error": None
        }

    except Exception as e:
        return {
            "doctor_id": doctor_id,
            "status": "failed",
            "error": str(e)
        }
    finally:
        if samples_file and os.path.exists(samples_file):
            os.remove(samples_file)


def merge_json_results(tasks: List[Dict], final_dir: str, job_id: str) -> Optional[str]:
    """Merges all successful JSON results into a single file."""
    combined_data = {"matches": []}
    successful_tasks = [task for task in tasks if task["status"] in ["completed", "completed_from_cache"]]
    if not successful_tasks:
        return None
        
    for task in successful_tasks:
        try:
            with open(task["json_path"], 'r') as f:
                data = json.load(f)
            if "matches" in data and isinstance(data["matches"], list):
                combined_data["matches"].extend(data["matches"])
        except Exception as e:
            logger.error(f"Error reading JSON for doctor_id {task['doctor_id']}: {e}")

    final_json_path = os.path.join(final_dir, f"combined_results_{job_id}.json")
    with open(final_json_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    return final_json_path


def merge_excel_results(tasks: List[Dict], final_dir: str, job_id: str) -> Optional[str]:
    """Merges all successful Excel results into a single file."""
    all_dfs = []
    successful_tasks = [task for task in tasks if task["status"] in ["completed", "completed_from_cache"]]
    if not successful_tasks:
        return None

    for task in successful_tasks:
        try:
            df = pd.read_excel(task["excel_path"])
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading Excel for doctor_id {task['doctor_id']}: {e}")
            
    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    final_excel_path = os.path.join(final_dir, f"combined_results_{job_id}.xlsx")
    combined_df.to_excel(final_excel_path, index=False)
    return final_excel_path


async def run_batch_pipeline_async(job_id: str, request: BatchPipelineRequest):
    """The main background task to run and manage the batch processing, with a concurrent retry mechanism."""
    job_data = get_job_status(job_id)
    job_data["status"] = "running"
    job_data["message"] = f"Processing {len(request.requests)} doctor lists..."
    set_job_status(job_id, job_data)
    
    # --- First Pass: Run all tasks concurrently ---
    initial_coroutines = [run_single_pipeline_task(job_id, req, request.refresh) for req in request.requests]
    logger.info(f"Starting initial concurrent run for {len(initial_coroutines)} sub-tasks.")
    initial_results = await asyncio.gather(*initial_coroutines)
    
    # Identify successful and failed tasks from the first pass
    successful_on_first_pass = [res for res in initial_results if res["status"] in ["completed", "completed_from_cache"]]
    failed_requests = [req for req, res in zip(request.requests, initial_results) if res["status"] == "failed"]
    
    final_results = successful_on_first_pass

    # --- Retry Pass: Concurrently re-run only the failed tasks ---
    if failed_requests:
        logger.warning(f"Initial run failed for {len(failed_requests)} sub-tasks. Starting concurrent retry pass.")
        retry_coroutines = [run_single_pipeline_task(job_id, req, request.refresh) for req in failed_requests]
        retry_results = await asyncio.gather(*retry_coroutines)
        
        # Combine the successful results from the first pass with the results of the retry pass
        final_results.extend(retry_results)

    # --- Process and Aggregate Final Results ---
    successful_tasks = [res for res in final_results if res["status"] in ["completed", "completed_from_cache"]]
    failed_tasks = [res for res in final_results if res["status"] == "failed"]
    
    job_data["doctor_ids_success"] = sorted(list(set(task["doctor_id"] for task in successful_tasks)))
    job_data["doctor_ids_failed"] = sorted(list(set(task["doctor_id"] for task in failed_tasks)))
    job_data["errors"] = [{"doctor_id": task["doctor_id"], "error": task["error"]} for task in failed_tasks]

    final_output_dir = os.path.join(os.getcwd(), "api_outputs", job_id, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    final_json = merge_json_results(successful_tasks, final_output_dir, job_id)
    final_excel = merge_excel_results(successful_tasks, final_output_dir, job_id)
    
    if final_json and final_excel:
        job_data["status"] = "completed"
        job_data["message"] = f"Processed {len(successful_tasks)} of {len(request.requests)} lists successfully."
        job_data["output_files"] = {"json": final_json, "excel": final_excel}
    else:
        job_data["status"] = "failed"
        job_data["message"] = "Batch processing failed to generate any combined results."

    job_data["completed_at"] = datetime.now().isoformat()
    set_job_status(job_id, job_data)
    
    # Optional: Clean up intermediate sub-task directories
    for res in final_results: # Use final_results here
        sub_task_dir = os.path.join(os.getcwd(), "api_outputs", job_id, str(res["doctor_id"]))
        try:
            shutil.rmtree(sub_task_dir)
        except Exception as e:
            logger.error(f"Failed to clean up sub-task directory {sub_task_dir}: {e}")


# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "OncoTwin Simplified API", 
        "version": "1.0.0",
        "features": ["Redis caching", "Excel/JSON output", "Job tracking"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint, updated to remove Redis status."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/process")
async def process_patients_batch(
    request: BatchPipelineRequest,
    background_tasks: BackgroundTasks
):
    """Processes multiple sample lists from multiple doctors in a single job."""
    
    if not request.requests:
        raise HTTPException(status_code=400, detail="The 'requests' list cannot be empty.")
    
    job_id = str(uuid.uuid4())
    
    initial_job_data = {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued for execution.",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "doctor_ids_total": [req.doctor_id for req in request.requests],
        "doctor_ids_success": [],
        "doctor_ids_failed": [],
        "errors": [],
        "output_files": None
    }
    set_job_status(job_id, initial_job_data)
    
    background_tasks.add_task(run_batch_pipeline_async, job_id, request)
    
    return { "job_id": job_id, "message": "Batch job started.", "status": "pending" }


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    """Get the simplified status of a batch job."""
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Map the internal, detailed status to the simplified public response
    return JobStatusResponse(**job_data)

@app.get("/job/{job_id}/details")
async def get_job_details_debug(job_id: str):
    """
    (DEBUGGING) Returns the full internal state of a job, including detailed
    error messages for each failed sub-task.
    """
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data

@app.get("/download/{job_id}/{format}")
async def download_results(job_id: str, format: str):
    """Download results in specified format (json or excel)"""
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job_data.get("output_files"):
        raise HTTPException(status_code=404, detail="No output files found")
    
    if format not in ["json", "excel"]:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'excel'")
    
    file_path = job_data["output_files"].get(format)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{format.upper()} file not found")
    
    filename = f"results_{job_id}.{format if format != 'excel' else 'xlsx'}"
    return FileResponse(file_path, filename=filename)

@app.get("/results/{job_id}")
async def get_results_json(job_id: str):
    """Get results in JSON format directly. The doctor_id is now part of the source file."""
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logger.info(f"Getting results for job {job_id}, status: {job_data.get('status')}")
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job_data['status']}")
    
    # The inject_doctor_id function is no longer needed as the ID comes from the source file.

    # Check cache first
    cache_key = job_data.get("cache_key")
    if cache_key:
        cached_result = get_cache(cache_key)
        if cached_result and "json_results" in cached_result:
            logger.info(f"Returning cached results for job {job_id}")
            results = cached_result["json_results"]
            if not results:
                logger.warning(f"Cached results are empty for job {job_id}")
            return results
    
    # Fallback to reading from file
    output_files = job_data.get("output_files", {})
    logger.info(f"Output files for job {job_id}: {output_files}")
    
    json_file = output_files.get("json")
    
    if not json_file:
        if cache_key:
            cached_result = get_cache(cache_key)
            if cached_result and "json_results" in cached_result:
                return cached_result["json_results"]
        
        return { "error": "No JSON results file found", "debug_info": { "job_data": job_data, "cache_key": cache_key, "available_files": output_files } }
    
    if not os.path.exists(json_file):
        logger.error(f"JSON file {json_file} does not exist for job {job_id}")
        raise HTTPException(status_code=404, detail=f"JSON file not found: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Successfully read JSON file for job {job_id}, size: {len(str(results))} chars")
        if not results:
            logger.warning(f"JSON file is empty for job {job_id}")
        return results
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file} for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its status"""
    job_data = get_job_status(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete from Redis or memory
    if job_id in job_status_memory:
        del job_status_memory[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/cache/clear")
async def clear_cache():
    """Clear all cached results from the in-memory cache."""
    cleared_count = len(cache_memory)
    cache_memory.clear()
    return {"message": "In-memory cache cleared", "items_cleared": cleared_count}

@app.get("/cache/stats")
async def cache_stats():
    """Get in-memory cache statistics."""
    return {
        "caching_backend": "in_memory",
        "memory_cache_size": len(cache_memory),
        "job_status_size": len(job_status_memory)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 