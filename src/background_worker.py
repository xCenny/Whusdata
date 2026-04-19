import logging
import threading
import json
from typing import Dict, Any
from src.db import DatabaseManager
from src.llm_client import LLMClient
from src.prompts import METADATA_GENERATOR_PROMPT

logger = logging.getLogger(__name__)


def run_retag_job(job_id: int, targets: list, retag_model: str, only_unknown: bool):
    """Runs re-tagging job in background thread."""
    db = DatabaseManager()
    llm = LLMClient()
    
    success_count = 0
    error_count = 0
    
    for i, target in enumerate(targets):
        try:
            try:
                convo = json.loads(target.get("conversation_history", "[]"))
            except (json.JSONDecodeError, TypeError):
                convo = []
            
            if not convo:
                error_count += 1
                db.update_background_job(job_id, progress=i+1, success_count=success_count, error_count=error_count)
                continue
            
            history_text = ""
            for msg in convo:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n\n"
            
            prompt = METADATA_GENERATOR_PROMPT.format(history=history_text)
            
            result_wrapper = llm.generate(
                prompt=prompt,
                temperature=0.1,
                role="fast",
                force_model=retag_model
            )
            result = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            if usage:
                db.log_cost(
                    model=usage.get("model", "unknown"),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
            
            if isinstance(result, dict) and result:
                new_tags = {}
                for field in ["domain", "conflict_type", "resolution_style", "difficulty_level", "broad_category", "detailed_persona"]:
                    val = result.get(field, "")
                    if val and val != "Unknown":
                        if only_unknown:
                            current_val = target.get(field, "") or ""
                            if current_val in ("", "Unknown"):
                                new_tags[field] = val
                        else:
                            new_tags[field] = val
                
                if new_tags:
                    db.update_generation_tags(target["id"], new_tags)
                    success_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"Re-tag error for ID {target.get('id')}: {e}")
            error_count += 1
        
        db.update_background_job(job_id, progress=i+1, success_count=success_count, error_count=error_count)
    
    db.update_background_job(
        job_id, 
        status="DONE", 
        progress=len(targets),
        success_count=success_count, 
        error_count=error_count,
        result_message=f"✅ {success_count} records updated, {error_count} skipped/failed."
    )
    logger.info(f"Re-tag job {job_id} completed: {success_count} success, {error_count} errors.")


def run_augment_job(job_id: int, targets: list, multiplier: int, model_name: str):
    """Runs augmentation job in background thread."""
    from src.augmenter import DataAugmenter
    
    db = DatabaseManager()
    llm = LLMClient()
    augmenter = DataAugmenter(db, llm)
    
    success_count = 0
    error_count = 0
    
    for i, gen in enumerate(targets):
        try:
            count = augmenter.augment_generation(gen, multiplier, model_name)
            success_count += count
            if count == 0:
                error_count += 1
        except Exception as e:
            logger.error(f"Augment error for ID {gen.get('id')}: {e}")
            error_count += 1
        
        db.update_background_job(job_id, progress=i+1, success_count=success_count, error_count=error_count)
    
    db.update_background_job(
        job_id, 
        status="DONE", 
        progress=len(targets),
        success_count=success_count, 
        error_count=error_count,
        result_message=f"✅ {success_count} augmented records created from {len(targets)} originals. {error_count} failed."
    )
    logger.info(f"Augment job {job_id} completed: {success_count} augmented, {error_count} errors.")


def start_retag_job(targets: list, retag_model: str, only_unknown: bool) -> int:
    """Creates and starts a re-tag background job. Returns job_id."""
    db = DatabaseManager()
    job_id = db.create_background_job(
        job_type="retag",
        total=len(targets),
        config={"model": retag_model, "only_unknown": only_unknown}
    )
    
    thread = threading.Thread(
        target=run_retag_job,
        args=(job_id, targets, retag_model, only_unknown),
        daemon=True,
        name=f"retag-job-{job_id}"
    )
    thread.start()
    logger.info(f"Started background re-tag job {job_id} with {len(targets)} targets.")
    return job_id


def start_augment_job(targets: list, multiplier: int, model_name: str) -> int:
    """Creates and starts an augmentation background job. Returns job_id."""
    db = DatabaseManager()
    job_id = db.create_background_job(
        job_type="augment",
        total=len(targets),
        config={"model": model_name, "multiplier": multiplier}
    )
    
    thread = threading.Thread(
        target=run_augment_job,
        args=(job_id, targets, multiplier, model_name),
        daemon=True,
        name=f"augment-job-{job_id}"
    )
    thread.start()
    logger.info(f"Started background augment job {job_id} with {len(targets)} targets.")
    return job_id
