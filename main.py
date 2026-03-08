import os
import json
import time
import logging
import psutil
from dotenv import load_dotenv

from src.db import DatabaseManager
from src.llm_client import LLMClient
from src.graph import PipelineGraph
from src.researcher import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Tier Classification ──
def classify_tier(confidence: float) -> int:
    """Classifies a generation into a quality tier based on critic confidence."""
    if confidence >= 0.85:
        return 1  # Gold
    elif confidence >= 0.70:
        return 2  # Silver
    elif confidence >= 0.60:
        return 3  # Bronze
    return 0  # Discard

# ── Turn Order Validation ──
def validate_turn_order(conversation_history: list) -> bool:
    """Validates strict user → assistant → user → assistant alternation."""
    if not conversation_history or len(conversation_history) < 2:
        return False
    expected_role = "user"
    for msg in conversation_history:
        if msg.get("role") != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"
    return True

# ── Metadata Completeness Check ──
REQUIRED_METADATA = ["persona_type", "conflict_type", "resolution_style", "difficulty_level", "domain"]
def validate_metadata(metadata: dict) -> bool:
    """Checks that all required metadata fields are present and non-empty."""
    if not metadata:
        return False
    for key in REQUIRED_METADATA:
        val = metadata.get(key, "")
        if not val or val == "Unknown":
            return False
    return True

def check_system_resources() -> str:
    """Checks CPU and Memory. Returns 'NORMAL', 'HIGH', or 'CRITICAL'"""
    try:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        
        if cpu > 95.0 or ram > 95.0:
            return "CRITICAL"
        elif cpu > 80.0 or ram > 80.0:
            return "HIGH"
        return "NORMAL"
    except Exception as e:
        logger.error(f"Failed to check resources: {e}")
        return "NORMAL"

def orchestrator_loop():
    logger.info("🚀 Starting Production Multi-Agent Orchestrator (Phase 1 Hardened)")
    load_dotenv()

    # Init Subsystems
    db = DatabaseManager()
    llm = LLMClient()
    teacher_graph = PipelineGraph(llm_client=llm)
    researcher = ResearchAgent(db_manager=db, llm_client=llm)
    
    while True:
        try:
            # ── Hot Reload API Keys ──
            llm.reload_keys()

            # ── UI Pause Check ──
            pipeline_flag = db.get_setting("pipeline_status")
            if pipeline_flag == "paused":
                logger.info("Pipeline PAUSED by Admin UI. Sleeping for 30 seconds...")
                time.sleep(30)
                continue

            # ── Calibration Mode Auto-Off ──
            gen_count = db.get_generation_count()
            if gen_count >= 500 and db.get_setting("calibration_mode") == "true":
                db.set_setting("calibration_mode", "false")
                logger.info("📊 Calibration mode COMPLETE. 500 generations reached. Switching to production mode.")

            # dynamic idle read
            idle_speed = int(db.get_setting("pipeline_idle") or 60)

            # ── Resource Check ──
            status = check_system_resources()
            if status == "CRITICAL":
                logger.warning(f"CRITICAL resource load detected! Suspending all operations for {idle_speed} seconds.")
                time.sleep(idle_speed)
                continue
                
            # ── 1. Research Agent Phase ──
            if status == "NORMAL":
                logger.info("System load normal. Waking up Research Agent...")
                researcher.generate_and_store_topics()
                
                # Persona distribution check (every cycle)
                dist_data = db.get_persona_distribution(last_n=500)
                if dist_data.get("warning"):
                    logger.warning(dist_data["warning"])
            else:
                logger.warning("System load HIGH. Delaying Research Agent to prioritize Teacher.")
            
            # ── Cost & Budget Check ──
            daily_cost = db.get_daily_cost()
            if daily_cost > 10.0: # $10.0 Daily Limit
                logger.warning(f"💸 DAILY BUDGET BREACHED (${daily_cost:.2f} > $10.00). Pausing pipeline for 1 hour.")
                time.sleep(3600)
                continue

            # ── 2. Teacher Agent Phase ──
            topic_record = db.get_pending_topic()
            
            if not topic_record:
                logger.info(f"No PENDING topics found in database. Sleeping for {idle_speed} seconds...")
                time.sleep(idle_speed)
                continue
                
            topic_id = topic_record["id"]
            topic_str = f"{topic_record['topic_title']} - {topic_record['topic_description']}"
            
            logger.info(f"Teacher Agent picked up topic ID {topic_id}: {topic_record['topic_title']}")
            
            cal_flag = db.get_setting("calibration_mode") or "false"
            gen_mode = "calibration" if cal_flag == "true" else "production"

            initial_state = {
                "topic": topic_str,
                "conversation_history": [],
                "metadata": {},
                "critic_data": {},
                "iterations": 0,
                "rejected": False,
                "api_failure": False,
                "current_turn": 1,
                "usage_log": [],
                "status": "pending_evaluation"
            }
            
            final_state = teacher_graph.graph.invoke(initial_state)
            
            # ── Extract primary model used for observability ──
            primary_model = "unknown"
            for usage in final_state.get("usage_log", []):
                db.log_cost(
                    model=usage.get("model", "unknown"),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
                if usage.get("model") != "unknown":
                    primary_model = usage.get("model")
                    
            if "critic_data" not in final_state:
                final_state["critic_data"] = {}
            final_state["critic_data"]["model_used"] = primary_model

            # ═══════════════════════════════════════════
            # 3. GUARD LAYER — Production Validation
            # ═══════════════════════════════════════════
            critic_data = final_state.get("critic_data", {})
            metadata = final_state.get("metadata", {})
            conversation = final_state.get("conversation_history", [])
            confidence = float(critic_data.get("confidence", 0.0))
            failure_type = critic_data.get("failure_type", "UNKNOWN")
            feedback = critic_data.get("feedback", "")
            iterations = final_state.get("iterations", 0)

            # Guard 0: API Failure Rescue
            if final_state.get("api_failure", False):
                logger.warning(f"⚠️ Topic {topic_id} hit an API/Rate Limit failure. Reverting to PENDING to save the topic.")
                db.mark_topic_status(topic_id, "PENDING")
                time.sleep(15)
                continue

            # Guard 1: Rejected or non-success status
            if final_state.get("rejected", False) or final_state.get("status") != "success":
                logger.warning(f"❌ Topic {topic_id} REJECTED. Failure: {failure_type}, Confidence: {confidence:.2f}")
                db.log_failure(topic_str, failure_type, confidence, feedback, iterations)
                db.mark_topic_status(topic_id, "FAILED")
                time.sleep(15)
                continue
            
            # Guard 2: Turn order validation
            if not validate_turn_order(conversation):
                logger.warning(f"❌ Topic {topic_id} INVALID turn order. Discarding.")
                db.log_failure(topic_str, "INVALID_TURN_ORDER", confidence, "Turn alternation violated", iterations)
                db.mark_topic_status(topic_id, "FAILED")
                time.sleep(15)
                continue
            
            # Guard 3: Metadata completeness
            if not validate_metadata(metadata):
                logger.warning(f"⚠️ Topic {topic_id} has incomplete metadata. Discarding.")
                db.log_failure(topic_str, "INCOMPLETE_METADATA", confidence, f"Missing fields in: {metadata}", iterations)
                db.mark_topic_status(topic_id, "FAILED")
                time.sleep(15)
                continue

            # Guard 4: Tier classification
            tier = classify_tier(confidence)
            if tier == 0:
                logger.warning(f"❌ Topic {topic_id} below Tier 3 threshold (conf: {confidence:.2f}). Discarding.")
                db.log_failure(topic_str, failure_type, confidence, feedback, iterations)
                db.mark_topic_status(topic_id, "FAILED")
                time.sleep(15)
                continue
            
            # ═══════════════════════════════════════════
            # 4. SAVE to Database (passed all guards)
            # ═══════════════════════════════════════════
            row_id = db.insert_generation(
                topic=topic_str,
                conversation_history=conversation,
                metadata=metadata,
                critic_data=critic_data,
                tier=tier,
                mode=gen_mode
            )
            if row_id:
                db.mark_topic_status(topic_id, "PROCESSED")
                tier_label = {1: "🥇 Tier 1 (Gold)", 2: "🥈 Tier 2 (Silver)", 3: "🥉 Tier 3 (Bronze)"}
                logger.info(f"✅ Topic {topic_id} → {tier_label.get(tier)} [{gen_mode}] | Confidence: {confidence:.2f} | Saved!")
            else:
                db.mark_topic_status(topic_id, "FAILED (DUPLICATE)")
            
            # Dynamic speed control from UI
            pipeline_speed = int(db.get_setting("pipeline_speed") or 15)
            logger.info(f"Orchestrator sleeping for {pipeline_speed} seconds before next cycle (configurable via UI)...")
            time.sleep(pipeline_speed)
            
        except Exception as e:
            logger.error(f"Critical Orchestrator Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    orchestrator_loop()
