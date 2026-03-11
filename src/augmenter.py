import logging
import json
import random
from typing import Dict, Any, List
from src.db import DatabaseManager
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

AUGMENTER_SYSTEM_PROMPT = """You are an Expert Data Paraphraser running a synthetic data pipeline.
Your job is to take a complete multi-turn debate (JSON array of messages) and rewrite IT ENTIRELY according to a specific stylistic variation.

CRITICAL RULES:
1. PRESERVE THE LOGIC EXACTLY: You must not change the underlying facts, logical fallacies used, the winner of the debate, or the specific points being argued.
2. PRESERVE THE ROLES: Keep the identical sequence of "user" and "assistant" turns.
3. APPLY THE VARIATION STYLE: Rewrite the content to strictly adhere to the requested Variation Style.
4. ABSOLUTELY NO METADATA CHANGES: Output ONLY the updated JSON array of messages.

VARIATION STYLE: {variation_style}

[WARNING: Respond ONLY with the raw JSON array. Do NOT include ANY conversational text before or after the JSON. Do NOT use markdown code blocks like ```json.]

OUTPUT FORMAT:
[
  {{"role": "user", "content": "rewritten message..."}},
  {{"role": "assistant", "content": "rewritten message..."}},
  ...
]
"""

VARIATION_STYLES = [
    "VOCABULARY_SHIFT: Keep the exact same tone and length, but completely change the vocabulary. Use synonyms and rephrase sentences structurally while keeping the literal meaning identical.",
    "CONCISE_AND_PUNCHY: Make every single turn much shorter, more direct, and concise. Remove any conversational filler. Be extremely blunt but strictly preserve the core arguments.",
    "ACADEMIC_AND_VERBOSE: Rewrite every turn to sound like it is taking place in a highly formal, verbose academic debate. Use overly intellectual language and complex sentence structures."
]

class DataAugmenter:
    def __init__(self, db: DatabaseManager, llm: LLMClient):
        self.db = db
        self.llm = llm

    def augment_generation(self, gen: Dict[str, Any], multiplier: int, model_name: str) -> int:
        """
        Takes a single generation dictionary, generates 'multiplier' variations,
        inserts them into the DB as is_augmented=1, and returns the number of successful augmentations.
        """
        try:
            history = json.loads(gen["conversation_history"])
            original_id = gen["id"]
        except Exception as e:
            logger.error(f"Cannot parse history for augmentation (ID {gen['id']}): {e}")
            return 0
            
        # Strip out 'reasoning' from assistant turns to just focus on the visible conversation
        stripped_history = []
        for msg in history:
            stripped_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
            
        history_str = json.dumps(stripped_history, ensure_ascii=False, indent=2)
        
        success_count = 0
        
        # Pick random distinct styles
        styles_to_use = random.sample(VARIATION_STYLES, min(multiplier, len(VARIATION_STYLES)))
        
        for style in styles_to_use:
            logger.info(f"Augmenting ID {original_id} with style: {style[:20]}...")
            system_msg = AUGMENTER_SYSTEM_PROMPT.format(variation_style=style)
            prompt = f"Original Conversation:\n{history_str}\n\nRewrite this conversation as instructed."
            
            try:
                # We expect a JSON array back
                result = self.llm.generate(
                    prompt=prompt,
                    system_message=system_msg,
                    temperature=0.7, # Higher temperature for variation
                    force_model=model_name
                )
                
                new_history = result.get("data", [])
                
                if not isinstance(new_history, list) or len(new_history) != len(stripped_history):
                    logger.warning(f"Augmentation mismatch for ID {original_id}. Expected {len(stripped_history)} turns, got {len(new_history) if isinstance(new_history, list) else 'non-list'}.")
                    continue
                    
                # Re-attach fake empty reasoning just to keep DB schema happy if needed, or leave it out
                final_history = []
                for i, msg in enumerate(new_history):
                    if msg.get("role") == "assistant":
                        msg["reasoning"] = f"Augmented variation ({style[:15]}...)"
                    final_history.append(msg)
                    
                # Reconstruct metadata and critic data from the original DB row
                metadata = {
                    "persona_type": gen["persona_type"],
                    "conflict_type": gen["conflict_type"],
                    "resolution_style": gen["resolution_style"],
                    "difficulty_level": gen["difficulty_level"],
                    "domain": gen["domain"],
                    "model_used": model_name
                }
                
                critic_data = {
                    "status": "PASS", # Intentionally passing augmented data by default based on user agreement
                    "confidence": gen["critic_confidence"],
                    "memory_consistency": gen["memory_consistency_score"],
                    "logic_score": gen["logic_score"],
                    "winner": gen.get("winner", "Unknown"),
                    "failure_type": "NONE",
                    "model_used": "Bypassed (Augmented)",
                    "analytics": json.loads(gen.get("critic_analytics", "{}")),
                    "factual_score": gen.get("factual_score", 0.0)
                }
                
                logger.info(f"Inserting Augmented Generation (Derived from ID {original_id})")
                
                res = self.db.insert_generation(
                    topic=gen["topic"] + " (Augmented)",
                    conversation_history=final_history,
                    metadata=metadata,
                    critic_data=critic_data,
                    tier=gen.get("tier", 0),
                    mode="augmentation",
                    is_augmented=True,
                    original_id=original_id
                )
                
                if res:
                    success_count += 1
                
            except Exception as e:
                logger.error(f"Failed augmentation iteration for ID {original_id}: {e}")
                
        return success_count
