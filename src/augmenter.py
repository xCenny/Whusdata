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
2. PRESERVE THE ROLES: Keep the identical sequence of "user" and "assistant" turns. Same number of turns, same order.
3. APPLY THE VARIATION STYLE: Rewrite the content to strictly adhere to the requested Variation Style.
4. ABSOLUTELY NO METADATA CHANGES: Output ONLY the updated JSON array of messages.
5. MAINTAIN QUALITY: The rewritten conversation must be at least as engaging and well-structured as the original. Do NOT simplify or dumb down the arguments.
6. NO HALLUCINATION: Do NOT add new facts, statistics, or claims that were not in the original.
7. NATURAL FLOW: The rewritten conversation must read naturally — like two real people having this discussion in the specified style.

VARIATION STYLE: {variation_style}

[WARNING: Respond ONLY with the raw JSON array. Do NOT include ANY conversational text before or after the JSON. Do NOT use markdown code blocks like ```json.]

OUTPUT FORMAT:
[
  {{"role": "user", "content": "rewritten message..."}},
  {{"role": "assistant", "reasoning": "brief internal reasoning...", "content": "rewritten message..."}},
  ...
]
"""

VARIATION_STYLES = [
    # Original 3 — improved
    "VOCABULARY_SHIFT: Keep the exact same tone and length, but completely change the vocabulary. Use synonyms and rephrase every sentence structurally. The meaning must be identical but the wording must be entirely different. Vary sentence openings and structures.",
    
    "CONCISE_AND_PUNCHY: Make every single turn much shorter, more direct, and concise. Remove any conversational filler. Be extremely blunt but strictly preserve the core arguments. Use short declarative sentences. Maximum 2-3 sentences per turn.",
    
    "ACADEMIC_AND_VERBOSE: Rewrite every turn to sound like a highly formal academic debate between two professors. Use scholarly vocabulary, complex sentence structures, and frequent references to theoretical frameworks. Each turn should be longer and more detailed.",
    
    # New styles for variety
    "SOCRATIC_METHOD: Rewrite the user's turns to use more leading questions and Socratic dialogue techniques. The user should challenge through questions rather than statements. The assistant should respond with structured logical breakdowns.",
    
    "AGGRESSIVE_STREET_DEBATE: Rewrite the entire conversation in a casual, street-smart tone. Use colloquial language, slang expressions, and informal grammar. The arguments must stay the same but the delivery should feel like two passionate people arguing at a coffee shop.",
    
    "EMOTIONALLY_CHARGED: Amplify the emotional intensity of both sides. The user should be more frustrated, passionate, or sarcastic. The assistant should be firm but empathetic. Add emotional texture while preserving the logical arguments exactly.",
    
    "TECHNICAL_EXPERT: Rewrite the conversation as if both participants are domain experts. Use precise technical terminology, reference specific methodologies, and structure arguments with numbered points or clear logical chains. Avoid any hand-waving.",
    
    "JOURNALIST_INTERVIEW: Rewrite the user's turns as probing journalistic questions — sharp, pointed, demanding answers. The assistant should respond like a knowledgeable expert being interviewed, giving clear, quotable answers with analogies.",
    
    "DEVIL_ADVOCATE_INTENSE: Amplify the adversarial nature. The user should be much more aggressive, confrontational, and use stronger rhetorical devices. The assistant must counter with calm precision, dismantling each point methodically.",
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
            entry = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            stripped_history.append(entry)
            
        history_str = json.dumps(stripped_history, ensure_ascii=False, indent=2)
        
        success_count = 0
        
        # Pick random distinct styles — allow up to multiplier styles
        styles_to_use = random.sample(VARIATION_STYLES, min(multiplier, len(VARIATION_STYLES)))
        
        for style in styles_to_use:
            style_name = style.split(":")[0].strip()
            logger.info(f"Augmenting ID {original_id} with style: {style_name}")
            system_msg = AUGMENTER_SYSTEM_PROMPT.format(variation_style=style)
            prompt = f"Original Conversation ({len(stripped_history)} turns):\n{history_str}\n\nIMPORTANT: Your output MUST contain exactly {len(stripped_history)} messages in the same role order. Rewrite this conversation as instructed."
            
            try:
                result = self.llm.generate(
                    prompt=prompt,
                    system_message=system_msg,
                    temperature=0.7,
                    force_model=model_name
                )
                
                raw_data = result.get("data", [])
                usage = result.get("usage", {})
                
                # Log cost
                if usage:
                    self.db.log_cost(
                        model=usage.get("model", "unknown"),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0)
                    )
                
                # Handle case where data might be dict with a key containing the array
                new_history = raw_data
                if isinstance(raw_data, dict):
                    # Try common keys
                    for key in ["messages", "conversation", "history", "data"]:
                        if key in raw_data and isinstance(raw_data[key], list):
                            new_history = raw_data[key]
                            break
                
                if not isinstance(new_history, list):
                    logger.warning(f"Augmentation for ID {original_id} returned non-list: {type(new_history)}")
                    continue
                
                if len(new_history) != len(stripped_history):
                    logger.warning(f"Augmentation mismatch for ID {original_id}. Expected {len(stripped_history)} turns, got {len(new_history)}.")
                    # Try to salvage if close (within 1 turn)
                    if abs(len(new_history) - len(stripped_history)) > 1:
                        continue
                    # Truncate or pad to match
                    new_history = new_history[:len(stripped_history)]
                    
                # Re-attach reasoning for assistant turns
                final_history = []
                for i, msg in enumerate(new_history):
                    entry = {
                        "role": msg.get("role", stripped_history[i]["role"] if i < len(stripped_history) else "user"),
                        "content": msg.get("content", "")
                    }
                    if entry["role"] == "assistant":
                        entry["reasoning"] = msg.get("reasoning", f"Augmented ({style_name})")
                    final_history.append(entry)
                    
                # Reconstruct metadata from original
                metadata = {
                    "persona_type": gen.get("persona_type", "Unknown"),
                    "conflict_type": gen.get("conflict_type", "Unknown"),
                    "resolution_style": gen.get("resolution_style", "Unknown"),
                    "difficulty_level": gen.get("difficulty_level", "Unknown"),
                    "domain": gen.get("domain", "Unknown"),
                    "model_used": model_name
                }
                
                critic_data = {
                    "status": "PASS",
                    "confidence": gen.get("critic_confidence", 0.0),
                    "memory_consistency": gen.get("memory_consistency_score", 0.0),
                    "logic_score": gen.get("logic_score", 0.0),
                    "winner": gen.get("winner", "Unknown"),
                    "failure_type": "NONE",
                    "model_used": "Bypassed (Augmented)",
                    "analytics": json.loads(gen.get("critic_analytics", "{}")),
                    "factual_score": gen.get("factual_score", 0.0)
                }
                
                logger.info(f"Inserting Augmented Generation (Derived from ID {original_id}, style: {style_name})")
                
                res = self.db.insert_generation(
                    topic=gen["topic"] + f" (Aug:{style_name})",
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
