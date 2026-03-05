import logging
import random
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END

from src.llm_client import LLMClient
from src.prompts import (
    USER_TURN_PROMPT, 
    ASSISTANT_TURN_PROMPT, 
    METADATA_GENERATOR_PROMPT,
    CRITIC_SYSTEM_PROMPT, 
    REFLECTION_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

# ── Personas & Conflict Types for User Simulation ──
ADVERSARIAL_PERSONAS = [
    "Skeptical Scientist", "Confidently Incorrect Layman", "Aggressive Debunker", 
    "Conspiracy Theorist", "Nitpicking Expert", "Provocative Journalist"
]
CONFLICT_STRATEGIES = [
    "Factual Misconception", "Logical Fallacy (Strawman)", "Moving the Goalposts",
    "Emotional Bias", "Demanding Impossible Evidence", "False Dilemma"
]

class GraphState(TypedDict):
    topic: str
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]
    critic_data: Dict[str, Any]
    status: str
    iterations: int
    rejected: bool
    current_turn: int  # 1 to 3
    usage_log: List[Dict[str, Any]] # To track cost across all nodes

class PipelineGraph:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        
        # Define Nodes
        workflow.add_node("user_turn", self.node_user_turn)
        workflow.add_node("assistant_turn", self.node_assistant_turn)
        workflow.add_node("generate_metadata", self.node_generate_metadata)
        workflow.add_node("evaluate", self.node_evaluate)
        workflow.add_node("reflect", self.node_reflect)

        # Define Edges
        workflow.set_entry_point("user_turn")
        
        workflow.add_edge("user_turn", "assistant_turn")
        
        workflow.add_conditional_edges(
            "assistant_turn",
            self.edge_after_assistant,
            {"next_turn": "user_turn", "finalize": "generate_metadata"}
        )
        
        workflow.add_edge("generate_metadata", "evaluate")
        
        workflow.add_conditional_edges(
            "evaluate",
            self.edge_after_evaluate,
            {"end": END, "reflect": "reflect"}
        )
        workflow.add_edge("reflect", "evaluate")

        return workflow.compile()

    def node_user_turn(self, state: GraphState) -> Dict[str, Any]:
        turn = state.get("current_turn", 1)
        history = state.get("conversation_history", [])
        
        # Set persona/conflict if 1st turn
        if turn == 1:
            persona = random.choice(ADVERSARIAL_PERSONAS)
            conflict = random.choice(CONFLICT_STRATEGIES)
            state["metadata"] = {"persona_type": persona, "conflict_type": conflict}
        else:
            persona = state["metadata"].get("persona_type", "Skeptical")
            conflict = state["metadata"].get("conflict_type", "General Doubt")

        logger.info(f"Graph: Generating User Turn {turn} (Persona: {persona})")
        
        system_msg = USER_TURN_PROMPT.format(persona_type=persona, conflict_type=conflict)
        prompt = f"Topic: {state['topic']}\n\nConversation so far:\n{history}\n\nGenerate your turn:"
        
        try:
            # User turn is a simple string, generate returns {"data": parsed_json, "usage": ...}
            # BUT raw_content is actually what we want for User. LLMClient.generate tries to parse JSON.
            # I'll use a trick: LLMClient.extract_json will fail, and maybe LLMClient.generate needs a 'raw' mode.
            # For now, I'll rely on LLMClient._make_api_call directly for raw string or update generate.
            # Actually, I updated generate to return {"data": parsed_json, "usage": usage}.
            # If it's a raw string, extract_json returns it if it can't find {}.
            
            result_wrapper = self.llm.generate(prompt=prompt, system_message=system_msg, temperature=0.8, expect_json=False)
            content = result_wrapper.get("data")
            
            usage = result_wrapper.get("usage", {})
            
            history.append({"role": "user", "content": str(content)})
            
            return {
                "conversation_history": history,
                "current_turn": turn,
                "usage_log": state.get("usage_log", []) + [usage]
            }
        except Exception as e:
            logger.error(f"User turn generation failed: {e}")
            return {"rejected": True, "status": "failed"}

    def node_assistant_turn(self, state: GraphState) -> Dict[str, Any]:
        turn = state.get("current_turn", 1)
        history = state.get("conversation_history", [])
        logger.info(f"Graph: Generating Assistant Turn {turn}")
        
        prompt = f"Topic: {state['topic']}\n\nConversation thus far:\n{history}\n\nGenerate your response (JSON):"
        
        try:
            result_wrapper = self.llm.generate(prompt=prompt, system_message=ASSISTANT_TURN_PROMPT, temperature=0.3)
            data = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            history.append({
                "role": "assistant",
                "reasoning": data.get("reasoning", ""),
                "content": data.get("content", "")
            })
            
            return {
                "conversation_history": history,
                "usage_log": state.get("usage_log", []) + [usage],
                "current_turn": turn + 1
            }
        except Exception as e:
            logger.error(f"Assistant turn generation failed: {e}")
            return {"rejected": True}

    def edge_after_assistant(self, state: GraphState) -> str:
        if state.get("rejected"):
            return "finalize"
        turn = state.get("current_turn", 1)
        if turn > 3:
            return "finalize"
        return "next_turn"

    def node_generate_metadata(self, state: GraphState) -> Dict[str, Any]:
        logger.info("Graph: Generating final conversation metadata...")
        history = state.get("conversation_history", [])
        prompt = METADATA_GENERATOR_PROMPT.format(history=history)
        
        try:
            result_wrapper = self.llm.generate(prompt=prompt, temperature=0.1)
            data = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            return {
                "metadata": data,
                "usage_log": state.get("usage_log", []) + [usage]
            }
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"metadata": state.get("metadata", {})}

    def node_evaluate(self, state: GraphState) -> Dict[str, Any]:
        """Critic Agent: cross-model evaluation with server-side weighted average."""
        if state.get("rejected"):
            return state

        logger.info("Critic Agent: Evaluating granular conversation...")
        history = state.get("conversation_history", [])
        
        prompt = f"Evaluate this turn-by-turn conversation:\n\n{history}"
        
        try:
            result_wrapper = self.llm.generate(
                prompt=prompt, 
                system_message=CRITIC_SYSTEM_PROMPT, 
                temperature=0.1, 
                role="reasoning"
            )
            result = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            scores = result.get("scores", {})
            mem = float(scores.get("memory_consistency", 0.0))
            logic = float(scores.get("logic", 0.0))
            conflict = float(scores.get("conflict_resolution", 0.0))
            empathy = float(scores.get("empathy", 0.0))
            
            computed_confidence = round(mem * 0.35 + logic * 0.30 + conflict * 0.25 + empathy * 0.10, 3)
            
            critic_status = "PASS" if computed_confidence >= 0.60 else "FAIL"
            failure_type = result.get("failure_type", "NONE")
            
            critic_data = {
                "status": critic_status,
                "confidence": computed_confidence,
                "memory_consistency": mem,
                "failure_type": failure_type,
                "feedback": result.get("feedback", ""),
                "scores": scores
            }
            
            return {
                "critic_data": critic_data,
                "metadata": result.get("verified_metadata", state.get("metadata")),
                "status": "success" if critic_status == "PASS" else "needs_reflection",
                "usage_log": state.get("usage_log", []) + [usage]
            }
        except Exception as e:
            logger.error(f"Critic Agent failed: {e}")
            return {"status": "needs_reflection", "rejected": True}

    def edge_after_evaluate(self, state: GraphState) -> str:
        if state.get("status") == "success" or state.get("rejected"):
            return "end"
        if state.get("iterations", 0) >= 1:
            return "end"
        return "reflect"

    def node_reflect(self, state: GraphState) -> Dict[str, Any]:
        iterations = state.get("iterations", 0)
        logger.info(f"Teacher Agent: Reflecting (Iteration {iterations + 1})")
        
        history = state.get("conversation_history", [])
        critic = state.get("critic_data", {})
        
        # We still use the monolithic reflection for now, but strictly for Assistant turns
        prompt = f"Correct the assistant responses in this history based on feedback: {critic.get('feedback')}\n\nHistory: {history}"
        
        try:
            result_wrapper = self.llm.generate(prompt=prompt, system_message=REFLECTION_SYSTEM_PROMPT, temperature=0.3, role="reasoning")
            data = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            return {
                "conversation_history": data.get("conversation_history", history),
                "iterations": iterations + 1,
                "usage_log": state.get("usage_log", []) + [usage],
                "current_turn": 3 # Enforce end of loop if we reflect
            }
        except Exception as e:
            logger.error(f"Teacher reflect failed: {e}")
            return {"rejected": True}
