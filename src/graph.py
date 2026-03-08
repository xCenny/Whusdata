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

# ── Personas, Lengths & Tactics ──
def get_random_persona() -> str:
    r = random.random()
    if r < 0.30: # 30% Hostile/Troll
        return random.choice(["Aggressive Debunker", "Angry Troll", "Hostile Critic"])
    elif r < 0.70: # 40% Normal Skeptical
        return random.choice(["Skeptical Scientist", "Nitpicking Expert", "Devils Advocate", "Cautious Reviewer"])
    elif r < 0.90: # 20% Educational
        return random.choice(["Curious Student", "Inquisitive Researcher", "Layman seeking truth"])
    else: # 10% Absurd
        return random.choice(["Conspiracy Theorist", "Flat-Earther", "Sci-Fi Fanatic"])

def get_length_directive() -> str:
    r = random.random()
    if r < 0.30:
        return "SHORT (2-3 sentences max, strictly punchy and direct)"
    elif r < 0.70:
        return "MEDIUM (4-6 sentences, balanced counter-argument)"
    else:
        return "LONG (Comprehensive and detailed rebuttal)"

def get_user_tactic(turn: int) -> str:
    if turn == 1:
        return "Assert a deeply flawed initial claim or severe misunderstanding with absolute confidence."
    elif turn == 2:
        return random.choice([
            "Demand highly specific evidence/sources for the AI's previous claim. ('Source?', 'Who ran this study?')",
            "Attack the AI using a Logical Fallacy (e.g., Strawman, Ad Hominem, Appeal to Authority).",
            "Nitpick a minor detail from the AI's response and blow it completely out of proportion."
        ])
    elif turn == 3:
        return random.choice([
            "Attempt a Contradiction Trap ('Earlier you said X, now you're saying Y. Which is it?').",
            "Aggressively dismiss the AI's evidence as biased, fake, or irrelevant.",
            "Stubbornly double down on your original fallacy and refuse to concede."
        ])
    elif turn == 4:
        return random.choice([
            "Employ a False Dilemma ('So you are saying it is either 100% perfect or completely useless?').",
            "Move the Goalposts: Completely change your standard of proof now that the AI has answered your previous point.",
            "Feign ignorance and pretend you don't understand the AI's core explanation."
        ])
    else:  # Turn 5 and 6
        return random.choice([
            "Resort to a pure emotional appeal or anecdotal evidence that 'proves' you are right.",
            "Try to quickly change the subject to a completely different but related provocative topic.",
            "Make one final desperate logical leap to try and win the argument before giving up."
        ])

class GraphState(TypedDict):
    topic: str
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]
    critic_data: Dict[str, Any]
    status: str
    iterations: int
    rejected: bool
    api_failure: bool
    conclude_debate: bool
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
        
        # Set persona if 1st turn
        if turn == 1:
            persona = get_random_persona()
            # Dynamic Turn Optimization: Favor 3-4 turns to prevent logical drift, rarely 5-6 for deep topics.
            target_turns = random.choices([3, 4, 5, 6], weights=[0.35, 0.45, 0.15, 0.05])[0]
            state["metadata"] = {
                "persona_type": persona, 
                "conflict_type": "Variable Tactic",
                "target_turns": target_turns
            }
        else:
            persona = state["metadata"].get("persona_type", "Skeptical")

        tactic = get_user_tactic(turn)

        logger.info(f"Graph: Generating User Turn {turn} (Persona: {persona}) | Tactic: {tactic[:40]}...")
        
        system_msg = USER_TURN_PROMPT.format(persona_type=persona, user_tactic=tactic)
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
                "usage_log": state.get("usage_log", []) + [usage],
                "metadata": state.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"User turn generation failed: {e}")
            return {"rejected": True, "api_failure": True, "status": "failed"}

    def node_assistant_turn(self, state: GraphState) -> Dict[str, Any]:
        turn = state.get("current_turn", 1)
        history = state.get("conversation_history", [])
        length_directive = get_length_directive()
        
        logger.info(f"Graph: Generating Assistant Turn {turn} | Length: {length_directive[:13]}")
        
        system_msg = ASSISTANT_TURN_PROMPT.format(length_directive=length_directive)
        prompt = f"Topic: {state['topic']}\n\nConversation thus far:\n{history}\n\nGenerate your response (JSON):"
        
        try:
            result_wrapper = self.llm.generate(prompt=prompt, system_message=system_msg, temperature=0.3)
            data = result_wrapper.get("data", {})
            usage = result_wrapper.get("usage", {})
            
            history.append({
                "role": "assistant",
                "reasoning": data.get("reasoning", ""),
                "content": data.get("content", "")
            })
            
            conclude = data.get("conclude_debate", False)
            if conclude:
                logger.info("Graph: Assistant elected to conclude the debate autonomously to prevent drift.")
            
            return {
                "conversation_history": history,
                "usage_log": state.get("usage_log", []) + [usage],
                "current_turn": turn + 1,
                "conclude_debate": conclude
            }
        except Exception as e:
            logger.error(f"Assistant turn generation failed: {e}")
            return {"rejected": True, "api_failure": True}

    def edge_after_assistant(self, state: GraphState) -> str:
        if state.get("rejected"):
            return "finalize"
        
        turn = state.get("current_turn", 1) # This is actually the NEXT turn number now
        
        # Enforce minimum 2 turns before allowing autonomous conclusion
        if state.get("conclude_debate", False) and turn > 2:
            return "finalize"
            
        turn = state.get("current_turn", 1) # This is actually the NEXT turn number now
        target_turns = state.get("metadata", {}).get("target_turns", 3)
        if turn > target_turns:
            return "finalize"
        return "next_turn"

    def node_generate_metadata(self, state: GraphState) -> Dict[str, Any]:
        if state.get("rejected"):
            return state
            
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
            return {"metadata": state.get("metadata", {}), "rejected": True, "api_failure": True}

    def node_evaluate(self, state: GraphState) -> Dict[str, Any]:
        """Critic Agent: cross-model evaluation with server-side weighted average."""
        if state.get("rejected"):
            return state

        logger.info("Critic Agent: Evaluating granular conversation...")
        history = state.get("conversation_history", [])
        
        prompt = f"Evaluate this turn-by-turn conversation:\n\n{history}"
        critic_override = self.llm.db.get_setting("critic_model_override") or "Default (Round-Robin)"
        
        try:
            result_wrapper = self.llm.generate(
                prompt=prompt, 
                system_message=CRITIC_SYSTEM_PROMPT, 
                temperature=0.1, 
                role="reasoning",
                force_model=critic_override
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
                "logic_score": logic,
                "winner": result.get("winner", "Unknown"),
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
            return {"status": "needs_reflection", "rejected": True, "api_failure": True}

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
