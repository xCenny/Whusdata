import logging
import wikipedia
import random
from typing import List, Dict, Any
from src.db import DatabaseManager
from src.llm_client import LLMClient
from src.prompts import RESEARCHER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, db_manager: DatabaseManager, llm_client: LLMClient):
        self.db = db_manager
        self.llm = llm_client
        # Deep internet knowledge seeds to trigger complex Wikipedia exploration
        self.seed_words = [
            "Quantum entanglement paradoxes", "Epigenetic inheritance", "Bronze Age Collapse mathematics",
            "Non-Euclidean geometry principles", "Neuroplasticity in adult brains", "Cryptographic hash collisions",
            "Thermodynamic entropy paradoxes", "Linguistic relativity", "The Fermi Paradox solutions",
            "Socio-cultural evolution theories", "Post-structuralism concepts", "Advanced robotics kinematics",
            "Geopolitical choke points", "Paleoclimatology data", "Microbiome symbiosis",
            "Theoretical astrophysics anomalies", "Behavioral economics heuristics", "Ancient metallurgy techniques",
            "CRISPR off-target effects", "Turing machine limitations", "Dark matter detection methods"
        ]

    def _get_search_words(self) -> list:
        """Prioritizes UI-injected keywords over random seed words."""
        active_keywords = self.db.get_active_keywords()
        if active_keywords:
            words = [kw["keyword"] for kw in active_keywords]
            logger.info(f"Research Agent using {len(words)} user-injected keywords from Weekly Planner.")
            return words
        else:
            logger.info("No target keywords set. Falling back to complex internet knowledge exploration.")
            return random.sample(self.seed_words, min(3, len(self.seed_words)))

    def fetch_wikipedia_summaries(self, num_results: int = 3) -> List[str]:
        """Fetches random Wikipedia summaries as seed material."""
        summaries = []
        try:
            wikipedia.set_lang("en")
            words = self._get_search_words()
            
            for word in words[:num_results]:
                try:
                    search_results = wikipedia.search(word, results=3)
                    if not search_results:
                        continue
                    
                    page_title = random.choice(search_results)
                    summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
                    
                    text = f"Wiki Title: {page_title}. Summary: {summary}"
                    summaries.append(text)
                except wikipedia.exceptions.DisambiguationError as e:
                    logger.debug(f"Disambiguation hit for {word}, skipping.")
                except wikipedia.exceptions.PageError:
                    logger.debug(f"Page not found for {word}, skipping.")
                except Exception as e:
                    logger.error(f"Error fetching wiki summary for {word}: {e}")
                    
            logger.info(f"Fetched {len(summaries)} summaries from Wikipedia.")
        except Exception as e:
            logger.error(f"Meta error in fetch_wikipedia_summaries: {e}")
            
        return summaries

    def generate_and_store_topics(self) -> int:
        """
        Runs the research cycle: 
        1. Check for UI target keywords (prioritize) or fallback to Wikipedia
        2. Prompt LLM for JSON Topic
        3. Check Diversity (DB + Chroma)
        4. Store in DB
        """
        abstracts = self.fetch_wikipedia_summaries(num_results=3)
        inserted_count = 0
        
        for abstract in abstracts:
            # Check if the raw abstract itself is totally duplicate in vector db
            if not self.db.is_topic_novel(abstract, threshold=0.90): # higher threshold for raw text
                logger.info("Wiki topic similar to existing knowledge base, skipping generation.")
                continue
                
            prompt = f"Based on this informational excerpt, generate a highly engaging, thought-provoking conversation topic for an empathetic AI to discuss with a human.\n\nExcerpt:\n{abstract}"
            
            try:
                result_wrapper = self.llm.generate(prompt=prompt, system_message=RESEARCHER_SYSTEM_PROMPT)
                result_json = result_wrapper.get("data", {})
                usage = result_wrapper.get("usage", {})

                # Log cost for research
                if usage:
                    self.db.log_cost(
                        model=usage.get("model", "unknown"),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0)
                    )
                
                # Check required keys
                required_keys = ["topic_title", "topic_description"]
                if not all(k in result_json for k in required_keys):
                    logger.warning(f"LLM returned invalid Research JSON missing keys: {result_json}")
                    continue
                
                # Insert into database (this will also do the strict embedding/sha check)
                row_id = self.db.insert_seed_topic(topic_data=result_json, raw_text_for_embedding=abstract)
                if row_id:
                    inserted_count += 1
                    
            except Exception as e:
                logger.error(f"Error during research generation for abstract: {e}")
                
        logger.info(f"Research cycle completed. Inserted {inserted_count} new novel topics.")
        return inserted_count
