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

    def _get_search_items(self, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Combines active user-injected keywords (weighted by priority) with random seed words.
        Auto-deactivates keywords that have expired (past their week_label).
        """
        import datetime
        import re
        
        now = datetime.datetime.now()
        current_year = now.year
        # UI uses %U which is Sunday-based week number
        current_week = int(now.strftime("%U"))
        
        active_keywords = self.db.get_active_keywords()
        valid_keywords = []
        
        for kw in active_keywords:
            # Parse "Week 09 - 2026"
            week_label = kw.get("week_label", "")
            match = re.search(r"Week\s+(\d+)\s*-\s*(\d+)", week_label, re.IGNORECASE)
            if match:
                kw_week = int(match.group(1))
                kw_year = int(match.group(2))
                
                # If year is in the past, or same year but week is in the past -> Expired
                if kw_year < current_year or (kw_year == current_year and kw_week < current_week):
                    logger.info(f"Keyword '{kw['keyword']}' has expired (Target: Week {kw_week:02d} - {kw_year}, Current: Week {current_week:02d} - {current_year}). Deactivating.")
                    self.db.deactivate_keyword(kw["id"])
                    continue
                    
            valid_keywords.append(kw)
            
        pool = []
        
        # 1. Add valid keywords based on priority weight
        priority_weights = {"critical": 5, "high": 3, "normal": 1}
        for kw in valid_keywords:
            weight = priority_weights.get(kw.get("priority", "normal").lower(), 1)
            for _ in range(weight):
                pool.append({"id": kw["id"], "word": kw["keyword"], "is_target": True})
                
        # 2. Add random complex topics to maintain mixed diversity
        # If we have targets, we mix in some randoms. If we don't have targets, we fill entirely with randoms.
        num_random = max(1, num_results - 1) if valid_keywords else num_results
        
        # Handle case where sample is larger than population
        sample_size = min(num_random, len(self.seed_words))
        random_seeds = random.sample(self.seed_words, sample_size)
        
        for w in random_seeds:
            pool.append({"id": None, "word": w, "is_target": False})
            
        # 3. Shuffle the hybrid pool
        random.shuffle(pool)
        
        # 4. Pick top N
        selected = pool[:num_results]
        
        target_count = sum(1 for item in selected if item["is_target"])
        random_count = len(selected) - target_count
        logger.info(f"Research Search Mix: {target_count} target keywords, {random_count} random seeds.")
        
        return selected

    def fetch_wikipedia_summaries(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetches random Wikipedia summaries as seed material, returning dicts with keyword IDs."""
        summaries = []
        try:
            wikipedia.set_lang("en")
            
            for item in items:
                word = item["word"]
                logger.info(f"Querying Wikipedia API for: {word}")
                try:
                    search_results = wikipedia.search(word, results=10)
                    if not search_results:
                        continue
                    
                    page_title = random.choice(search_results)
                    
                    # Phase 12: Wikipedia Random Walk (Wiki-Jumping)
                    # 60% chance to jump to a random link on the page to prevent infinite duplicate loops
                    if random.random() < 0.60:
                        try:
                            # We must load the page to get its links
                            page = wikipedia.page(page_title, auto_suggest=False)
                            if page.links:
                                jumped_title = random.choice(page.links)
                                logger.info(f"Wiki-Jump! 🦘 Branching from '{page_title}' -> '{jumped_title}'")
                                page_title = jumped_title
                        except Exception as e:
                            logger.debug(f"Wiki-Jump failed for {page_title}, falling back to original. Reason: {e}")
                            
                    summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
                    
                    text = f"Wiki Title: {page_title}. Summary: {summary}"
                    summaries.append({
                        "text": text,
                        "keyword_id": item["id"],
                        "word": word,
                        "source": "wikipedia"
                    })
                except wikipedia.exceptions.DisambiguationError:
                    logger.debug(f"Disambiguation hit for {word}, skipping.")
                except wikipedia.exceptions.PageError:
                    logger.debug(f"Page not found for {word}, skipping.")
                except Exception as e:
                    logger.error(f"Error fetching wiki summary for {word}: {e}")
                    
            logger.info(f"Fetched {len(summaries)} summaries from Wikipedia.")
        except Exception as e:
            logger.error(f"Meta error in fetch_wikipedia_summaries: {e}")
            
        return summaries

    def fetch_arxiv_summaries(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetches highly technical academic abstracts from the public ArXiv API."""
        import urllib.request
        import xml.etree.ElementTree as ET
        import urllib.parse
        
        summaries = []
        for item in items:
            word = item["word"]
            logger.info(f"Querying ArXiv API for: {word}")
            # Format query for ArXiv API
            query = urllib.parse.quote(word)
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
            
            try:
                # Add a timeout to prevent hanging
                with urllib.request.urlopen(url, timeout=10) as response:
                    xml_data = response.read()
                    
                root = ET.fromstring(xml_data)
                
                # ArXiv XML namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    logger.debug(f"ArXiv returned no results for '{word}'.")
                    continue
                    
                # Pick a random paper from the top 5 results
                entry = random.choice(entries)
                title = entry.find('atom:title', ns).text.strip()
                # Clean up newlines in abstract
                abstract = " ".join(entry.find('atom:summary', ns).text.strip().split())
                
                text = f"ArXiv Paper: {title}. Abstract: {abstract}"
                summaries.append({
                    "text": text,
                    "keyword_id": item["id"],
                    "word": word,
                    "source": "arxiv"
                })
            except Exception as e:
                logger.error(f"Error fetching from ArXiv for '{word}': {e}")
                
        logger.info(f"Fetched {len(summaries)} summaries from ArXiv.")
        return summaries

    def generate_and_store_topics(self) -> int:
        """
        Runs the research cycle: 
        1. Check for UI target keywords (prioritize) or fallback to Wikipedia
        2. Route keywords to Wikipedia or ArXiv
        3. Prompt LLM for JSON Topic
        4. Check Diversity (DB + Chroma)
        5. Store in DB
        """
        items = self._get_search_items(num_results=3)
        
        # Route items to different sources (50/50 split)
        wiki_items = []
        arxiv_items = []
        
        for item in items:
            if random.random() < 0.50:
                arxiv_items.append(item)
            else:
                wiki_items.append(item)
                
        abstracts_data = []
        
        # Fetch ArXiv
        if arxiv_items:
            arxiv_results = self.fetch_arxiv_summaries(arxiv_items)
            abstracts_data.extend(arxiv_results)
            
            # Graceful Fallback: If ArXiv found nothing for a keyword, send it to Wiki
            successful_arxiv_words = {res["word"] for res in arxiv_results}
            for item in arxiv_items:
                if item["word"] not in successful_arxiv_words:
                    logger.info(f"Fallback: Sending '{item['word']}' to Wikipedia since ArXiv found nothing.")
                    wiki_items.append(item)
                    
        # Fetch Wiki
        if wiki_items:
            wiki_results = self.fetch_wikipedia_summaries(wiki_items)
            abstracts_data.extend(wiki_results)
            
        inserted_count = 0
        
        for data in abstracts_data:
            abstract = data["text"]
            kw_id = data["keyword_id"]
            
            # Check if the raw abstract itself is totally duplicate in vector db
            if not self.db.is_topic_novel(abstract, threshold=0.70): # higher threshold for raw text
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
                    # Phase 8: Target Keywords are no longer deactivated automatically here.
                    # They remain ACTIVE for continuous deep research until the week expires or manual deletion.
                    if kw_id is not None:
                        logger.info(f"✅ Topic generated for continuous target keyword ID {kw_id}.")
                    
            except Exception as e:
                logger.error(f"Error during research generation for abstract: {e}")
                
        logger.info(f"Research cycle completed. Inserted {inserted_count} new novel topics.")
        return inserted_count
