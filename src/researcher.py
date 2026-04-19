import logging
import wikipedia
import random
from typing import List, Dict, Any
from src.db import DatabaseManager
from src.llm_client import LLMClient
from src.prompts import RESEARCHER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Domain-specific seed word pools for topic focus
DOMAIN_SEED_WORDS = {
    "fizik": [
        "Quantum entanglement paradoxes", "Thermodynamic entropy paradoxes", "Dark matter detection methods",
        "Non-Euclidean geometry principles", "Theoretical astrophysics anomalies", "Standard Model limitations",
        "General relativity edge cases", "Higgs boson implications", "Quantum chromodynamics",
        "Particle accelerator physics", "String theory criticisms", "Wave-particle duality"
    ],
    "physics": [
        "Quantum entanglement paradoxes", "Thermodynamic entropy paradoxes", "Dark matter detection methods",
        "Non-Euclidean geometry principles", "Theoretical astrophysics anomalies", "Standard Model limitations",
        "General relativity edge cases", "Higgs boson implications", "Quantum chromodynamics",
        "Particle accelerator physics", "String theory criticisms", "Wave-particle duality"
    ],
    "teknoloji": [
        "Cryptographic hash collisions", "Advanced robotics kinematics", "Turing machine limitations",
        "CRISPR off-target effects", "Neural network interpretability", "Autonomous vehicle ethics",
        "Blockchain scalability trilemma", "Quantum computing error correction", "LLM hallucination taxonomy",
        "Brain-computer interface risks", "Zero-knowledge proof systems", "Homomorphic encryption challenges"
    ],
    "technology": [
        "Cryptographic hash collisions", "Advanced robotics kinematics", "Turing machine limitations",
        "CRISPR off-target effects", "Neural network interpretability", "Autonomous vehicle ethics",
        "Blockchain scalability trilemma", "Quantum computing error correction", "LLM hallucination taxonomy",
        "Brain-computer interface risks", "Zero-knowledge proof systems", "Homomorphic encryption challenges"
    ],
    "biyoloji": [
        "Epigenetic inheritance", "Microbiome symbiosis", "CRISPR off-target effects",
        "Neuroplasticity in adult brains", "Horizontal gene transfer", "Endosymbiotic theory controversies",
        "Prion disease mechanisms", "Synthetic biology ethics", "Telomere biology and aging"
    ],
    "biology": [
        "Epigenetic inheritance", "Microbiome symbiosis", "CRISPR off-target effects",
        "Neuroplasticity in adult brains", "Horizontal gene transfer", "Endosymbiotic theory controversies",
        "Prion disease mechanisms", "Synthetic biology ethics", "Telomere biology and aging"
    ],
    "felsefe": [
        "Post-structuralism concepts", "The Fermi Paradox solutions", "Linguistic relativity",
        "Consciousness hard problem", "Free will compatibilism", "Epistemological skepticism",
        "Trolley problem variations", "Philosophy of mind dualism", "Existentialism vs nihilism"
    ],
    "philosophy": [
        "Post-structuralism concepts", "The Fermi Paradox solutions", "Linguistic relativity",
        "Consciousness hard problem", "Free will compatibilism", "Epistemological skepticism",
        "Trolley problem variations", "Philosophy of mind dualism", "Existentialism vs nihilism"
    ],
    "tarih": [
        "Bronze Age Collapse mathematics", "Ancient metallurgy techniques", "Geopolitical choke points",
        "Paleoclimatology data", "Socio-cultural evolution theories", "Ottoman decline theories",
        "Industrial revolution social impacts", "Cold War proxy conflicts", "Ancient trade routes"
    ],
    "history": [
        "Bronze Age Collapse mathematics", "Ancient metallurgy techniques", "Geopolitical choke points",
        "Paleoclimatology data", "Socio-cultural evolution theories", "Ottoman decline theories",
        "Industrial revolution social impacts", "Cold War proxy conflicts", "Ancient trade routes"
    ],
    "ekonomi": [
        "Behavioral economics heuristics", "Modern monetary theory debates", "Cryptocurrency regulation paradox",
        "Supply-side vs demand-side economics", "Gini coefficient limitations", "Dutch disease economics"
    ],
    "economics": [
        "Behavioral economics heuristics", "Modern monetary theory debates", "Cryptocurrency regulation paradox",
        "Supply-side vs demand-side economics", "Gini coefficient limitations", "Dutch disease economics"
    ],
    "matematik": [
        "Non-Euclidean geometry principles", "Gödel incompleteness theorems", "Riemann hypothesis implications",
        "Chaos theory applications", "Fractal geometry in nature", "P vs NP problem"
    ],
    "mathematics": [
        "Non-Euclidean geometry principles", "Gödel incompleteness theorems", "Riemann hypothesis implications",
        "Chaos theory applications", "Fractal geometry in nature", "P vs NP problem"
    ],
    "psikoloji": [
        "Neuroplasticity in adult brains", "Behavioral economics heuristics", "Cognitive dissonance theory",
        "Stockholm syndrome debates", "Dark triad personality traits", "Milgram experiment ethics"
    ],
    "psychology": [
        "Neuroplasticity in adult brains", "Behavioral economics heuristics", "Cognitive dissonance theory",
        "Stockholm syndrome debates", "Dark triad personality traits", "Milgram experiment ethics"
    ]
}

class ResearchAgent:
    def __init__(self, db_manager: DatabaseManager, llm_client: LLMClient):
        self.db = db_manager
        self.llm = llm_client
        # Deep internet knowledge seeds to trigger complex Wikipedia exploration
        self.default_seed_words = [
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
                
        # 2. Build seed words: use focused seeds if topic_focus is active, otherwise default
        active_seeds = self._get_focused_seed_words()
        
        # If we have targets, we mix in some randoms. If we don't have targets, we fill entirely with randoms.
        num_random = max(1, num_results - 1) if valid_keywords else num_results
        
        # Handle case where sample is larger than population
        sample_size = min(num_random, len(active_seeds))
        random_seeds = random.sample(active_seeds, sample_size)
        
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

    def _get_focused_seed_words(self) -> List[str]:
        """Returns seed words biased towards focus domains if set, otherwise returns default seeds."""
        topic_focus = self.db.get_setting("topic_focus") or ""
        if not topic_focus.strip():
            return self.default_seed_words
        
        # Parse comma-separated domains
        domains = [d.strip().lower() for d in topic_focus.split(",") if d.strip()]
        if not domains:
            return self.default_seed_words
        
        # Collect seeds from matching domain pools
        focused_seeds = []
        for domain in domains:
            if domain in DOMAIN_SEED_WORDS:
                focused_seeds.extend(DOMAIN_SEED_WORDS[domain])
        
        # Deduplicate
        focused_seeds = list(set(focused_seeds))
        
        if not focused_seeds:
            # Unknown domain — fall back to defaults but log it
            logger.warning(f"Topic focus domains {domains} have no predefined seeds. Using defaults.")
            return self.default_seed_words
        
        # Mix in ~20% default seeds for diversity even when focused
        num_mix = max(2, len(focused_seeds) // 5)
        sample_size = min(num_mix, len(self.default_seed_words))
        mix_seeds = random.sample(self.default_seed_words, sample_size)
        focused_seeds.extend(mix_seeds)
        focused_seeds = list(set(focused_seeds))
        
        logger.info(f"🎯 Topic Focus active: {domains} → {len(focused_seeds)} focused seed words")
        return focused_seeds

    def fetch_wikipedia_summaries(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetches standard Wikipedia summaries based on search keywords."""
        summaries = []
        try:
            wikipedia.set_lang("en")
            for item in items:
                word = item["word"]
                logger.info(f"Querying Wikipedia Search API for: {word}")
                try:
                    search_results = wikipedia.search(word, results=10)
                    if not search_results:
                        continue
                    
                    page_title = random.choice(search_results)
                    
                    if random.random() < 0.60:
                        try:
                            page = wikipedia.page(page_title, auto_suggest=False)
                            if page.links:
                                page_title = random.choice(page.links)
                        except Exception:
                            pass
                            
                    summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
                    text = f"Wiki Title: {page_title}. Summary: {summary}"
                    summaries.append({
                        "text": text,
                        "keyword_id": item["id"],
                        "word": word,
                        "source": "wikipedia_search"
                    })
                except Exception as e:
                    logger.debug(f"Wiki fetch skipped for {word}: {e}")
        except Exception as e:
            logger.error(f"Meta error in fetch_wikipedia_summaries: {e}")
        return summaries

    def fetch_wikipedia_random(self) -> List[Dict[str, Any]]:
        """Fetches completely random Wikipedia articles."""
        summaries = []
        try:
            wikipedia.set_lang("en")
            # Grab up to 3 random titles
            random_titles = wikipedia.random(3)
            if isinstance(random_titles, str):
                random_titles = [random_titles]
                
            for title in random_titles:
                logger.info(f"Querying Wikipedia Random for: {title}")
                try:
                    summary = wikipedia.summary(title, sentences=5, auto_suggest=False)
                    text = f"Random Wiki Title: {title}. Summary: {summary}"
                    summaries.append({
                        "text": text,
                        "keyword_id": None,
                        "word": title,
                        "source": "wikipedia_random"
                    })
                except Exception as e:
                    logger.debug(f"Random wiki fetch failed for {title}: {e}")
        except Exception as e:
            logger.error(f"Error in fetch_wikipedia_random: {e}")
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

    def fetch_rss_feeds(self, config_str: str) -> List[Dict[str, Any]]:
        import json
        import feedparser
        summaries = []
        try:
            config = json.loads(config_str)
            feeds = config.get("feeds", [])
            for feed_url in feeds:
                logger.info(f"Querying RSS Feed: {feed_url}")
                parsed = feedparser.parse(feed_url)
                if not parsed.entries:
                    continue
                # Pick up to 3 random recent entries
                entries = random.sample(parsed.entries, min(3, len(parsed.entries)))
                for entry in entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"RSS News: {title}. Excerpt: {summary}"
                    summaries.append({
                        "text": text,
                        "keyword_id": None,
                        "word": title,
                        "source": "rss"
                    })
        except Exception as e:
            logger.error(f"Error fetching RSS: {e}")
        return summaries

    def fetch_reddit(self, config_str: str) -> List[Dict[str, Any]]:
        import json
        import requests
        summaries = []
        headers = {"User-Agent": "Python:WhusdataResearchPipeline:v1.0 (by /u/xCenny)"}
        try:
            config = json.loads(config_str)
            subreddits = config.get("subreddits", [])
            if subreddits:
                sub = random.choice(subreddits)
                logger.info(f"Querying Reddit: r/{sub}")
                resp = requests.get(f"https://www.reddit.com/r/{sub}/hot.json?limit=10", headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])
                    # Pick 2 random posts
                    selected = random.sample(posts, min(2, len(posts)))
                    for p in selected:
                        pdata = p.get("data", {})
                        title = pdata.get("title", "")
                        selftext = pdata.get("selftext", "")
                        text = f"Reddit Post (r/{sub}): {title}. Body: {selftext[:1000]}"
                        summaries.append({
                            "text": text,
                            "keyword_id": None,
                            "word": title,
                            "source": "reddit"
                        })
                else:
                    logger.warning(f"Reddit API returned {resp.status_code} for r/{sub} (Possibly rate limited)")
        except Exception as e:
            logger.error(f"Error fetching Reddit: {e}")
        return summaries

    def fetch_hackernews(self) -> List[Dict[str, Any]]:
        import requests
        summaries = []
        try:
            logger.info("Querying Hacker News Top Stories")
            resp = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
            if resp.status_code == 200:
                story_ids = resp.json()[:20] # top 20
                if story_ids:
                    s_id = random.choice(story_ids)
                    s_resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{s_id}.json", timeout=10)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json()
                        title = s_data.get("title", "")
                        text = f"Hacker News Discussion: {title}."
                        
                        # Fetch top comments to get arguments
                        kids = s_data.get("kids", [])[:3]
                        for kid in kids:
                            c_resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{kid}.json", timeout=5)
                            if c_resp.status_code == 200:
                                c_data = c_resp.json()
                                c_text = c_data.get("text", "")
                                if c_text:
                                    text += f" Comment: {c_text}"
                                    
                        # Strip html tags slightly
                        import re
                        text = re.sub('<[^<]+?>', '', text)
                        
                        summaries.append({
                            "text": text,
                            "keyword_id": None,
                            "word": title,
                            "source": "hackernews"
                        })
        except Exception as e:
            logger.error(f"Error fetching Hacker News: {e}")
        return summaries

    def generate_and_store_topics(self) -> int:
        """
        Runs the research cycle using dynamic DB sources.
        """
        import datetime
        
        # 1. Fetch available sources that are OFF cooldown
        all_sources = self.db.get_knowledge_sources(active_only=True)
        available_sources = []
        now = datetime.datetime.now()
        
        for src in all_sources:
            last_fetched = src.get("last_fetched_at")
            cooldown = src.get("cooldown_minutes", 60)
            
            if not last_fetched:
                available_sources.append(src)
                continue
                
            try:
                lf_dt = datetime.datetime.strptime(last_fetched, "%Y-%m-%d %H:%M:%S")
                if (now - lf_dt).total_seconds() / 60 >= cooldown:
                    available_sources.append(src)
                else:
                    logger.debug(f"Source '{src['name']}' is on cooldown.")
            except Exception:
                available_sources.append(src) # Unparsable, let's just use it
                
        if not available_sources:
            logger.warning("No Knowledge Sources available (all on cooldown or inactive). Falling back to basic Wikipedia Search.")
            # Create a mock source to force fallback
            available_sources = [{"id": 0, "source_type": "wikipedia_search", "config": "{}"}]
            
        # 2. Pick a random source to use for this research cycle
        selected_source = random.choice(available_sources)
        stype = selected_source["source_type"]
        sconfig = selected_source.get("config", "{}")
        s_id = selected_source.get("id", 0)
        
        logger.info(f"🎯 Research cycle selected knowledge source: {stype.upper()}")
        
        if s_id > 0:
            self.db.touch_knowledge_source(s_id)
            
        abstracts_data = []

        # 3. Route to specific fetcher based on source type
        if stype == "wikipedia_random":
            abstracts_data = self.fetch_wikipedia_random()
        elif stype == "rss":
            abstracts_data = self.fetch_rss_feeds(sconfig)
        elif stype == "reddit":
            abstracts_data = self.fetch_reddit(sconfig)
        elif stype == "hackernews":
            abstracts_data = self.fetch_hackernews()
        else: # wikipedia_search or arxiv need keywords
            items = self._get_search_items(num_results=3)
            if stype == "arxiv":
                abstracts_data = self.fetch_arxiv_summaries(items)
                if not abstracts_data:
                    logger.info("ArXiv yielded nothing. Falling back to Wiki Search.")
                    abstracts_data = self.fetch_wikipedia_summaries(items)
            else:
                abstracts_data = self.fetch_wikipedia_summaries(items)

        # Build topic focus block for system prompt
        topic_focus = self.db.get_setting("topic_focus") or ""
        topic_focus_instructions = self.db.get_setting("topic_focus_instructions") or ""
        
        if topic_focus.strip():
            focus_block = f"\nRule 3 (TOPIC FOCUS — CRITICAL): You MUST generate topics within these domains: {topic_focus}. "
            focus_block += "Do NOT generate topics outside of these focus areas. Every generated topic MUST be directly related to at least one of the specified domains."
            if topic_focus_instructions.strip():
                focus_block += f"\nAdditional Focus Instructions: {topic_focus_instructions}"
            focus_block += "\n"
        else:
            focus_block = "\n"
        
        formatted_system_prompt = RESEARCHER_SYSTEM_PROMPT.format(topic_focus_block=focus_block)
        
        inserted_count = 0
        
        for data in abstracts_data:
            abstract = data["text"]
            kw_id = data.get("keyword_id")
            
            if not self.db.is_topic_novel(abstract, threshold=0.70):
                logger.info(f"{stype} topic similar to existing knowledge base, skipping generation.")
                continue
                
            prompt = f"Based on this informational excerpt from {stype}, generate a highly engaging, thought-provoking conversation topic for an empathetic AI to discuss with a human.\n\nExcerpt:\n{abstract}"
            
            try:
                result_wrapper = self.llm.generate(prompt=prompt, system_message=formatted_system_prompt)
                result_json = result_wrapper.get("data", {})
                usage = result_wrapper.get("usage", {})

                if usage:
                    self.db.log_cost(
                        model=usage.get("model", "unknown"),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0)
                    )
                
                required_keys = ["topic_title", "topic_description"]
                if not all(k in result_json for k in required_keys):
                    logger.warning(f"LLM returned invalid Research JSON missing keys: {result_json}")
                    continue
                
                row_id = self.db.insert_seed_topic(topic_data=result_json, raw_text_for_embedding=abstract)
                if row_id:
                    inserted_count += 1
                    if kw_id is not None:
                        logger.info(f"✅ Topic generated for target keyword ID {kw_id}.")
                    
            except Exception as e:
                logger.error(f"Error during research generation for abstract: {e}")
                
        logger.info(f"Research cycle completed using '{stype}'. Inserted {inserted_count} new novel topics.")
        return inserted_count
