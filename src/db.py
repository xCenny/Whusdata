import sqlite3
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="pipeline.db"):
        self.db_path = db_path
        self._init_db()
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="seed_topics",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully.")
        except ImportError:
            logger.warning("chromadb not installed. Semantic search disabled.")
            self.chroma_client = None

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the database with WAL mode and all required tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS generations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT,
                        conversation_history TEXT,
                        persona_type TEXT,
                        conflict_type TEXT,
                        resolution_style TEXT,
                        difficulty_level TEXT,
                        domain TEXT,
                        tier INTEGER DEFAULT 0,
                        critic_status TEXT,
                        critic_confidence REAL,
                        memory_consistency_score REAL,
                        logic_score REAL,
                        winner TEXT DEFAULT 'Unknown',
                        failure_type TEXT DEFAULT 'NONE',
                        generation_mode TEXT DEFAULT 'production',
                        model_used TEXT DEFAULT 'unknown',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        sha256_hash TEXT UNIQUE
                    );
                """)
                
                # Graceful migration for existing DBs
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN model_used TEXT DEFAULT 'unknown';")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN logic_score REAL;")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN winner TEXT DEFAULT 'Unknown';")
                except sqlite3.OperationalError:
                    pass
                    
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cost_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model TEXT,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        cost_usd REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS seed_topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_title TEXT,
                        topic_description TEXT,
                        status TEXT DEFAULT 'PENDING',
                        hash TEXT UNIQUE,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    );
                """)
                cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('pipeline_status', 'running');")
                cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('calibration_mode', 'true');")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS target_keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT,
                        priority TEXT DEFAULT 'normal',
                        week_label TEXT,
                        status TEXT DEFAULT 'ACTIVE',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS failure_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT,
                        failure_type TEXT,
                        critic_confidence REAL,
                        critic_feedback TEXT,
                        reflection_count INTEGER DEFAULT 0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                logger.info("Database initialized successfully with WAL mode.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # ── Settings Helpers ──
    def get_setting(self, key: str) -> Optional[str]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_setting(self, key: str, value: str):
        with self.get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()

    # ── Target Keywords Helpers ──
    def add_target_keyword(self, keyword: str, priority: str = "normal", week_label: str = ""):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO target_keywords (keyword, priority, week_label) VALUES (?, ?, ?)",
                (keyword, priority, week_label)
            )
            conn.commit()

    def get_active_keywords(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM target_keywords WHERE status = 'ACTIVE' ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def deactivate_keyword(self, keyword_id: int):
        with self.get_connection() as conn:
            conn.execute("UPDATE target_keywords SET status = 'DONE' WHERE id = ?", (keyword_id,))
            conn.commit()

    # ── Generation Count Helper ──
    def get_generation_count(self) -> int:
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]

    # ── Dashboard Stats Helpers ──
    def get_dashboard_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM generations").fetchone()["c"]
            passed = conn.execute("SELECT COUNT(*) as c FROM generations WHERE critic_status = 'PASS'").fetchone()["c"]
            failed_total = conn.execute("SELECT COUNT(*) as c FROM generations WHERE critic_status = 'FAIL'").fetchone()["c"]
            avg_conf = conn.execute("SELECT AVG(critic_confidence) as a FROM generations WHERE critic_status = 'PASS'").fetchone()["a"] or 0.0
            avg_mem = conn.execute("SELECT AVG(memory_consistency_score) as a FROM generations WHERE critic_status = 'PASS'").fetchone()["a"] or 0.0
            pending_topics = conn.execute("SELECT COUNT(*) as c FROM seed_topics WHERE status = 'PENDING'").fetchone()["c"]
            t1 = conn.execute("SELECT COUNT(*) as c FROM generations WHERE tier = 1").fetchone()["c"]
            t2 = conn.execute("SELECT COUNT(*) as c FROM generations WHERE tier = 2").fetchone()["c"]
            t3 = conn.execute("SELECT COUNT(*) as c FROM generations WHERE tier = 3").fetchone()["c"]
            discarded = conn.execute("SELECT COUNT(*) as c FROM failure_log").fetchone()["c"]
            return {
                "total_generations": total,
                "passed": passed,
                "failed": failed_total,
                "pass_rate": round((passed / total * 100), 1) if total > 0 else 0.0,
                "avg_confidence": round(avg_conf, 3),
                "avg_memory_consistency": round(avg_mem, 3),
                "pending_topics": pending_topics,
                "tier_1": t1, "tier_2": t2, "tier_3": t3,
                "discarded": discarded
            }

    def get_failure_type_breakdown(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT failure_type, COUNT(*) as count FROM generations WHERE critic_status = 'FAIL' GROUP BY failure_type ORDER BY count DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_domain_breakdown(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT domain, COUNT(*) as count FROM generations WHERE critic_status = 'PASS' GROUP BY domain ORDER BY count DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_generations(self, limit: int = 20, status_filter: str = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM generations WHERE critic_status = ? ORDER BY timestamp DESC LIMIT ?",
                    (status_filter, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM generations ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def export_jsonl(self, tier_filter: int = None, domain_filter: str = None, difficulty_filter: str = None) -> List[Dict[str, Any]]:
        """Exports PASSED generations in rich SFT-ready JSONL format including metadata."""
        with self.get_connection() as conn:
            query = "SELECT * FROM generations WHERE critic_status = 'PASS'"
            params = []
            if tier_filter is not None:
                query += " AND tier = ?"
                params.append(tier_filter)
            if domain_filter:
                query += " AND domain = ?"
                params.append(domain_filter)
            if difficulty_filter:
                query += " AND difficulty_level = ?"
                params.append(difficulty_filter)
            query += " ORDER BY timestamp ASC"
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                try:
                    convo = json.loads(r["conversation_history"])
                    clean = []
                    for msg in convo:
                        if msg.get("role") == "assistant":
                            clean.append({"role": "assistant", "content": msg.get("content", "")})
                        else:
                            clean.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                            
                    # Construct rich dataset point
                    data_point = {
                        "topic": r["topic"],
                        "domain": r["domain"],
                        "difficulty": r["difficulty_level"],
                        "persona": r["persona_type"],
                        "scenario_conflict": r["conflict_type"],
                        "winner": r.get("winner", "Unknown"),
                        "logic_score": r.get("logic_score", 0.0),
                        "critic_confidence": r["critic_confidence"],
                        "memory_score": r["memory_consistency_score"],
                        "model_used": r.get("model_used", "unknown"),
                        "messages": clean
                    }
                    results.append(data_point)
                except json.JSONDecodeError:
                    continue
            return results

    @staticmethod
    def generate_hash(code: str) -> str:
        """Generates a SHA256 hash for the code to prevent exact duplicates."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()

    def insert_generation(
        self, 
        topic: str, 
        conversation_history: List[Dict[str, str]], 
        metadata: Dict[str, Any],
        critic_data: Dict[str, Any],
        tier: int = 0,
        mode: str = "production"
    ) -> Optional[int]:
        """Inserts a multi-turn generation with tier and mode classification."""
        convo_json = json.dumps(conversation_history, ensure_ascii=False)
        sha256_hash = self.generate_hash(convo_json)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO generations (
                        topic, conversation_history, 
                        persona_type, conflict_type, resolution_style, difficulty_level, domain,
                        tier, critic_status, critic_confidence, memory_consistency_score,
                        logic_score, winner, failure_type,
                        generation_mode, model_used, sha256_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    topic,
                    convo_json,
                    metadata.get("persona_type", "Unknown"),
                    metadata.get("conflict_type", "Unknown"),
                    metadata.get("resolution_style", "Unknown"),
                    metadata.get("difficulty_level", "Unknown"),
                    metadata.get("domain", "Unknown"),
                    tier,
                    critic_data.get("status", "FAIL"),
                    critic_data.get("confidence", 0.0),
                    critic_data.get("memory_consistency", 0.0),
                    critic_data.get("logic_score", 0.0),
                    critic_data.get("winner", "Unknown"),
                    critic_data.get("failure_type", "NONE"),
                    mode,
                    critic_data.get("model_used", "unknown"),
                    sha256_hash
                ))
                row_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Inserted Tier {tier} generation for topic: {topic}")
                return row_id
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate content detected for hash: {sha256_hash}. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error inserting into database: {e}")
            raise

    def log_failure(self, topic: str, failure_type: str, confidence: float, feedback: str, reflection_count: int = 0):
        """Logs a discarded/failed generation to the failure_log table."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT INTO failure_log (topic, failure_type, critic_confidence, critic_feedback, reflection_count) VALUES (?, ?, ?, ?, ?)",
                    (topic, failure_type, confidence, feedback, reflection_count)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging failure: {e}")

    def is_topic_novel(self, text: str, threshold: float = 0.70) -> bool:
        """
        Checks if the topic is novel using semantic search (cosine similarity).
        threshold 0.70 means if similarity is > 0.70, it's considered a duplicate.
        """
        if not self.chroma_client:
            return True # Fallback if no chromadb
            
        results = self.collection.query(
            query_texts=[text],
            n_results=1
        )
        
        # ChromaDB cosine returns distances (1 - cosine_similarity).
        # Distance 0 means identical, Distance > 0.30 means novel enough (similarity < 0.70)
        if results and results['distances'] and len(results['distances'][0]) > 0:
            distance = results['distances'][0][0]
            if distance < (1.0 - threshold):
                logger.warning(f"Topic not novel enough. Distance: {distance:.3f} (Required: >{1.0 - threshold:.3f})")
                return False
        return True

    def insert_seed_topic(self, topic_data: Dict[str, Any], raw_text_for_embedding: str = None) -> Optional[int]:
        """Inserts a new seed topic from the Research Agent."""
        try:
            topic_title = topic_data.get("topic_title", "")
            topic_desc = topic_data.get("topic_description", "")
            
            # Combine title and desc if raw text not provided
            embed_text = raw_text_for_embedding or f"{topic_title}. {topic_desc}"
            
            # Semantic search
            if not self.is_topic_novel(embed_text):
                return None
                
            # Literal hash
            topic_hash = self.generate_hash(embed_text)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO seed_topics (topic_title, topic_description, status, hash)
                    VALUES (?, ?, 'PENDING', ?)
                """, (
                    topic_title,
                    topic_desc,
                    topic_hash
                ))
                row_id = cursor.lastrowid
                conn.commit()
                
            # Save to ChromaDB for future lookups
            if self.chroma_client:
                self.collection.add(
                    documents=[embed_text],
                    metadatas=[{"title": topic_title}],
                    ids=[topic_hash]
                )
            
            logger.info(f"Inserted new seed topic: {topic_title}")
            return row_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Exact duplicate topic detected for hash: {topic_hash}. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error inserting seed topic: {e}")
            raise
            
    def get_pending_topic(self) -> Optional[Dict[str, Any]]:
        """Retrieves and locks a PENDING topic for the Teacher Agent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Find one pending
                cursor.execute("SELECT * FROM seed_topics WHERE status = 'PENDING' LIMIT 1")
                row = cursor.fetchone()
                
                if row:
                    # Optional: mark as PROCESSING or let Teacher mark it when done
                    # Assuming teacher does processing quickly in one script
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error fetching pending topic: {e}")
            return None
            
    def mark_topic_status(self, topic_id: int, status: str):
        """Updates the status of a seed topic (e.g., PROCESSED, FAILED)."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE seed_topics SET status = ? WHERE id = ?", (status, topic_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating topic status: {e}")

    # ── Phase 2: Distribution Control ──
    def get_persona_distribution(self, last_n: int = 500) -> Dict[str, Any]:
        """Returns persona type distribution for the last N generations + imbalance warning."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT persona_type, COUNT(*) as count FROM (SELECT persona_type FROM generations ORDER BY timestamp DESC LIMIT ?) GROUP BY persona_type ORDER BY count DESC",
                (last_n,)
            ).fetchall()
            total = sum(r["count"] for r in rows) if rows else 0
            distribution = []
            warning = None
            for r in rows:
                pct = round((r["count"] / total * 100), 1) if total > 0 else 0.0
                distribution.append({"persona_type": r["persona_type"], "count": r["count"], "pct": pct})
                if pct > 70.0:
                    warning = f"⚠️ IMBALANCE: '{r['persona_type']}' is {pct}% of last {last_n} records!"
            return {"distribution": distribution, "total": total, "warning": warning}

    def get_conflict_histogram(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT conflict_type, COUNT(*) as count FROM generations GROUP BY conflict_type ORDER BY count DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_drift_metrics(self) -> Dict[str, Any]:
        """Rolling 7-day PASS rate, average confidence trend, avg memory consistency."""
        with self.get_connection() as conn:
            # Rolling 7-day
            r7_total = conn.execute("SELECT COUNT(*) as c FROM generations WHERE timestamp >= datetime('now', '-7 days')").fetchone()["c"]
            r7_pass = conn.execute("SELECT COUNT(*) as c FROM generations WHERE critic_status='PASS' AND timestamp >= datetime('now', '-7 days')").fetchone()["c"]
            r7_avg_conf = conn.execute("SELECT AVG(critic_confidence) as a FROM generations WHERE critic_status='PASS' AND timestamp >= datetime('now', '-7 days')").fetchone()["a"] or 0.0
            r7_avg_mem = conn.execute("SELECT AVG(memory_consistency_score) as a FROM generations WHERE critic_status='PASS' AND timestamp >= datetime('now', '-7 days')").fetchone()["a"] or 0.0
            return {
                "rolling_7d_total": r7_total,
                "rolling_7d_pass": r7_pass,
                "rolling_7d_pass_rate": round((r7_pass / r7_total * 100), 1) if r7_total > 0 else 0.0,
                "rolling_7d_avg_confidence": round(r7_avg_conf, 3),
                "rolling_7d_avg_memory": round(r7_avg_mem, 3)
            }

    def log_cost(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Logs the token usage and estimated cost for an LLM call."""
        # Estimated costs per 1M tokens (very simplified, for tracking purposes)
        # Gemini 1.5 Pro: ~$3.5 (avg) | Groq: ~$0.5 (avg)
        cost_per_token = 3.5 / 1_000_000 if "gemini" in model.lower() else 0.5 / 1_000_000
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = total_tokens * cost_per_token
        
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT INTO cost_log (model, prompt_tokens, completion_tokens, cost_usd) VALUES (?, ?, ?, ?)",
                    (model, prompt_tokens, completion_tokens, cost_usd)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging cost: {e}")

    def get_daily_cost(self) -> float:
        """Returns the total cost for the last 24 hours."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT SUM(cost_usd) as total FROM cost_log WHERE timestamp >= datetime('now', '-1 day')"
            ).fetchone()
            return row["total"] if row and row["total"] else 0.0

    def get_total_cost(self) -> float:
        """Returns the total lifetime cost."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT SUM(cost_usd) as total FROM cost_log").fetchone()
            return row["total"] if row and row["total"] else 0.0
