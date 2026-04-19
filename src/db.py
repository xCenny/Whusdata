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
                        sha256_hash TEXT UNIQUE,
                        critic_analytics TEXT,
                        factual_score REAL DEFAULT 0.0
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
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN critic_model_used TEXT DEFAULT 'unknown';")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN critic_analytics TEXT;")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN factual_score REAL DEFAULT 0.0;")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN is_augmented BOOLEAN DEFAULT 0;")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN original_id INTEGER;")
                except sqlite3.OperationalError:
                    pass
                try:
                    cursor.execute("ALTER TABLE generations ADD COLUMN dataset_name TEXT DEFAULT 'default';")
                except sqlite3.OperationalError:
                    pass

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hf_export_targets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        repo_id TEXT,
                        hf_token TEXT,
                        tier_filter INTEGER,
                        domain_filter TEXT,
                        difficulty_filter TEXT,
                        dataset_filter TEXT,
                        is_active BOOLEAN DEFAULT 1
                    );
                """)
                try:
                    cursor.execute("ALTER TABLE hf_export_targets ADD COLUMN dataset_filter TEXT;")
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_health (
                        api_key TEXT PRIMARY KEY,
                        provider TEXT,
                        status TEXT DEFAULT 'ACTIVE',
                        last_error TEXT,
                        cooldown_until DATETIME
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider_base TEXT,
                        api_key TEXT UNIQUE,
                        is_free_tier BOOLEAN DEFAULT 0,
                        free_tier_delay INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT 1
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_sources (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        source_type TEXT,
                        config TEXT,
                        last_fetched_at DATETIME,
                        cooldown_minutes INTEGER DEFAULT 60,
                        is_active BOOLEAN DEFAULT 1
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_providers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        provider_base TEXT,
                        api_key_env_prefix TEXT,
                        base_url TEXT,
                        model_name TEXT,
                        role_tier TEXT,
                        is_free_tier BOOLEAN DEFAULT 0,
                        free_tier_delay INTEGER DEFAULT 0,
                        cost_input_1m REAL DEFAULT 0.0,
                        cost_output_1m REAL DEFAULT 0.0,
                        is_active BOOLEAN DEFAULT 1
                    );
                """)
                conn.commit()
                
                # Check if llm_providers is empty
                c = cursor.execute("SELECT COUNT(*) FROM llm_providers").fetchone()[0]
                if c == 0:
                    defaults = [
                        ("gemini-flash", "gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash", "fast", 1, 10, 0.15, 0.60, 1),
                        ("gemini-pro", "gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro", "reasoning", 1, 10, 2.50, 10.00, 1),
                        ("groq-fast", "groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant", "fast", 1, 15, 0.05, 0.08, 1),
                        ("groq-large", "groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "reasoning", 1, 15, 0.59, 0.79, 1),
                        ("openai-mini", "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini", "fast", 0, 0, 0.15, 0.60, 1),
                        ("openai-large", "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o", "reasoning", 0, 0, 2.50, 10.00, 1),
                        ("deepseek", "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat", "reasoning", 0, 0, 0.14, 0.28, 1)
                    ]
                    for d in defaults:
                        cursor.execute("""
                            INSERT INTO llm_providers (name, provider_base, api_key_env_prefix, base_url, model_name, role_tier, is_free_tier, free_tier_delay, cost_input_1m, cost_output_1m, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, d)
                    conn.commit()
                    conn.commit()

                # Check if knowledge_sources is empty
                c_ks = cursor.execute("SELECT COUNT(*) FROM knowledge_sources").fetchone()[0]
                if c_ks == 0:
                    import json
                    ks_defaults = [
                        ("Wikipedia Random", "wikipedia_random", "{}", 0, 1),
                        ("Reddit Tech", "reddit", json.dumps({"subreddits": ["Futurology", "MachineLearning", "AskScience"]}), 60, 1),
                        ("Hacker News Top", "hackernews", "{}", 60, 1),
                        ("Popular RSS Feeds", "rss", json.dumps({"feeds": ["http://rss.cnn.com/rss/edition_technology.rss", "https://feeds.bbci.co.uk/news/technology/rss.xml"]}), 120, 1)
                    ]
                    for ks in ks_defaults:
                        cursor.execute("""
                            INSERT INTO knowledge_sources (name, source_type, config, cooldown_minutes, is_active)
                            VALUES (?, ?, ?, ?, ?)
                        """, ks)
                    conn.commit()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS background_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_type TEXT,
                        status TEXT DEFAULT 'RUNNING',
                        progress INTEGER DEFAULT 0,
                        total INTEGER DEFAULT 0,
                        success_count INTEGER DEFAULT 0,
                        error_count INTEGER DEFAULT 0,
                        config TEXT,
                        result_message TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

                logger.info("Database initialized successfully with WAL mode.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # ── Knowledge Sources Helpers ──
    def get_knowledge_sources(self, active_only: bool = False) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = "SELECT * FROM knowledge_sources"
            if active_only:
                query += " WHERE is_active = 1"
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def update_knowledge_source(self, s_id: int, updates: Dict[str, Any]):
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [s_id]
        with self.get_connection() as conn:
            conn.execute(f"UPDATE knowledge_sources SET {fields} WHERE id = ?", values)
            conn.commit()

    def touch_knowledge_source(self, s_id: int):
        """Updates last_fetched_at to CURRENT_TIMESTAMP."""
        with self.get_connection() as conn:
            conn.execute("UPDATE knowledge_sources SET last_fetched_at = CURRENT_TIMESTAMP WHERE id = ?", (s_id,))
            conn.commit()

    # ── API Keys Helpers ──
    def get_api_keys(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM api_keys ORDER BY provider_base").fetchall()
            return [dict(r) for r in rows]

    def insert_api_key(self, data: Dict[str, Any]):
        fields = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        with self.get_connection() as conn:
            conn.execute(f"INSERT INTO api_keys ({fields}) VALUES ({placeholders})", values)
            conn.commit()

    def update_api_key(self, k_id: int, updates: Dict[str, Any]):
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [k_id]
        with self.get_connection() as conn:
            conn.execute(f"UPDATE api_keys SET {fields} WHERE id = ?", values)
            conn.commit()

    def delete_api_key(self, k_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM api_keys WHERE id = ?", (k_id,))
            conn.commit()

    # ── Dynamic Models Helpers ──
    def get_all_providers(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM llm_providers ORDER BY provider_base, role_tier").fetchall()
            return [dict(r) for r in rows]

    def update_provider(self, p_id: int, updates: Dict[str, Any]):
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [p_id]
        with self.get_connection() as conn:
            conn.execute(f"UPDATE llm_providers SET {fields} WHERE id = ?", values)
            conn.commit()

    def insert_provider(self, data: Dict[str, Any]):
        fields = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        with self.get_connection() as conn:
            conn.execute(f"INSERT INTO llm_providers ({fields}) VALUES ({placeholders})", values)
            conn.commit()

    def delete_provider(self, p_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM llm_providers WHERE id = ?", (p_id,))
            conn.commit()

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

    # ── API Health Helpers ──
    def update_api_health(self, api_key: str, provider: str, status: str, last_error: str = None, cooldown_until: str = None):
        """Upserts health status (ACTIVE, ERROR, COOLDOWN) for a specific API Key."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO api_health (api_key, provider, status, last_error, cooldown_until)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(api_key) DO UPDATE SET
                    status=excluded.status,
                    last_error=excluded.last_error,
                    cooldown_until=excluded.cooldown_until
            """, (api_key, provider, status, last_error, cooldown_until))
            conn.commit()

    def get_api_health(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dict of api_key -> health data for UI rendering."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM api_health").fetchall()
            return {r["api_key"]: dict(r) for r in rows}

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

    def get_daily_token_usage_chart(self, days: int = 14) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT DATE(timestamp) as date, SUM(prompt_tokens + completion_tokens) as total_tokens "
                "FROM cost_log "
                "WHERE timestamp >= date('now', ?) "
                "GROUP BY DATE(timestamp) ORDER BY date ASC",
                (f"-{days} days",)
            ).fetchall()
            return [dict(r) for r in rows]

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

    def export_jsonl(self, tier_filter: int = None, domain_filter: str = None, difficulty_filter: str = None, dataset_filter: str = None) -> List[Dict[str, Any]]:
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
            if dataset_filter:
                query += " AND dataset_name = ?"
                params.append(dataset_filter)
            query += " ORDER BY timestamp ASC"
            rows = conn.execute(query, params).fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                # Parse stored JSON strings
                try:
                    r["conversation_history"] = json.loads(r.get("conversation_history", "[]"))
                except (json.JSONDecodeError, TypeError):
                    r["conversation_history"] = []
                try:
                    r["critic_analytics"] = json.loads(r.get("critic_analytics", "{}"))
                except (json.JSONDecodeError, TypeError):
                    r["critic_analytics"] = {}
                    
                # Build clean SFT-ready output
                messages = []
                for msg in r["conversation_history"]:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                
                results.append({
                    "topic": r.get("topic", ""),
                    "domain": r.get("domain", ""),
                    "difficulty": r.get("difficulty_level", ""),
                    "persona": r.get("persona_type", ""),
                    "scenario_conflict": r.get("conflict_type", ""),
                    "winner": r.get("winner", ""),
                    "logic_score": r.get("logic_score", 0),
                    "factual_score": r.get("factual_score", 0),
                    "critic_confidence": r.get("critic_confidence", 0),
                    "memory_score": r.get("memory_consistency", 0),
                    "tier": r.get("tier", 0),
                    "model_used": r.get("model_used", ""),
                    "messages": messages,
                    "critic_analytics": r["critic_analytics"],
                    "dataset_name": r.get("dataset_name", "default")
                })
            return results

    def get_hf_targets(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM hf_export_targets ORDER BY id ASC").fetchall()
            return [dict(r) for r in rows]

    def insert_hf_target(self, data: Dict[str, Any]):
        fields = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        with self.get_connection() as conn:
            conn.execute(f"INSERT INTO hf_export_targets ({fields}) VALUES ({placeholders})", values)
            conn.commit()

    def update_hf_target(self, t_id: int, updates: Dict[str, Any]):
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [t_id]
        with self.get_connection() as conn:
            conn.execute(f"UPDATE hf_export_targets SET {fields} WHERE id = ?", values)
            conn.commit()

    def delete_hf_target(self, t_id: int):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM hf_export_targets WHERE id = ?", (t_id,))
            conn.commit()

    def get_unique_datasets(self) -> List[str]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT dataset_name FROM generations WHERE dataset_name IS NOT NULL AND dataset_name != ''").fetchall()
            return [r["dataset_name"] for r in rows] if rows else ["default"]

    def delete_generations(self, dataset_name: str, augmented_only: bool = False) -> int:
        with self.get_connection() as conn:
            sql = "DELETE FROM generations WHERE dataset_name = ?"
            if augmented_only:
                sql += " AND is_augmented = 1"
            cursor = conn.execute(sql, (dataset_name,))
            conn.commit()
            return cursor.rowcount

    def get_generations_for_augmentation(self, limit: int = 10, tier: int = 1, dataset_filter: str = None) -> List[Dict[str, Any]]:
        """Fetch base un-augmented data that hasn't been overly augmented."""
        with self.get_connection() as conn:
            sql = "SELECT * FROM generations WHERE critic_status = 'PASS' AND tier = ? AND is_augmented = 0"
            params = [tier]
            if dataset_filter:
                sql += " AND dataset_name = ?"
                params.append(dataset_filter)
            sql += " ORDER BY RANDOM() LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

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
        mode: str = "production",
        is_augmented: bool = False,
        original_id: int = None,
        dataset_name: str = "default"
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
                        generation_mode, model_used, critic_model_used, sha256_hash,
                        critic_analytics, factual_score, is_augmented, original_id, dataset_name
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    metadata.get("model_used", "unknown"),
                    critic_data.get("model_used", "unknown"),
                    sha256_hash,
                    json.dumps(critic_data.get("analytics", {}), ensure_ascii=False),
                    critic_data.get("factual_score", 0.0),
                    1 if is_augmented else 0,
                    original_id,
                    dataset_name
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
        """Logs the token usage and calculated cost for an LLM call using db pricing.
        Correctly handles free tiers by forcing cost_usd to 0.0."""
        try:
            with self.get_connection() as conn:
                # Fetch explicit pricing and free tier status from llm_providers
                row = conn.execute(
                    "SELECT cost_input_1m, cost_output_1m, is_free_tier FROM llm_providers WHERE model_name = ? COLLATE NOCASE LIMIT 1",
                    (model,)
                ).fetchone()
                
                if row and row["is_free_tier"]:
                    c_in = 0.0
                    c_out = 0.0
                elif row:
                    c_in = row["cost_input_1m"] / 1_000_000
                    c_out = row["cost_output_1m"] / 1_000_000
                else:
                    # Fallback approximation (assumes paid if not found)
                    c_in = 0.5 / 1_000_000
                    c_out = 0.5 / 1_000_000

                cost_usd = (prompt_tokens * c_in) + (completion_tokens * c_out)
                
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

    def get_ai_insights(self) -> Dict[str, Any]:
        """Calculates AI-driven insights based on generation history and costs."""
        with self.get_connection() as conn:
            total_cost = self.get_total_cost()
            total_gens = conn.execute("SELECT COUNT(*) as c FROM generations").fetchone()["c"]
            
            avg_cost = total_cost / total_gens if total_gens > 0 else 0.0
            
            # Estimate how many conversations you get for 1 million combined tokens at current mix
            total_tokens = conn.execute("SELECT SUM(prompt_tokens + completion_tokens) as t FROM cost_log").fetchone()["t"] or 0
            tokens_per_gen = total_tokens / total_gens if total_gens > 0 else 0
            convos_per_1m = int(1_000_000 / tokens_per_gen) if tokens_per_gen > 0 else 0
            
            est_1000 = avg_cost * 1000
            
            return {
                "avg_cost_per_gen": avg_cost,
                "convos_per_1m_tokens": convos_per_1m,
                "est_cost_1000": est_1000
            }

    # ── AI Re-Tagging Helpers ──
    def get_retag_stats(self) -> Dict[str, Any]:
        """Returns counts of records that need re-tagging (Unknown or empty metadata fields)."""
        with self.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM generations").fetchone()["c"]
            unknown_domain = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE domain IS NULL OR domain = '' OR domain = 'Unknown'"
            ).fetchone()["c"]
            unknown_persona = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE persona_type IS NULL OR persona_type = '' OR persona_type = 'Unknown'"
            ).fetchone()["c"]
            unknown_difficulty = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE difficulty_level IS NULL OR difficulty_level = '' OR difficulty_level = 'Unknown'"
            ).fetchone()["c"]
            unknown_conflict = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE conflict_type IS NULL OR conflict_type = '' OR conflict_type = 'Unknown'"
            ).fetchone()["c"]
            unknown_resolution = conn.execute(
                "SELECT COUNT(*) as c FROM generations WHERE resolution_style IS NULL OR resolution_style = '' OR resolution_style = 'Unknown'"
            ).fetchone()["c"]
            # Any record with at least one Unknown field
            any_unknown = conn.execute(
                """SELECT COUNT(*) as c FROM generations WHERE 
                   (domain IS NULL OR domain = '' OR domain = 'Unknown')
                   OR (persona_type IS NULL OR persona_type = '' OR persona_type = 'Unknown')
                   OR (difficulty_level IS NULL OR difficulty_level = '' OR difficulty_level = 'Unknown')
                   OR (conflict_type IS NULL OR conflict_type = '' OR conflict_type = 'Unknown')
                   OR (resolution_style IS NULL OR resolution_style = '' OR resolution_style = 'Unknown')
                """
            ).fetchone()["c"]
            return {
                "total": total,
                "unknown_domain": unknown_domain,
                "unknown_persona": unknown_persona,
                "unknown_difficulty": unknown_difficulty,
                "unknown_conflict": unknown_conflict,
                "unknown_resolution": unknown_resolution,
                "any_unknown": any_unknown
            }

    def get_generations_for_retagging(self, limit: int = 50, dataset_filter: str = None, 
                                       domain_filter: str = None, only_unknown: bool = True) -> List[Dict[str, Any]]:
        """Fetches generations that need re-tagging."""
        with self.get_connection() as conn:
            sql = "SELECT * FROM generations WHERE 1=1"
            params = []
            
            if only_unknown:
                sql += """ AND (
                    (domain IS NULL OR domain = '' OR domain = 'Unknown')
                    OR (persona_type IS NULL OR persona_type = '' OR persona_type = 'Unknown')
                    OR (difficulty_level IS NULL OR difficulty_level = '' OR difficulty_level = 'Unknown')
                    OR (conflict_type IS NULL OR conflict_type = '' OR conflict_type = 'Unknown')
                    OR (resolution_style IS NULL OR resolution_style = '' OR resolution_style = 'Unknown')
                )"""
            
            if dataset_filter:
                sql += " AND dataset_name = ?"
                params.append(dataset_filter)
            if domain_filter and domain_filter != "All":
                sql += " AND domain = ?"
                params.append(domain_filter)
                
            sql += " ORDER BY id ASC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def update_generation_tags(self, gen_id: int, new_tags: Dict[str, str]):
        """Updates metadata fields (domain, persona_type, etc.) for a single generation."""
        allowed_fields = {"domain", "persona_type", "conflict_type", "resolution_style", "difficulty_level"}
        updates = {k: v for k, v in new_tags.items() if k in allowed_fields and v}
        
        if not updates:
            return
            
        fields = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [gen_id]
        
        with self.get_connection() as conn:
            conn.execute(f"UPDATE generations SET {fields} WHERE id = ?", values)
            conn.commit()

    # ── Background Job Helpers ──
    def create_background_job(self, job_type: str, total: int, config: Dict = None) -> int:
        """Creates a new background job and returns its ID."""
        import json as _json
        with self.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO background_jobs (job_type, status, total, config) VALUES (?, 'RUNNING', ?, ?)",
                (job_type, total, _json.dumps(config or {}))
            )
            conn.commit()
            return cursor.lastrowid

    def update_background_job(self, job_id: int, progress: int = None, success_count: int = None, 
                               error_count: int = None, status: str = None, result_message: str = None):
        """Updates a background job's progress."""
        updates = []
        params = []
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if success_count is not None:
            updates.append("success_count = ?")
            params.append(success_count)
        if error_count is not None:
            updates.append("error_count = ?")
            params.append(error_count)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if result_message is not None:
            updates.append("result_message = ?")
            params.append(result_message)
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        
        with self.get_connection() as conn:
            conn.execute(f"UPDATE background_jobs SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

    def get_background_job(self, job_id: int) -> Dict[str, Any]:
        """Gets a single background job by ID."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM background_jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else {}

    def get_active_background_jobs(self, job_type: str = None) -> List[Dict[str, Any]]:
        """Gets all active (RUNNING) background jobs, optionally filtered by type."""
        with self.get_connection() as conn:
            if job_type:
                rows = conn.execute(
                    "SELECT * FROM background_jobs WHERE status = 'RUNNING' AND job_type = ? ORDER BY created_at DESC", 
                    (job_type,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM background_jobs WHERE status = 'RUNNING' ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_background_jobs(self, job_type: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Gets recent background jobs for display."""
        with self.get_connection() as conn:
            if job_type:
                rows = conn.execute(
                    "SELECT * FROM background_jobs WHERE job_type = ? ORDER BY created_at DESC LIMIT ?",
                    (job_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM background_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

