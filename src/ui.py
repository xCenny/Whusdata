import streamlit as st
import json
import pandas as pd
from datetime import datetime
from src.db import DatabaseManager

# ── Page Config ──
st.set_page_config(
    page_title="Whusdata — Data Pipeline Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460; border-radius: 16px; padding: 20px; text-align: center; color: #e0e0e0;
    }
    .metric-card h2 { font-size: 2rem; font-weight: 700; margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card p { font-size: 0.8rem; color: #8892b0; margin: 4px 0 0 0; }
    .tier-gold { border-left: 4px solid #FFD700; }
    .tier-silver { border-left: 4px solid #C0C0C0; }
    .tier-bronze { border-left: 4px solid #CD7F32; }
    .pass-badge { background: #0d7377; color: #00ffd5; padding: 3px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
    .fail-badge { background: #6b2737; color: #ff6b6b; padding: 3px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a23 0%, #1a1a3e 100%); }
    .convo-bubble-user { background: #1e3a5f; border-radius: 12px; padding: 12px 16px; margin: 6px 0; border-left: 3px solid #3a7bd5; }
    .convo-bubble-assistant { background: #1a3a2a; border-radius: 12px; padding: 12px 16px; margin: 6px 0; border-left: 3px solid #00d2ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/nolan/64/artificial-intelligence.png", width=50)
st.sidebar.title("🧠 Whusdata Pipeline")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "📈 Drift Monitor", "💬 Conversations", "🎯 Weekly Planner", "⚙️ Pipeline Control", "🤖 Models & Prices", "🔑 API Keys", "📚 Knowledge Sources", "📥 Export Dataset"],
    label_visibility="collapsed"
)
pipeline_status = db.get_setting("pipeline_status") or "running"
cal_mode = db.get_setting("calibration_mode") or "false"
st.sidebar.markdown(f"**Pipeline:** {'🟢' if pipeline_status == 'running' else '🔴'} `{pipeline_status.upper()}`")
if cal_mode == "true":
    st.sidebar.markdown("**Mode:** 🧪 `CALIBRATION`")
st.sidebar.markdown("---")

# ═══════════════════════════════════════════════
# 📊 DASHBOARD
# ═══════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Pipeline Dashboard")
    stats = db.get_dashboard_stats()
    
    # Row 1: Main Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f'<div class="metric-card"><h2>{stats["total_generations"]}</h2><p>Total Generations</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h2>{stats["passed"]}</h2><p>✅ PASS</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h2>{stats["failed"]}</h2><p>❌ FAIL</p></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><h2>{stats.get("pass_rate", 0)}%</h2><p>Pass Rate</p></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-card"><h2>{stats["discarded"]}</h2><p>🗑 Discarded</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row: Cost Stats
    daily_cost = db.get_daily_cost()
    total_cost = db.get_total_cost()
    co1, co2, co3 = st.columns([1, 1, 2])
    with co1: st.metric("Daily Cost (24h)", f"${daily_cost:.4f}", delta=f"{ (daily_cost/10.0)*100:.1f}% of budget")
    with co2: st.metric("Total Pipeline Cost", f"${total_cost:.4f}")
    with co3:
        st.caption("Daily Budget Usage ($10.00 Limit)")
        st.progress(min(daily_cost / 10.0, 1.0))

    st.markdown("---")
    
    ai = db.get_ai_insights()
    st.markdown("### 🤖 Minimum Viable AI Insights")
    ai1, ai2, ai3 = st.columns(3)
    ai1.metric("Avg Cost / Convo", f"${ai.get('avg_cost_per_gen', 0):.4f}")
    ai2.metric("Convos per 1M Tokens", f"{ai.get('convos_per_1m_tokens', 0):,}")
    ai3.metric("Est. Cost for 1k More", f"${ai.get('est_cost_1000', 0):.2f}")
    
    st.markdown("---")
    
    # Row 2: Tier Breakdown + Quality
    t1, t2, t3, t4 = st.columns(4)
    with t1: st.markdown(f'<div class="metric-card tier-gold"><h2>{stats["tier_1"]}</h2><p>🥇 Tier 1 (Gold ≥0.85)</p></div>', unsafe_allow_html=True)
    with t2: st.markdown(f'<div class="metric-card tier-silver"><h2>{stats["tier_2"]}</h2><p>🥈 Tier 2 (Silver ≥0.70)</p></div>', unsafe_allow_html=True)
    with t3: st.markdown(f'<div class="metric-card tier-bronze"><h2>{stats["tier_3"]}</h2><p>🥉 Tier 3 (Bronze ≥0.65)</p></div>', unsafe_allow_html=True)
    with t4: st.markdown(f'<div class="metric-card"><h2>{stats["avg_confidence"]}</h2><p>Avg Confidence</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📊 14-Day Token Usage Tracker")
    token_chart_data = db.get_daily_token_usage_chart(days=14)
    if token_chart_data:
        df_tokens = pd.DataFrame(token_chart_data).set_index("date")
        st.bar_chart(df_tokens["total_tokens"], color="#00d2ff")
    else:
        st.info("No token usage data recorded yet.")
        
    st.markdown("---")
    
    # Row 3: Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("🏷 Domain Distribution")
        domain_data = db.get_domain_breakdown()
        if domain_data:
            st.bar_chart(pd.DataFrame(domain_data).set_index("domain")["count"])
        else:
            st.info("No data yet.")
    with ch2:
        st.subheader("⚠️ Failure Type Breakdown")
        failure_data = db.get_failure_type_breakdown()
        if failure_data:
            st.bar_chart(pd.DataFrame(failure_data).set_index("failure_type")["count"])
        else:
            st.info("No failures recorded.")
    
    # Persona Distribution Warning
    dist = db.get_persona_distribution()
    if dist.get("warning"):
        st.error(dist["warning"])
    if dist["distribution"]:
        st.subheader("🎭 Persona Distribution (Last 500)")
        st.bar_chart(pd.DataFrame(dist["distribution"]).set_index("persona_type")["pct"])

# ═══════════════════════════════════════════════
# 📈 DRIFT MONITOR
# ═══════════════════════════════════════════════
elif page == "📈 Drift Monitor":
    st.title("📈 Drift Monitor — 7-Day Rolling")
    st.caption("Track quality trends to detect pipeline degradation early")
    
    drift = db.get_drift_metrics()
    
    d1, d2, d3 = st.columns(3)
    d1.metric("7-Day Pass Rate", f"{drift['rolling_7d_pass_rate']}%", delta=f"{drift['rolling_7d_pass']}/{drift['rolling_7d_total']} total")
    d2.metric("7-Day Avg Confidence", f"{drift['rolling_7d_avg_confidence']}")
    d3.metric("7-Day Avg Memory Consistency", f"{drift['rolling_7d_avg_memory']}")
    
    st.markdown("---")
    st.subheader("🎭 Conflict Type Histogram")
    conflict_data = db.get_conflict_histogram()
    if conflict_data:
        st.bar_chart(pd.DataFrame(conflict_data).set_index("conflict_type")["count"])
    else:
        st.info("No conflict data yet.")

# ═══════════════════════════════════════════════
# 💬 CONVERSATIONS
# ═══════════════════════════════════════════════
elif page == "💬 Conversations":
    st.title("💬 Conversation Browser")
    
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        status_filter = st.selectbox("Critic Status", ["All", "PASS", "FAIL"])
    
    sf = status_filter if status_filter != "All" else None
    conversations = db.get_recent_generations(limit=50, status_filter=sf)
    
    if not conversations:
        st.info("No conversations generated yet.")
    
    for convo in conversations:
        badge = '<span class="pass-badge">PASS</span>' if convo.get("critic_status") == "PASS" else '<span class="fail-badge">FAIL</span>'
        conf = convo.get("critic_confidence", 0.0) or 0.0
        tier = convo.get("tier", 0)
        tier_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(tier, "⚪")
        
        mode = convo.get("generation_mode", "production")
        mode_label = "🧪 CAL" if mode == "calibration" else "🚀 PROD"
        
        with st.expander(f"{tier_emoji} #{convo['id']} {mode_label} — {convo.get('topic', 'N/A')[:55]}... | Conf: {conf:.2f}"):
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            tc1.markdown(f"**Persona:** `{convo.get('persona_type', '-')}`")
            tc2.markdown(f"**Conflict:** `{convo.get('conflict_type', '-')}`")
            tc3.markdown(f"**Domain:** `{convo.get('domain', '-')}`")
            tc4.markdown(f"**Tier:** `{tier}`")
            tc5.markdown(f"**🤖 Gen:** `{convo.get('model_used', 'unknown')}` | **⚖️ Critic:** `{convo.get('critic_model_used', 'unknown')}`")
            st.markdown("---")
            try:
                history = json.loads(convo.get("conversation_history", "[]"))
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        st.markdown(f'<div class="convo-bubble-user">🧑 <strong>User:</strong><br>{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="convo-bubble-assistant">🤖 <strong>Assistant:</strong><br>{content}</div>', unsafe_allow_html=True)
                        reasoning = msg.get("reasoning", "")
                        if reasoning:
                            with st.popover("🧠 Reasoning"):
                                st.markdown(reasoning)
            except json.JSONDecodeError:
                st.error("Could not parse conversation.")

# ═══════════════════════════════════════════════
# 🎯 WEEKLY PLANNER
# ═══════════════════════════════════════════════
elif page == "🎯 Weekly Planner":
    st.title("🎯 Weekly Focus Planner")
    
    with st.form("add_kw", clear_on_submit=True):
        st.subheader("➕ Add Target Keyword")
        kc1, kc2, kc3 = st.columns([3, 1, 1])
        keyword = kc1.text_input("Keyword", placeholder="e.g., Quantum Computing")
        priority = kc2.selectbox("Priority", ["normal", "high", "critical"])
        week_label = kc3.text_input("Week", value=datetime.now().strftime("Week %U - %Y"))
        if st.form_submit_button("🚀 Add", use_container_width=True) and keyword.strip():
            db.add_target_keyword(keyword.strip(), priority, week_label)
            st.success(f"Added '{keyword}'!")
            st.rerun()
    
    st.markdown("---")
    st.subheader("📋 Active Keywords")
    keywords = db.get_active_keywords()
    if not keywords:
        st.info("No active keywords. Research Agent uses random Wikipedia.")
    else:
        for kw in keywords:
            kc1, kc2, kc3 = st.columns([4, 1, 1])
            emoji = {"critical": "🔴", "high": "🟠", "normal": "🟢"}.get(kw["priority"], "⚪")
            kc1.markdown(f"{emoji} **{kw['keyword']}** — `{kw.get('week_label', '')}`")
            kc2.caption(kw["priority"])
            if kc3.button("✅", key=f"d_{kw['id']}"):
                db.deactivate_keyword(kw["id"])
                st.rerun()

# ═══════════════════════════════════════════════
# ⚙️ PIPELINE CONTROL
# ═══════════════════════════════════════════════
elif page == "⚙️ Pipeline Control":
    st.title("⚙️ Pipeline Control")
    cur = db.get_setting("pipeline_status") or "running"
    st.subheader(f"Status: {'🟢 Running' if cur == 'running' else '🔴 Paused'}")
    
    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("▶️ Start", use_container_width=True, disabled=(cur == "running")):
            db.set_setting("pipeline_status", "running")
            st.rerun()
    with pc2:
        if st.button("⏸️ Pause", use_container_width=True, disabled=(cur == "paused")):
            db.set_setting("pipeline_status", "paused")
            st.rerun()
    
    st.markdown("---")
    st.subheader("⚙️ Behavior Settings")
    
    # Load current settings, fallback to defaults
    current_speed = int(db.get_setting("pipeline_speed") or 15)
    current_idle = int(db.get_setting("pipeline_idle") or 60)
    current_critic = db.get_setting("critic_model_override") or "Default (Round-Robin)"
    
    # List available model names from configs
    all_providers = db.get_all_providers()
    available_models = ["Default (Round-Robin)"] + [p["name"] for p in all_providers]
    # Ensure current is in list
    if current_critic not in available_models:
        available_models.append(current_critic)
        
    critic_idx = available_models.index(current_critic)
    
    with st.form("pipeline_control_form"):
        s1, s2, s3 = st.columns(3)
        new_speed = s1.slider("Cycle Delay (Seconds between generations)", min_value=30, max_value=120, value=max(30, current_speed))
        new_idle = s2.slider("Idle Wait (Seconds if no topics found)", min_value=10, max_value=300, value=current_idle, step=10)
        new_critic = s3.selectbox("Dedicated Critic Model", options=available_models, index=critic_idx, help="Forces the evaluation node to ONLY use this specific model, for consistency.")
        
        if st.form_submit_button("💾 Save Settings", use_container_width=True):
            db.set_setting("pipeline_speed", str(new_speed))
            db.set_setting("pipeline_idle", str(new_idle))
            db.set_setting("critic_model_override", str(new_critic))
            st.success("Pipeline settings saved!")

    st.markdown("---")
    st.subheader("📜 Pipeline Log (Last 50 Lines)")
    try:
        with open("pipeline.log", "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            st.code("".join(lines[-50:] if len(lines) > 50 else lines), language="log")
    except FileNotFoundError:
        st.info("No log file yet.")

# ═══════════════════════════════════════════════
# 🤖 MODELS & PRICES
# ═══════════════════════════════════════════════
elif page == "🤖 Models & Prices":
    st.title("🤖 Models & Prices")
    st.caption("Manage LLM endpoints, free tier delays, and input exact token prices for cost calculation.")
    
    providers = db.get_all_providers()
    df = pd.DataFrame(providers)
    
    if not df.empty:
        # Reorder columns for friendliness
        cols = ["id", "is_active", "name", "provider_base", "api_key_env_prefix", "base_url", "model_name", "role_tier", "cost_input_1m", "cost_output_1m"]
        df = df[cols]
        
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": None, # hide ID
                "is_active": st.column_config.CheckboxColumn("Active?", help="Enable or disable entire model"),
                "cost_input_1m": st.column_config.NumberColumn("In Cost / 1M ($)", format="$%.3f"),
                "cost_output_1m": st.column_config.NumberColumn("Out Cost / 1M ($)", format="$%.3f")
            }
        )
        
        if st.button("💾 Save Changes", type="primary"):
            st.toast("Saving model configurations...")
            all_ids = set([p["id"] for p in providers])
            current_ids = set()
            
            for _, row in edited_df.iterrows():
                row_dict = row.to_dict()
                r_id = row_dict.pop("id", None)
                
                # Fill NaNs
                for k, v in row_dict.items():
                    if pd.isna(v): row_dict[k] = None
                
                row_dict["is_active"] = 1 if row_dict.get("is_active") else 0
                
                if pd.isna(r_id):
                    # INSERT
                    db.insert_provider(row_dict)
                else:
                    r_id = int(r_id)
                    current_ids.add(r_id)
                    # UPDATE
                    db.update_provider(r_id, row_dict)
                    
            # DELETE any missing
            for d in (all_ids - current_ids):
                db.delete_provider(d)
                
            st.success("✅ Models & Prices successfully saved! Pipeline will load dynamically.")
            st.rerun()

# ═══════════════════════════════════════════════
# 🔑 API KEYS
# ═══════════════════════════════════════════════
elif page == "🔑 API Keys":
    st.title("🔑 API Provider Dashboard")
    st.caption("Manage granular API keys, free tier behaviors, and daily token limits.")
    
    # Display API Health Statuses
    health_data = db.get_api_health()
    if health_data:
        st.markdown("### 📡 Current API Health Monitor")
        for k, info in health_data.items():
            status = info.get("status", "ACTIVE")
            provider = info.get("provider", "Unknown")
            err = info.get("last_error", "")
            masked_key = f"{k[:8]}***{k[-4:]}" if len(k) > 12 else "****"
            
            if status == "ACTIVE":
                st.success(f"**{provider}** ({masked_key}): 🟢 OK - {err or 'No errors'}")
            elif status == "COOLDOWN":
                st.warning(f"**{provider}** ({masked_key}): ⏳ COOLDOWN (Rate Limit) - {err} | Until: {info.get('cooldown_until', 'Unknown')}")
            elif status == "ERROR":
                st.error(f"**{provider}** ({masked_key}): ❌ ERROR - {err}")
        st.markdown("---")

    # Provider-level global settings (Toggles & Daily Limits)
    st.markdown("### 🌐 Global Provider Limits")
    all_p = db.get_all_providers()
    unique_b = list(set([p["provider_base"] for p in all_p]))
    
    if unique_b:
        with st.form("provider_limits_form"):
            db_updates = {}
            for b in unique_b:
                col1, col2 = st.columns(2)
                curr_active = db.get_setting(f"provider_{b}") != "false"
                is_active = col1.checkbox(f"Enable {b.capitalize()}", value=curr_active, key=f"tgl_{b}")
                curr_limit = db.get_setting(f"limit_{b}") or "2000000"
                limit = col2.number_input(f"Daily Token Limit for {b.capitalize()}", min_value=1000, max_value=100000000, value=int(curr_limit), step=10000, key=f"lim_{b}")
                
                db_updates[f"provider_{b}"] = "true" if is_active else "false"
                db_updates[f"limit_{b}"] = str(limit)
                
            if st.form_submit_button("💾 Save Global Limits"):
                for k, v in db_updates.items():
                    db.set_setting(k, v)
                st.success("Global Limits saved!")
                st.rerun()
            
    st.markdown("---")
    
    # Granular Key Management
    st.markdown("### 🔑 Granular Key Management")
    st.info("💡 You can add multiple keys per provider. Explicitly mark specific keys as 'Free Tier' to give them dedicated safety delays.")
    
    keys_data = db.get_api_keys()
    df_keys = pd.DataFrame(keys_data)
    
    if df_keys.empty:
        df_keys = pd.DataFrame(columns=["id", "is_active", "provider_base", "api_key", "is_free_tier", "free_tier_delay"])
    else:
        cols = ["id", "is_active", "provider_base", "api_key", "is_free_tier", "free_tier_delay"]
        df_keys = df_keys[[c for c in cols if c in df_keys.columns]]
        
    edited_keys = st.data_editor(
        df_keys,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": None,
            "is_active": st.column_config.CheckboxColumn("Active?", default=True),
            "provider_base": st.column_config.SelectboxColumn("Provider", options=unique_b, required=True),
            "api_key": st.column_config.TextColumn("Secret API Key", required=True),
            "is_free_tier": st.column_config.CheckboxColumn("Free Tier?", help="Applies delay when this specific key is used.", default=False),
            "free_tier_delay": st.column_config.NumberColumn("Delay (s)", min_value=0, default=0)
        }
    )
    
    if st.button("💾 Save API Keys", type="primary"):
        st.toast("Saving API keys...")
        all_ids = set([k["id"] for k in keys_data])
        current_ids = set()
        
        has_error = False
        for _, row in edited_keys.iterrows():
            row_dict = row.to_dict()
            r_id = row_dict.pop("id", None)
            
            for k, v in row_dict.items():
                if pd.isna(v): row_dict[k] = None
                
            if not row_dict.get("api_key") or not row_dict.get("provider_base"):
                continue # Skip invalid rows
                
            row_dict["is_free_tier"] = 1 if row_dict.get("is_free_tier") else 0
            row_dict["is_active"] = 1 if row_dict.get("is_active") else 0
            
            if pd.isna(r_id):
                # INSERT
                try:
                    db.insert_api_key(row_dict)
                except Exception as e:
                    has_error = True
                    st.error(f"Error inserting key {row_dict['api_key'][:5]}: {e}")
            else:
                r_id = int(r_id)
                current_ids.add(r_id)
                db.update_api_key(r_id, row_dict)
                
        # DELETE any missing
        for d in (all_ids - current_ids):
            db.delete_api_key(d)
                
        if not has_error:
            st.success("✅ API Keys successfully saved! (Changes take effect on next generation)")
            st.rerun()

# ═══════════════════════════════════════════════
# 📚 KNOWLEDGE SOURCES
# ═══════════════════════════════════════════════
elif page == "📚 Knowledge Sources":
    st.title("📚 Knowledge Sources")
    st.caption("Manage where the Research Agent pulls its seed data from.")
    
    st.info("💡 **Tip:** Use `wikipedia_random` for chaotic variety, `rss` for news, `reddit` for trends, and `hackernews` for debates. Set `config` as valid JSON.")

    sources = db.get_knowledge_sources()
    df = pd.DataFrame(sources)
    
    if not df.empty:
        # Reorder for clarity
        df = df[["id", "is_active", "name", "source_type", "cooldown_minutes", "config", "last_fetched_at"]]
        
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn(disabled=True),
            "is_active": st.column_config.CheckboxColumn("Active?"),
            "name": st.column_config.TextColumn("Source Name", required=True),
            "source_type": st.column_config.SelectboxColumn("Type", options=["wikipedia_search", "wikipedia_random", "rss", "reddit", "arxiv", "hackernews"], required=True),
            "cooldown_minutes": st.column_config.NumberColumn("Cooldown (Min)", min_value=0),
            "config": st.column_config.TextColumn("Config (JSON)"),
            "last_fetched_at": st.column_config.DatetimeColumn("Last Fetched", disabled=True),
        },
        hide_index=True
    )

    if st.button("💾 Save Knowledge Sources", type="primary"):
        all_ids = set(df["id"].dropna()) if not df.empty else set()
        current_ids = set()
        
        has_error = False
        
        for index, row in edited_df.iterrows():
            row_dict = row.to_dict()
            r_id = row_dict.pop("id", None)
            
            if not row_dict.get("name") or not row_dict.get("source_type"):
                continue
                
            row_dict["is_active"] = 1 if row_dict.get("is_active") else 0
            
            # Simple JSON validation
            import json
            try:
                if row_dict.get("config"):
                    json.loads(row_dict["config"])
                else:
                    row_dict["config"] = "{}"
            except Exception:
                st.error(f"Invalid JSON in Config for '{row_dict['name']}'")
                has_error = True
                continue
            
            # last_fetched_at cannot be updated by user directly reliably via UI edit, ignore it for update
            row_dict.pop("last_fetched_at", None)

            if pd.isna(r_id):
                # INSERT
                try:
                    with db.get_connection() as conn:
                        fields = ", ".join(row_dict.keys())
                        placeholders = ", ".join(["?" for _ in row_dict])
                        conn.execute(f"INSERT INTO knowledge_sources ({fields}) VALUES ({placeholders})", list(row_dict.values()))
                        conn.commit()
                except Exception as e:
                    has_error = True
                    st.error(f"Error inserting source '{row_dict['name']}': {e}")
            else:
                r_id = int(r_id)
                current_ids.add(r_id)
                db.update_knowledge_source(r_id, row_dict)
                
        # DELETE missing
        for d in (all_ids - current_ids):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM knowledge_sources WHERE id = ?", (d,))
                conn.commit()
                
        if not has_error:
            st.success("✅ Knowledge Sources saved successfully!")
            st.rerun()

# ═══════════════════════════════════════════════
# 📥 EXPORT DATASET
# ═══════════════════════════════════════════════
elif page == "📥 Export Dataset":
    st.title("📥 Export Dataset")
    st.caption("Download conversations as a JSONL file for SFT fine-tuning")
    
    stats = db.get_dashboard_stats()
    st.metric("Total Exportable (PASS)", stats["passed"])
    
    st.markdown("---")
    st.subheader("🔧 Export Filters")
    ef1, ef2, ef3 = st.columns(3)
    tier_sel = ef1.selectbox("Tier Filter", ["All", "1 (Gold)", "2 (Silver)", "3 (Bronze)"])
    domain_sel = ef2.text_input("Domain Filter", placeholder="Leave empty for all")
    diff_sel = ef3.selectbox("Difficulty Filter", ["All", "Beginner", "Intermediate", "Advanced"])
    
    tier_val = None
    if tier_sel != "All":
        tier_val = int(tier_sel[0])
    domain_val = domain_sel.strip() if domain_sel.strip() else None
    diff_val = diff_sel if diff_sel != "All" else None
    
    st.markdown("---")
    st.subheader("🚀 Push to Hugging Face")
    st.info("Directly upload the filtered dataset to your Hugging Face account. The system will create the dataset if it doesn't exist.")
    
    hf_token = st.text_input("HF Write Token (hf_...)", type="password")
    hf_repo = st.text_input("Dataset Repo Name (e.g. username/whusdata-sft)", placeholder="username/dataset_name")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📦 Generate JSONL Download", use_container_width=True):
            with st.spinner("Exporting..."):
                data = db.export_jsonl(tier_filter=tier_val, domain_filter=domain_val, difficulty_filter=diff_val)
                if data:
                    jsonl = "\n".join([json.dumps(d, ensure_ascii=False) for d in data])
                    st.download_button(
                        label=f"⬇️ Download {len(data)} conversations (.jsonl)",
                        data=jsonl,
                        file_name=f"whusdata_sft_{datetime.now().strftime('%Y%m%d')}.jsonl",
                        mime="application/jsonl",
                        use_container_width=True
                    )
                    st.success(f"{len(data)} conversations ready!")
                else:
                    st.warning("No matching conversations found.")
                    
    with c2:
        if st.button("☁️ Push to HF Hub", use_container_width=True):
            if not hf_token or not hf_repo:
                st.error("Please provide both HF Token and Repo Name.")
            else:
                with st.spinner("Exporting & Pushing to Hugging Face..."):
                    data = db.export_jsonl(tier_filter=tier_val, domain_filter=domain_val, difficulty_filter=diff_val)
                    if not data:
                        st.warning("No matching conversations found.")
                    else:
                        try:
                            from huggingface_hub import HfApi
                            import tempfile
                            import os
                            
                            api = HfApi(token=hf_token.strip())
                            
                            # Ensure repo exists
                            api.create_repo(repo_id=hf_repo.strip(), repo_type="dataset", exist_ok=True)
                            
                            jsonl_data = "\n".join([json.dumps(d, ensure_ascii=False) for d in data])
                            
                            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8") as f:
                                f.write(jsonl_data)
                                temp_path = f.name
                                
                            api.upload_file(
                                path_or_fileobj=temp_path,
                                path_in_repo=f"train_{datetime.now().strftime('%Y%m%d')}.jsonl",
                                repo_id=hf_repo.strip(),
                                repo_type="dataset"
                            )
                            os.unlink(temp_path)
                            
                            st.success(f"✅ Successfully pushed {len(data)} rows to {hf_repo}!")
                            st.balloons()
                        except ImportError:
                            st.error("huggingface_hub is not installed! Run `pip install huggingface_hub`")
                        except Exception as e:
                            st.error(f"Push failed: {e}")
