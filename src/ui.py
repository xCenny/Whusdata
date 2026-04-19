import streamlit as st
import json
import pandas as pd
from datetime import datetime
from src.db import DatabaseManager

# ── Page Config ──
st.set_page_config(
    page_title="Whusdata — Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimalist CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Metric Cards — flat, clean, no gradients */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover { border-color: rgba(255,255,255,0.18); }
    .metric-card h2 {
        font-size: 1.8rem; font-weight: 700; margin: 0;
        color: #e8eaed; letter-spacing: -0.02em;
    }
    .metric-card p { font-size: 0.75rem; color: #9aa0a6; margin: 6px 0 0 0; font-weight: 400; letter-spacing: 0.03em; text-transform: uppercase; }
    
    /* Tier accents — subtle left border */
    .tier-gold { border-left: 3px solid #f5c542; }
    .tier-silver { border-left: 3px solid #9e9e9e; }
    .tier-bronze { border-left: 3px solid #bf8040; }
    
    /* Badges — minimal pill style */
    .pass-badge { background: rgba(52,168,83,0.15); color: #81c995; padding: 2px 10px; border-radius: 100px; font-weight: 500; font-size: 0.75rem; }
    .fail-badge { background: rgba(234,67,53,0.15); color: #f28b82; padding: 2px 10px; border-radius: 100px; font-weight: 500; font-size: 0.75rem; }
    
    /* Sidebar — clean dark */
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(255,255,255,0.06); }
    div[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem !important; letter-spacing: 0.01em; }
    
    /* Conversation bubbles — minimal */
    .convo-bubble-user {
        background: rgba(66,133,244,0.08);
        border-radius: 10px; padding: 12px 16px; margin: 4px 0;
        border-left: 2px solid rgba(66,133,244,0.4);
        font-size: 0.9rem; line-height: 1.6;
    }
    .convo-bubble-assistant {
        background: rgba(52,168,83,0.08);
        border-radius: 10px; padding: 12px 16px; margin: 4px 0;
        border-left: 2px solid rgba(52,168,83,0.4);
        font-size: 0.9rem; line-height: 1.6;
    }
    
    /* Reduce visual noise */
    .stMarkdown hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
    h1 { font-weight: 600 !important; letter-spacing: -0.03em !important; font-size: 1.6rem !important; }
    h2, .stSubheader { font-weight: 500 !important; font-size: 1.1rem !important; color: #bdc1c6 !important; }
    h3 { font-weight: 500 !important; font-size: 1rem !important; color: #9aa0a6 !important; }
    
    /* Buttons — clean outline style */
    .stButton > button { border-radius: 8px !important; font-weight: 500 !important; font-size: 0.85rem !important; }
    
    /* Data editor — tighter */
    .stDataFrame { font-size: 0.8rem !important; }
    
    /* Remove extra padding */
    .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# ── Sidebar ──
st.sidebar.markdown("#### ⚡ Whusdata")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "📈 Drift Monitor", "💬 Conversations", "🎯 Weekly Planner", "⚙️ Pipeline Control", "🤖 Models & Prices", "🔑 API Keys", "📚 Knowledge Sources", "🧬 Data Augmentation", "🏷️ AI Re-Tagger", "📥 Export Dataset"],
    label_visibility="collapsed"
)
pipeline_status = db.get_setting("pipeline_status") or "running"
cal_mode = db.get_setting("calibration_mode") or "false"
ws_name = db.get_setting("current_dataset_name") or "default"
st.sidebar.caption(f"{'🟢' if pipeline_status == 'running' else '🔴'} {pipeline_status.upper()} · 📂 `{ws_name}`")
if cal_mode == "true":
    st.sidebar.caption("🧪 CALIBRATION MODE")
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
        st.bar_chart(df_tokens["total_tokens"], color="#5f9ea0")
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
            # Show original persona and detailed persona if exists
            det_p = convo.get("detailed_persona", "")
            p_text = f"**Persona:** `{convo.get('persona_type', '-')}`"
            if det_p and det_p != "Unknown":
                p_text += f"\n\n<small><i>{det_p}</i></small>"
            tc1.markdown(p_text, unsafe_allow_html=True)
            
            tc2.markdown(f"**Conflict:** `{convo.get('conflict_type', '-')}`")
            
            bc = convo.get("broad_category", "")
            d_text = f"**Domain:** `{convo.get('domain', '-')}`"
            if bc and bc != "Unknown":
                d_text = f"**Tag:** `{bc}`\n\n" + d_text
            tc3.markdown(d_text)
            
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
    routing_enabled = db.get_setting("enable_dataset_routing") != "false"
    
    tgt_col1, tgt_col2 = st.columns([4, 1])
    tgt_col1.subheader("📂 Dataset Workspace Routing")
    
    if tgt_col2.toggle("Enable Routing", value=routing_enabled):
        if not routing_enabled:
            db.set_setting("enable_dataset_routing", "true")
            st.rerun()
    else:
        if routing_enabled:
            db.set_setting("enable_dataset_routing", "false")
            st.rerun()

    if routing_enabled:
        st.caption("All new generations will be tagged with this dataset name. Change it to start filling a different dataset.")
        
        existing_datasets = list(db.get_unique_datasets())
        if "default" not in existing_datasets:
            existing_datasets.insert(0, "default")
            
        current_ws = db.get_setting("current_dataset_name") or "default"
        
        ws_c1, ws_c2 = st.columns([2, 1])
        with ws_c1:
            ws_choice = st.selectbox("Select existing workspace", options=existing_datasets, index=existing_datasets.index(current_ws) if current_ws in existing_datasets else 0)
        with ws_c2:
            new_ws = st.text_input("Or create new workspace", placeholder="e.g. science-adv")
        
        final_ws = new_ws.strip() if new_ws.strip() else ws_choice
        if st.button("📂 Set Active Workspace", use_container_width=True):
            db.set_setting("current_dataset_name", final_ws)
            st.success(f"Active workspace set to: **{final_ws}**")
            st.rerun()
        
        st.info(f"🔵 Currently generating into: **`{current_ws}`**")
    else:
        st.info("🔵 Routing disabled. All new dataset generations will be tagged as **`default`**.")

    st.markdown("---")
    st.subheader("🎯 Topic Focus")
    st.caption("Set a domain focus to guide the Research Agent toward specific topics. Leave empty for random/diverse mode.")
    
    current_focus = db.get_setting("topic_focus") or ""
    current_focus_instructions = db.get_setting("topic_focus_instructions") or ""
    focus_enabled = bool(current_focus.strip())
    
    with st.form("topic_focus_form"):
        fc1, fc2 = st.columns([3, 1])
        new_focus = fc1.text_input(
            "Focus Domains (comma separated)",
            value=current_focus,
            placeholder="Fizik, Teknoloji, Biyoloji",
            help="Birden fazla alan virgülle ayrılabilir. Boş bırakılırsa rastgele çalışır."
        )
        fc2.markdown("")
        fc2.markdown("")
        if focus_enabled:
            fc2.success("🟢 Active")
        else:
            fc2.info("⚪ Off")
        
        new_instructions = st.text_area(
            "Additional Focus Instructions (Optional)",
            value=current_focus_instructions,
            placeholder="Kuantum mekaniği ve parçacık fiziği konularına odaklan...",
            height=80,
            help="Ek detaylı talimatlar. Research Agent bu yönergeleri dikkate alır."
        )
        
        available_domains = "Fizik/Physics, Teknoloji/Technology, Biyoloji/Biology, Felsefe/Philosophy, Tarih/History, Ekonomi/Economics, Matematik/Mathematics, Psikoloji/Psychology"
        st.caption(f"📝 Predefined domains: {available_domains}")
        st.caption("💡 Custom domains also work — the LLM will target any domain you specify.")
        
        fc_btn1, fc_btn2 = st.columns(2)
        with fc_btn1:
            if st.form_submit_button("💾 Save Focus", use_container_width=True):
                db.set_setting("topic_focus", new_focus.strip())
                db.set_setting("topic_focus_instructions", new_instructions.strip())
                if new_focus.strip():
                    st.success(f"🎯 Topic Focus set to: **{new_focus.strip()}**")
                else:
                    st.success("🔄 Topic Focus cleared. Research Agent will use random/diverse mode.")
                st.rerun()
        with fc_btn2:
            if st.form_submit_button("🗑️ Clear Focus", use_container_width=True):
                db.set_setting("topic_focus", "")
                db.set_setting("topic_focus_instructions", "")
                st.success("Focus cleared!")
                st.rerun()
    
    if focus_enabled:
        st.info(f"🎯 Currently focusing on: **`{current_focus}`**")

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
# 🧬 DATA AUGMENTATION
# ═══════════════════════════════════════════════
elif page == "🧬 Data Augmentation":
    st.title("🧬 Data Augmentation")
    st.caption("Multiply your verified Tier 1 data by paraphrasing dialogues in different styles.")
    
    st.info("💡 **How it works:** Taking a 100% verified Gold conversation and rewriting it in different styles (e.g., Slang, Academic) via a cheaper model is up to 10x cheaper than starting from scratch and passing it through the Strict Critic.")
    
    with st.form("aug_form"):
        sc1, sc2, sc3 = st.columns(3)
        tier_to_augment = sc1.selectbox("Target Tier source (e.g. Tier 1)", [1, 2, 3])
        num_seeds = sc2.number_input("How many conversations to parse?", min_value=1, max_value=100, value=10)
        multiplier = sc3.slider("Multiplier (Variations per conv)", min_value=1, max_value=5, value=3)
        
        aug_datasets = db.get_unique_datasets()
        aug_ds = st.selectbox("📂 Source Dataset Workspace", options=["All"] + aug_datasets)
        aug_ds_val = aug_ds if aug_ds != "All" else None
        
        all_providers = db.get_all_providers()
        available_models = [p["name"] for p in all_providers] if all_providers else ["gemini-flash"]
        model_choice = st.selectbox("Augmentation Model (Cheap/Fast recommended)", available_models)
        
        st.markdown("💡 *Job runs in background — you can navigate away from this page.*")
        
        if st.form_submit_button("🚀 Start Augmentation Process", use_container_width=True):
            targets = db.get_generations_for_augmentation(limit=int(num_seeds), tier=int(tier_to_augment), dataset_filter=aug_ds_val)
            
            if not targets:
                st.warning("No eligible un-augmented conversations found for that tier/dataset.")
            else:
                from src.background_worker import start_augment_job
                job_id = start_augment_job(targets, int(multiplier), model_choice)
                st.success(f"🚀 Background augmentation job #{job_id} started with {len(targets)} conversations!")
                st.rerun()
    
    # Show active & recent jobs
    st.markdown("---")
    st.subheader("📋 Job Status")
    
    active_jobs = db.get_active_background_jobs("augment")
    recent_jobs = db.get_recent_background_jobs("augment", limit=5)
    
    if active_jobs:
        for job in active_jobs:
            pct = (job["progress"] / job["total"] * 100) if job["total"] > 0 else 0
            st.progress(job["progress"] / max(job["total"], 1), text=f"Job #{job['id']} — {job['progress']}/{job['total']} ({pct:.0f}%) | ✅ {job['success_count']} | ❌ {job['error_count']}")
        
        if st.button("🔄 Refresh Status", key="refresh_aug"):
            st.rerun()
        st.info("⏳ Auto-refresh: click the button above or navigate away and come back.")
    
    if recent_jobs:
        with st.expander("📜 Recent Augmentation Jobs"):
            for job in recent_jobs:
                status_icon = "🟢" if job["status"] == "DONE" else "🔵" if job["status"] == "RUNNING" else "🔴"
                config = json.loads(job.get("config", "{}"))
                st.markdown(f"{status_icon} **Job #{job['id']}** — {job['status']} | Model: `{config.get('model', 'N/A')}` | ✅ {job['success_count']} created | {job.get('result_message', '')}")

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
    ef1, ef2, ef3, ef4 = st.columns(4)
    tier_sel = ef1.selectbox("Tier Filter", ["All", "1 (Gold)", "2 (Silver)", "3 (Bronze)"])
    domain_sel = ef2.text_input("Domain Filter", placeholder="Leave empty for all")
    diff_sel = ef3.selectbox("Difficulty Filter", ["All", "Beginner", "Intermediate", "Advanced"])
    
    export_datasets = db.get_unique_datasets()
    ds_sel = ef4.selectbox("📂 Dataset Workspace", ["All"] + export_datasets)
    
    tier_val = None
    if tier_sel != "All":
        tier_val = int(tier_sel[0])
    domain_val = domain_sel.strip() if domain_sel.strip() else None
    diff_val = diff_sel if diff_sel != "All" else None
    ds_val = ds_sel if ds_sel != "All" else None
    
    # ── Dataset Management (Delete) ──
    st.markdown("---")
    st.subheader("🗑️ Dataset Management")
    del_c1, del_c2 = st.columns(2)
    with del_c1:
        del_ds = st.selectbox("Select dataset to manage", export_datasets, key="del_ds")
    with del_c2:
        del_mode = st.radio("Delete mode", ["Only Augmented (Safe)", "Entire Dataset (⚠️ Destructive)"], horizontal=True)
    
    if st.button("🗑️ Delete Selected", type="secondary", use_container_width=True):
        aug_only = del_mode.startswith("Only")
        count = db.delete_generations(del_ds, augmented_only=aug_only)
        st.warning(f"Deleted {count} rows from '{del_ds}' ({'augmented only' if aug_only else 'ALL data'}).")
        st.rerun()
    
    st.markdown("---")
    st.subheader("🚀 Multi-Repo Hugging Face Push")
    st.info("Directly upload data to multiple Hugging Face datasets based on filters.")
    
    # HF Targets Manager
    st.markdown("#### ⚙️ Configure Targets")
    st.caption("Example — Target Label: `Main Dataset`, Repo: `xCenny/Whusdata-Main`")
    hf_targets = db.get_hf_targets()
    df_hf = pd.DataFrame(hf_targets)
    if not df_hf.empty:
        cols_to_show = ["id", "is_active", "name", "repo_id", "hf_token", "tier_filter", "domain_filter", "difficulty_filter", "dataset_filter"]
        for c in cols_to_show:
            if c not in df_hf.columns:
                df_hf[c] = None
        df_hf = df_hf[cols_to_show]
        
    edited_hf = st.data_editor(
        df_hf if not df_hf.empty else pd.DataFrame(columns=["id", "is_active", "name", "repo_id", "hf_token", "tier_filter", "domain_filter", "difficulty_filter", "dataset_filter"]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": None,
            "is_active": st.column_config.CheckboxColumn("Active?", default=True),
            "name": st.column_config.TextColumn("Target Label", required=True),
            "repo_id": st.column_config.TextColumn("Repo (user/dataset)", required=True),
            "hf_token": st.column_config.TextColumn("HF Token (hf_...)", required=True),
            "tier_filter": st.column_config.NumberColumn("Tier (1,2,3 or Empty)"),
            "domain_filter": st.column_config.TextColumn("Domain (or Empty)"),
            "difficulty_filter": st.column_config.SelectboxColumn("Difficulty", options=["Beginner", "Intermediate", "Advanced"]),
            "dataset_filter": st.column_config.TextColumn("Dataset (or Empty)")
        }
    )
    
    if st.button("💾 Save Export Targets", type="primary"):
        all_ids = set([t["id"] for t in hf_targets])
        current_ids = set()
        for _, row in edited_hf.iterrows():
            row_dict = row.to_dict()
            r_id = row_dict.pop("id", None)
            for k, v in row_dict.items():
                if pd.isna(v): row_dict[k] = None
            if not row_dict.get("name") or not row_dict.get("repo_id") or not row_dict.get("hf_token"):
                continue
            row_dict["is_active"] = 1 if row_dict.get("is_active") else 0
                
            if pd.isna(r_id):
                db.insert_hf_target(row_dict)
            else:
                r_id = int(r_id)
                current_ids.add(r_id)
                db.update_hf_target(r_id, row_dict)
                
        for d in (all_ids - current_ids):
            db.delete_hf_target(d)
        st.success("Target configurations saved!")
        st.rerun()

    st.markdown("#### 📤 Push Actions")
    c1, c2 = st.columns(2)
    with c1:
        st.info("⬇️ Instant JSONL Download")
        st.caption("Filters apply automatically to this file.")
        data = db.export_jsonl(tier_filter=tier_val, domain_filter=domain_val, difficulty_filter=diff_val, dataset_filter=ds_val)
        if data:
            jsonl = "\n".join([json.dumps(d, ensure_ascii=False) for d in data])
            st.download_button(
                label=f"⬇️ Download {len(data)} rows (.jsonl)",
                data=jsonl,
                file_name=f"whusdata_temp_{datetime.now().strftime('%Y%m%d')}.jsonl",
                mime="application/jsonl",
                use_container_width=True
            )
        else:
            st.button("❌ No Data Found", disabled=True, use_container_width=True)
                    
    with c2:
        if st.button("☁️ Push to ALL Active Targets", use_container_width=True):
            active_targets = [t for t in db.get_hf_targets() if t.get("is_active")]
            if not active_targets:
                st.warning("No active targets found.")
            else:
                try:
                    from huggingface_hub import HfApi
                    import tempfile
                    import os
                    
                    for t in active_targets:
                        with st.spinner(f"Pushing to {t['repo_id']}..."):
                            api = HfApi(token=t['hf_token'].strip())
                            api.create_repo(repo_id=t['repo_id'].strip(), repo_type="dataset", exist_ok=True)
                            
                            push_tier = int(t['tier_filter']) if t.get('tier_filter') else None
                            push_domain = str(t['domain_filter']).strip() if t.get('domain_filter') else None
                            push_diff = str(t['difficulty_filter']).strip() if t.get('difficulty_filter') else None
                            push_ds = str(t['dataset_filter']).strip() if t.get('dataset_filter') else None
                            
                            t_data = db.export_jsonl(tier_filter=push_tier, domain_filter=push_domain, difficulty_filter=push_diff, dataset_filter=push_ds)
                            
                            if not t_data:
                                st.warning(f"No data matched for {t['name']}. Skipping.")
                                continue
                                
                            jsonl_data = "\n".join([json.dumps(d, ensure_ascii=False) for d in t_data])
                            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8") as f:
                                f.write(jsonl_data)
                                temp_path = f.name
                                
                            api.upload_file(
                                path_or_fileobj=temp_path,
                                path_in_repo=f"train_{datetime.now().strftime('%Y%m%d')}.jsonl",
                                repo_id=t['repo_id'].strip(),
                                repo_type="dataset"
                            )
                            os.unlink(temp_path)
                            st.success(f"✅ {len(t_data)} rows pushed to {t['repo_id']}")
                    st.balloons()
                except Exception as e:
                    st.error(f"Push failed: {e}")

# ═══════════════════════════════════════════════
# 🏷️ AI RE-TAGGER
# ═══════════════════════════════════════════════
elif page == "🏷️ AI Re-Tagger":
    st.title("🏷️ AI Re-Tagger")
    st.caption("Retroactively classify/tag existing data using an AI model. Fix Unknown or missing domain, persona, difficulty etc.")
    
    retag_stats = db.get_retag_stats()
    
    # Stats Row
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(f'<div class="metric-card"><h2>{retag_stats["any_unknown"]}</h2><p>Need Re-Tag</p></div>', unsafe_allow_html=True)
    with rc2:
        st.markdown(f'<div class="metric-card"><h2>{retag_stats["unknown_domain"]}</h2><p>Unknown Domain</p></div>', unsafe_allow_html=True)
    with rc3:
        st.markdown(f'<div class="metric-card"><h2>{retag_stats["unknown_difficulty"]}</h2><p>Unknown Difficulty</p></div>', unsafe_allow_html=True)
    with rc4:
        st.markdown(f'<div class="metric-card"><h2>{retag_stats["total"]}</h2><p>Total Records</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detail breakdown
    with st.expander("📊 Detailed Unknown Breakdown"):
        det1, det2, det3 = st.columns(3)
        det1.metric("Unknown Persona", retag_stats["unknown_persona"])
        det2.metric("Unknown Conflict", retag_stats["unknown_conflict"])
        det3.metric("Unknown Resolution", retag_stats["unknown_resolution"])
    
    st.markdown("---")
    st.subheader("🚀 Batch Re-Tag")
    st.info("💡 **How it works:** The selected AI model reads each conversation and re-classifies domain, persona_type, conflict_type, resolution_style, and difficulty_level using the same prompt as the generation pipeline.")
    
    with st.form("retag_form"):
        rt1, rt2, rt3 = st.columns(3)
        
        all_providers = db.get_all_providers()
        available_models = [p["name"] for p in all_providers] if all_providers else ["gemini-flash"]
        retag_model = rt1.selectbox("AI Model for Re-Tagging", available_models, help="Choose a fast/cheap model for bulk re-tagging")
        
        batch_size = rt2.slider("Batch Size", min_value=1, max_value=200, value=25, step=5)
        
        retag_datasets = db.get_unique_datasets()
        retag_ds = rt3.selectbox("📂 Dataset Filter", options=["All"] + retag_datasets)
        retag_ds_val = retag_ds if retag_ds != "All" else None
        
        retag_mode = st.radio(
            "Re-Tag Mode",
            ["Only Unknown/Empty fields", "Re-tag ALL records (overwrite existing)"],
            horizontal=True,
            help="'Only Unknown' is safe — it only fills empty fields. 'ALL' overwrites even existing tags."
        )
        only_unknown = retag_mode.startswith("Only")
        
        st.markdown("💡 *Job runs in background — you can navigate away from this page.*")
        
        if st.form_submit_button("🚀 Start Re-Tagging", use_container_width=True):
            targets = db.get_generations_for_retagging(
                limit=int(batch_size),
                dataset_filter=retag_ds_val,
                only_unknown=only_unknown
            )
            
            if not targets:
                st.warning("No eligible records found for re-tagging with the selected filters.")
            else:
                from src.background_worker import start_retag_job
                job_id = start_retag_job(targets, retag_model, only_unknown)
                st.success(f"🚀 Background re-tag job #{job_id} started with {len(targets)} conversations!")
                st.rerun()

    # Show active & recent jobs
    st.markdown("---")
    st.subheader("📋 Job Status")
    
    active_jobs = db.get_active_background_jobs("retag")
    recent_jobs = db.get_recent_background_jobs("retag", limit=5)
    
    if active_jobs:
        for job in active_jobs:
            pct = (job["progress"] / job["total"] * 100) if job["total"] > 0 else 0
            st.progress(job["progress"] / max(job["total"], 1), text=f"Job #{job['id']} — {job['progress']}/{job['total']} ({pct:.0f}%) | ✅ {job['success_count']} | ❌ {job['error_count']}")
        
        if st.button("🔄 Refresh Status", key="refresh_retag"):
            st.rerun()
        st.info("⏳ Auto-refresh: click the button above or navigate away and come back.")
    
    if recent_jobs:
        with st.expander("📜 Recent Re-Tag Jobs"):
            for job in recent_jobs:
                status_icon = "🟢" if job["status"] == "DONE" else "🔵" if job["status"] == "RUNNING" else "🔴"
                config = json.loads(job.get("config", "{}"))
                st.markdown(f"{status_icon} **Job #{job['id']}** — {job['status']} | Model: `{config.get('model', 'N/A')}` | ✅ {job['success_count']} | {job.get('result_message', '')}")

