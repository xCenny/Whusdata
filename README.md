# WHUSDATA — Autonomous Synthetic Data Pipeline ⚡

Açık kaynaklı dil modellerini eğitmek (SFT / RLHF fine-tuning) amacıyla **7/24 otonom** çalışan, **çatışma mühendisliği (Conflict-Engineered)** ile yüksek kaliteli çoklu-turlu (multi-turn) sentetik sohbet verisi üreten bir **Çoklu-Ajan** sistemidir.

> **Hedef Model Profili:** Empatik + Reasoning yapabilen + Çatışma çözebilen + Tutarlı (Memory Consistent) → Genel Amaçlı Asistan

---

## 🏗 Sistem Mimarisi

```
┌──────────────────────────────────────────────────────────┐
│             Streamlit Admin UI (:8501)                    │
│  📊 Dashboard │ 💬 Browser │ 🧬 Augmentation │ 📥 Export  │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│               Orchestrator (main.py)                      │
│   Budget Guard ($10/day) │ Workspace Router │ State FSM   │
├──────────────────────┬───────────────────────────────────┤
│  Research Agent      │     Granular Teacher (6-Step)      │
│  ─────────────       │     ──────────────────────────     │
│  + Wikipedia Scraper │  ┌─ User Turn (Adversarial)        │
│  + RSS / Reddit / HN │  │  Assistant Turn (Reasoning)     │
│  + ChromaDB Dedup    │  │  Metadata Extraction            │
│                      │  │  Critic (Cross-Model Judging)   │
│                      │  └─ Reflect (Targeted Repair)      │
├──────────────────────┴───────────────────────────────────┤
│  Data Augmenter      │  Multi-Repo HF Exporter            │
│  + Paraphrasing      │  + Per-Target Filters              │
│  + 3 Variation Styles│  + Dataset Workspace Routing       │
├──────────────────────────────────────────────────────────┤
│         SQLite (WAL) + ChromaDB + Cost Log                │
│  generations │ hf_export_targets │ failure_log │ topics   │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Temel Özellikler

### Veri Üretimi
- **8-12 Mesajlık Adversarial Diyalog**: Sıralı ajan çağrıları ile gerçek tartışma dinamiği
- **Mantık Saldırıları**: Strawman, Ad Hominem, False Dilemma, Contradiction Trap
- **Halüsinasyon Koruması**: Citation Hedging — sahte kaynak uydurmak yerine belirsizlik itirafı
- **Contextual Memory Recall**: Asistan önceki turlardaki spesifik hatalara geri referans verir
- **Dinamik Bitiş**: LLM-Driven Termination — tekrar tespitinde otonom kapanış

### Kalite Kontrol (Elite Tier Capping)
- **Acımasız Hakem (Brutal Critic)**: 5 vektörde puanlama + Chain-of-Thought analitik
- **Strict Tier Sınırı**: `final_tier = max(confidence_tier, critic_quality_tier)` — Critic'in kalite kararı confidence'ı geçersiz kılabilir
- **Dengeli Tier 1 Kriterleri**: Spesifik kavramlar (packing/cracking), mekanizmalar (vote dilution), tarihsel referanslar (2017 redistribution) veya kurumsal atıflar (Brazilian Gymnastics Confederation) kullanan diyaloglar Tier 1 alabilir. Anti-hallucination kuralı gereği uydurma isim/rakam vermemek **doğru davranıştır** ve Tier 1'i engellemez.
- **Çan Eğrisi**: 🥇 Gold (Tier 1: Spesifik kavram + 0 tekrar) · 🥈 Silver (Tier 2: İyi tartışma ama generic) · 🥉 Bronze

### 🧬 Diyalog Dinamikleri & Uzunluk (Length Sequencing)
- **Dinamik Ritim**: Asistan cevapları `Medium -> Short -> Long` döngüsü izler. Monotonluğu önlemek için farklı turlarda farklı uzunluklarda (kısa vurucu cevaplar ve uzun çürütmeler) yanıt verir.
- **Sıfır Yumuşatma (No Soft Validation)**: Asistan "haklısın", "anlayabiliyorum" gibi onay kalıplarını asla kullanmaz.
- **Ad Hominem Defense**: Kişisel saldırılara boyun eğmeyip saldırıyı 1 cümleyle gösterip tekrar argümana döner.

### 📂 Dataset Workspaces (Koleksiyonlar)
- **Çoklu veri seti**: Pipeline Control'den aktif workspace seçimi (örn: `climate-science`, `math-reasoning`)
- **Routing Toggle**: İstenirse UI'dan "Dataset Routing" kapatılarak tüm verilerin `default` ismine yazılması sağlanabilir.
- **Seçici yönetim**: İstenen workspace'i tamamen veya sadece augmented kısmıyla silme yeteneği.

### 📥 Multi-Repo HF Export
- **Çoklu hedef**: Birden fazla HuggingFace reposu tanımlama (filtre kurallarıyla)
- **Tek tuş**: "Push to ALL Active Targets" ile eşzamanlı upload
- **JSONL indirme**: Anlık filtrelenmiş veri seti dosyası

### Maliyet Yönetimi
- **BudgetGuardian**: $10/gün otomatik durdurma
- **Token Caching**: Prefix-match prompt tasarımı ile %90 maliyet düşüşü
- **Critic Token Optimizasyonu**: Metadata verify kaldırıldı
- **Free Tier Delay**: Ücretsiz API'ler için yapay bekleme

---

## 📤 Örnek JSONL Çıktısı

```json
{
  "topic": "The Fermi Paradox and the Simulation Hypothesis",
  "domain": "Astrophysics",
  "difficulty": "Advanced",
  "persona": "Adversarial Skeptic",
  "scenario_conflict": "Logical Fallacy Trap",
  "winner": "Assistant",
  "logic_score": 0.95,
  "factual_score": 1.0,
  "critic_confidence": 0.887,
  "memory_score": 0.91,
  "model_used": "gemini-2.5-pro",
  "messages": [
    {"role": "user", "content": "If we were in a simulation, the processing power required..."},
    {"role": "assistant", "content": "Assuming a simulation requires rendering the entire universe simultaneously is an argument from incredulity..."}
  ],
  "critic_analytics": {
    "reasoning": "The assistant perfectly detected the user's Argument from Incredulity...",
    "detected_fallacies": ["Argument from Incredulity", "False Dilemma"],
    "assistant_counters": ["Logical Breakdown", "Citation Hedging"]
  }
}
```

---

## 📊 Streamlit Admin Dashboard

Tarayıcıdan `http://SUNUCU_IP:8501` adresine girerek erişilir.

| Sayfa | Özellik |
|---|---|
| 📊 Dashboard | AI Öngörüler, Maliyet, Tier kartları, 14 günlük token grafiği |
| 📈 Drift Monitor | 7 günlük rolling PASS/güven trendi |
| 💬 Conversations | PASS/FAIL sohbet tarayıcı + Critic kararları |
| 🎯 Weekly Planner | Hedef anahtar kelimeler ve öncelikler |
| ⚙️ Pipeline Control | Başlat/Durdur, **📂 Workspace seçimi**, Hız ayarları, Log |
| 🤖 Models & Prices | Dinamik LLM sağlayıcı ve fiyatlandırma |
| 🔑 API Keys | Çoklu anahtar, Cooldown, sağlık durumu |
| 📚 Knowledge Sources | Wikipedia, RSS, Reddit, HN, ArXiv kaynakları |
| 🧬 Data Augmentation | Tier/Workspace bazlı çoğaltma, Multiplier, Model seçimi |
| 📥 Export Dataset | 4 boyutlu filtre, **🗑️ Silme**, Multi-Repo HF Push |

---

## 🛠 Kurulum

### Gereksinimler
- Python 3.10+
- API anahtarı: Gemini, Groq veya DeepSeek (en az bir tanesi)

### Lokal (Windows / Linux)
```bash
git clone https://github.com/xCenny/Whusdata.git
cd Whusdata
cp .env.example .env   # .env dosyasını düzenle
pip install -r requirements.txt
python main.py         # Terminal 1
streamlit run src/ui.py  # Terminal 2
```

### CasaOS / Docker
```bash
docker-compose up -d --build
# Dashboard: http://SUNUCU_IP:8501
```

> **CasaOS Custom Install:** App Store → Custom Install → Import → `docker-compose.yml` içeriğini yapıştır.

---

## 📁 Proje Yapısı

```
whusdata/
├── main.py                 # Orchestrator — ana döngü + Workspace Router
├── src/
│   ├── db.py               # SQLite + ChromaDB (WAL mode) + HF Targets
│   ├── llm_client.py       # Multi-provider LLM client & JSON Extractor
│   ├── graph.py            # LangGraph state machine (Adversarial Loops)
│   ├── researcher.py       # Multi-Source Fetcher (Wikipedia, ArXiv, RSS)
│   ├── prompts.py          # Teacher, Critic ve Reflection prompt'ları
│   ├── augmenter.py        # Data Augmentation (Paraphrasing Engine)
│   └── ui.py               # Streamlit Admin Dashboard (Minimalist)
├── start.sh                # Docker entrypoint
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📝 Veritabanı Şeması

### `generations`
| Sütun | Tip | Açıklama |
|---|---|---|
| `conversation_history` | TEXT (JSON) | Multi-turn diyalog |
| `persona_type` | TEXT | Kullanıcı profili |
| `conflict_type` | TEXT | Çatışma türü |
| `domain` | TEXT | Konu alanı |
| `critic_status` | TEXT | PASS / FAIL |
| `critic_confidence` | REAL | 0.0 – 1.0 |
| `factual_score` | REAL | Bilgi doğruluğu |
| `critic_analytics` | TEXT (JSON) | Reasoning, fallacies, counters |
| `is_augmented` | BOOLEAN | Çoğaltılmış veri mi? |
| `original_id` | INTEGER | Orijinal verinin ID'si |
| `dataset_name` | TEXT | Hangi workspace'e ait |

### `hf_export_targets`
| Sütun | Açıklama |
|---|---|
| `name` | Hedef etiketi |
| `repo_id` | HuggingFace repo (user/dataset) |
| `tier_filter` / `domain_filter` / `dataset_filter` | Filtreler |

---

## 🗺 Yol Haritası (Roadmap)

- [x] Streamlit Admin Dashboard (Phase 3)
- [x] Turn-by-Turn Adversarial Machine (Phase 4)
- [x] Server-side Weighted Critic Scoring (Phase 1 Hardened)
- [x] Budget & Token Cost Monitoring (BudgetGuardian)
- [x] Persona & Conflict Distribution Control (Phase 9)
- [x] Anti-Hallucination & Extended Reasoning (Phase 10)
- [x] Organik Hafıza / Contextual Memory Recall (Phase 11)
- [x] Deep RLHF Analytics & Chain of Thought (Phase 12)
- [x] Critic Token Cost Optimization
- [x] HuggingFace Datasets oto-push
- [x] Synthetic Data Augmentation & Multi-Repo Export
- [x] Dataset Workspaces (Koleksiyonlar)
- [x] Minimalist UI Redesign
- [ ] Light Verifier (Augmented Data Quality Check)
- [ ] Counterfactual Augmentation & Deduplication
- [ ] Prompt A/B testing framework

---

## 📄 Lisans

MIT License
