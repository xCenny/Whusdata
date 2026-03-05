# WHUSDATA — Autonomous Synthetic Data Pipeline 🧠⚡

Açık kaynaklı dil modellerini eğitmek (SFT / RLHF fine-tuning) amacıyla **7/24 otonom** çalışan, **çatışma mühendisliği (Conflict-Engineered)** ile yüksek kaliteli çoklu-turlu (multi-turn) sentetik sohbet verisi üreten bir **Çoklu-Ajan** sistemidir.

> **Hedef Model Profili:** Empatik + Reasoning yapabilen + Çatışma çözebilen + Tutarlı (Memory Consistent) → Genel Amaçlı Asistan

---

## 🏗 Sistem Mimarisi

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit Admin UI (:8501)              │
│  📊 Dashboard │ 💬 Browser │ 📈 Drift │ 🎯 Planner        │
└──────────────┬───────────────────────────────────────────┘
               │ 💰 Cost Tracking (BudgetGuardian)
┌──────────────▼───────────────────────────────────────────┐
│                    Orchestrator (main.py)                 │
│        Budget Guard ($10/day) │ State machine             │
├──────────────────────┬───────────────────────────────────┤
│  Research Agent      │        Granular Teacher (6-Step)  │
│  ─────────────       │        ─────────────────────────  │
│  + Wikipedia Scraper │   ┌─ User Turn (Adversarial)      │
│  + UI Keywords       │   │  Assistant Turn (Reasoning)   │
│  + ChromaDB Dedup    │   │  Metadata (Post-Interaction)  │
│                      │   │  Critic (Cross-Model Judging) │
│                      │   └─ Reflect (Targeted Assistant) │
├──────────────────────┴───────────────────────────────────┤
│           SQLite (WAL) + ChromaDB + Cost Log              │
│  generations │ weight_costs │ failure_log │ topics        │
└──────────────────────────────────────────────────────────┘
```

### 🧠 Gelişmiş Özellikler (Phase 4 & 5)

- **Gelişmiş SFT Veriseti Disiplini**: Açık kaynak (Llama 3, vs) fine-tuning için üretilen JSONL verisi artık sadece sohbet mesajlarını değil; `topic`, `difficulty`, `domain`, `persona`, `winner`, ve `logic_score` gibi zengin meta etiketlerini de içerir (OpenAI Debate Dataset stili).
- **Ayrıştırılmış API-Key Load Balancing (Yük Dağıtımı)**: `🔑 API Keys` arayüzünden tek bir sağlayıcı (Örn: Gemini) için alt alta limitsiz sayıda API anahtarı eklenebilir. Sistem Rate Limit hatası alırsa tüm sağlayıcıyı DEĞİL, SADECE o spesifik anahtarı `2 Saatlik Cooldown (bekleme)` moduna alır. Başka geçerli anahtarlar varsa onlardan veya diğer modellerden kesintisiz devam eder.
- **Sıralı Ajan Diyaloğu (Granular Loop)**: Teacher artık tek seferde değil, her turda (User -> Asistan) ayrı model çağrıları yaparak gerçek bir çatışma dinamiği oluşturur.
- **Dinamik Hız ve Başarım Kontrolü**: Üretim hızı (pipeline speed) ve günlük token limitleri (örn: Gemini için 2M limit) UI üzerinden anlık olarak ayarlanabilir. Limit aşılırsa o sağlayıcı için üretim günlüğüne kadar durdurulur.
- **Kalibrasyon vs Üretim Modu**: İlk 500 üretim "Calibration" olarak işaretlenerek kalite kontrolü için optimize edilir.
- **Model Gözlemlenebilirliği (Observability)**: Her üretimin `model_used` (örn: `llama-3.3-70b-versatile`) etiketi doğrudan UI'dan takip edilebilir.

### 4 Ajan — 4 Görev

| Ajan | Görev | Model |
|---|---|---|
| **Research Agent** | Konu bulur, ChromaDB ile çeşitliliği garanti eder ve maliyeti loglar | Varsayılan (Gemini) |
| **Granular Teacher** | 6 aşamalı (3-tur) adversarial diyalog üretir. Her tur bağımsız akıl yürütür | Gemini (temp 0.8 / 0.3) |
| **Critic Agent** | Bağımsız LLM çağrısı ile kaliteyi 4 vektörde puanlar (Ağırlıklı Ortalama Sunucu-Tarafı) | Groq/DeepSeek (temp 0.1) |
| **Orchestrator** | Bütçe ve kaynak koruması yapar, ajanları koordine eder | — |

---

## 📊 Streamlit Admin Dashboard

Tarayıcıdan `http://SUNUCU_IP:8501` adresine girerek erişilir.

| Sayfa | Özellik |
|---|---|
| **📊 Dashboard** | Toplam üretim, **💰 Günlük/Toplam Maliyet**, PASS/FAIL oranı, Tier kartları |
| **📈 Drift Monitor** | 7 günlük rolling PASS oranı, güven trendi ve çatışma türü histogramı |
| **💬 Conversations** | **🧪 CAL / 🚀 PROD** etiketli sohbetleri balonlar halinde oku |
| **🎯 Weekly Planner** | Research Agent'a odak noktaları ata |
| **⚙️ Pipeline Control** | Start/Pause + Canlı Log Viewer |
| **📥 Export Dataset** | **Tier 1, 2, 3** veya Domain filtreleri ile `.jsonl` indir |

---

## 🛠 Kurulum

### Gereksinimler
- Python 3.10+
- API anahtarı: Gemini, Groq veya DeepSeek (en az bir tanesi)

### Lokal (Windows / Linux)
```bash
# 1. Klonla & Hazırla
git clone https://github.com/xCenny/Whusdata.git
cd Whusdata
cp .env.example .env # .env dosyasını düzenle

# 2. Kur & Başlat
pip install -r requirements.txt
python main.py
# Başka bir terminalde:
streamlit run src/ui.py
```

### CasaOS / Docker
```bash
# .env dosyasını oluştur ve docker-compose'u çalıştır
docker-compose up -d --build
# Dashboard: http://SUNUCU_IP:8501
```

> **CasaOS Custom Install:** App Store → Custom Install → Import → `docker-compose.yml` içeriğini yapıştır.

---

## 📁 Proje Yapısı

```
whusdata/
├── main.py                 # Orchestrator — ana döngü
├── src/
│   ├── db.py               # SQLite + ChromaDB (WAL mode)
│   ├── llm_client.py       # Multi-provider LLM client (Gemini/Groq/DeepSeek)
│   ├── graph.py             # LangGraph state machine (Generate → Critic → Reflect)
│   ├── researcher.py        # Wikipedia scraper + UI keyword entegrasyonu
│   ├── prompts.py           # Teacher, Critic ve Reflection system prompt'ları
│   └── ui.py                # Streamlit Admin Dashboard
├── start.sh                 # Docker entrypoint (pipeline + streamlit)
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
| `conversation_history` | TEXT (JSON) | `[{"role": "user", "content": "..."}, {"role": "assistant", "reasoning": "...", "content": "..."}]` |
| `persona_type` | TEXT | Kullanıcı profili (Skeptical, Provocative...) |
| `conflict_type` | TEXT | Çatışma türü (Factual Error, Logical Fallacy...) |
| `resolution_style` | TEXT | Çözüm tarzı (Socratic, Direct Evidence...) |
| `domain` | TEXT | Konu alanı (Physics, Psychology...) |
| `critic_status` | TEXT | PASS veya FAIL |
| `critic_confidence` | REAL | 0.0 – 1.0 arası ağırlıklı ortalama |
| `failure_type` | TEXT | NONE, LOGICAL_ERROR, HALLUCINATION vb. |

### `target_keywords`
| Sütun | Açıklama |
|---|---|
| `keyword` | Research Agent'ın odaklanacağı konu |
| `priority` | normal / high / critical |
| `week_label` | Hafta etiketi |

---

## 🗺 Yol Haritası (Roadmap)

- [x] Streamlit Admin Dashboard (Phase 3)
- [x] Turn-by-Turn Adversarial Machine (Phase 4)
- [x] Server-side Weighted Critic Scoring (Phase 1 Hardened)
- [x] Budget & Token Cost Monitoring (BudgetGuardian)
- [x] Persona & Conflict Distribution Control (Phase 2)
- [ ] HuggingFace Datasets oto-push (Tier 1)
- [ ] Prompt A/B testing framework

---

## 📄 Lisans

MIT License
