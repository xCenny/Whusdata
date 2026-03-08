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
- **Dinamik Model ve Fiyatlandırma Yönetimi**: Eski sabit model konfigürasyonları yerine, tamamen veritabanı destekli (SQLite) dinamik `🤖 Models & Prices` sayfası eklenmiştir. Bu sayfadan istediğiniz LLM uç noktasını ekleyebilir, 1 Milyon Token input/output maliyetlerini kuruşu kuruşuna girebilirsiniz. Böylece sistem, veri üretirken **gerçek** dolar maliyetini hesaplar.
- **Ücretsiz Katman (Free Tier) Gecikme Koruması**: Ücretsiz API'ler kullananlar için (örn. Groq veya Gemini Free), modeller arası `Delay (s)` yani yapay bekleme süresi eklenebilir. Böylece rate limit aşımlarına karşı üretim güvene alınır.
- **API Kurtarma (Discard Prevention)**: Olası bir Rate Limit (429) veya API çökmesi durumunda sistem, tamamlanmış uzun diyalogları çöpe atmak yerine tekrar denemek üzere `PENDING` konumuna alarak API maliyet israfını engeller.
- **Ayrıştırılmış API-Key Load Balancing (Yük Dağıtımı)**: `🔑 API Keys` arayüzü artık veritabanındaki aktif modellere göre dinamik olarak şekillenir. Tek bir sağlayıcı için limitsiz sayıda API anahtarı eklenebilir. Sistem Rate Limit hatası alırsa SADECE o spesifik anahtarı `2 Saatlik Cooldown (bekleme)` moduna alır.
- **Sıralı Ajan Diyaloğu (Granular Loop)**: Teacher artık tek seferde değil, her turda ayrı model çağrıları yaparak gerçek bir çatışma dinamiği oluşturur.
- **Dinamik Hız ve Başarım Kontrolü**: Üretim hızı (pipeline speed) ve günlük token limitleri (örn: Gemini için 2M limit) UI üzerinden anlık olarak ayarlanabilir.
- **Kalibrasyon vs Üretim Modu**: İlk 500 üretim "Calibration" olarak işaretlenerek kalite kontrolü için optimize edilir.

### 🌐 Derin Bilgi & Acımasız Kalite Kontrolü (Phase 6)
- **Dinamik Bilgi Kaynakları (Knowledge Sources)**: Research Ajanı artık sadece parametrik yapılandırılmış Wikipedia'dan değil; UI üzerinden eklenebilen rastgele Reddit Subreddit'leri (ör: r/MachineLearning), Hacker News sıcak tartışmaları ve Global RSS haber akışlarından beslenerek tamamen öngörülemez, taze konu başlıkları bulur.
- **Derin İnternet Bilgisi (Internet Knowledge)**: Eğer dinamik akışlar devredışıysa sistem; Kuantum Dolanıklığı, Epigenetik Miras, Fermi Paradoksu gibi akademik ve derin paradoksları araştırarak hibrit bir tohumlama yapar.
- **Çeşitlilik Zorunluluğu (Topic Diversity)**: ChromaDB Vektör Veritabanı benzeşim eşiği %70'e çekilmiştir. Sistem, birbirine hafif benzeyen konuları bile reddeder; sadece tamamen 'Novel' (Özgün) konuları sisteme alır.
- **Acımasız Eleştirmen (Brutal Critic) & Çan Eğrisi**: Critic Prompt'u "ChatGPT Sendromu"nu önlemek adına sıfır toleransla baştan yazıldı. Asistan, argümanında "Haklısın" gibi yumuşatıcı ifadeler ('Soft Validation') kullanırsa veya mantıksal boşluk bırakırsa o diyalog anında `FAIL` alır ve çöpe atılır. Hakemin bu aşırı katı puanlamasını ve haksız reddedişleri dengelemek için başarı eşiklerinde **Çan Eğrisi (Calibration)** uygulanır: 🥇 Tier 1 (Gold) >= **0.75**, 🥈 Tier 2 (Silver) >= **0.65**, 🥉 Tier 3 (Bronze) >= **0.55**.

### 🎯 Odaklı Derin Araştırma & Baş Hakem Sistemi (Phase 7 & 8)
- **Atanmış Baş Hakem (Dedicated Critic Model)**: UI üzerinden `Pipeline Control` sayfasından, tüm konuşmaları değerlendirecek tek bir **Baş Hakem LLM** (Örn: Llama-3.3-70b veya Gemini-2.5-Pro) seçebilirsiniz. Diğer ajanlar kendi aralarında paslaşıp üretse de, son kararı her zaman güvendiğiniz bu baş hakem verir; böylece Tier 1 standartları platformlar arası şaşmaz.
- **Sürekli Odaklı Araştırma (Continuous Deep Research)**: "Weekly Focus Planner" üzerinden verdiğiniz hedefler (örn: Kuantum Fiziği) artık tek seferlik değildir. Sistem o hedefe bir "Ağırlık Derecesi (Priority)" atar (`Critical` konular 5 kat daha sık araştırılır) ve belirlenen hafta bitene kadar o konunun dibine kadar iner. Veri zehirlenmesini önlemek için de araya rastgele Wikipedia ansiklopedi bilgileri harmanlar (Hybrid Mix).
- **API Sağlık Monitörü (API Health Monitor)**: Sisteme girdiğiniz tüm API Anahtarları UI'da anlık olarak izlenir. Bir anahtar `Authentication Error` veya `Rate Limit` yediğinde bunu anında kırmızı/sarı SQLite uyarılarıyla dashboard'a yansıtır ve o anahtarı otomatik 2 saatlik cezaya (Cooldown) gönderip diğerlerinden tam gaz devam eder.

### 🥊 Adversarial Reasoning Evolution & Analytics (Phase 9-12)
- **Gelişmiş Sentetik Akıl Yürütme (8-12 Mesaj)**: Diyaloglar artık sabit 3-tur değil, otonom olarak **3 ile 6 tur (6-12 toplam mesaj)** arasında derinleşmektedir. Bu, modelin uzun süreli tartışmaları ve karmaşık mantık zincirlerini (synthetic reasoning) öğrenmesini sağlar.
- **Yapay Zeka Destekli Otonom Bitiş (LLM-Driven Termination)**: Konuşmalar hedef tura ulaşmasa bile, Asistan (Teacher) karşı tarafın tekrara düştüğünü veya konunun saptığını (Logical Drift) hissederse `conclude_debate: true` bayrağı ile tartışmayı anında ve otonom olarak bitirip hakeme yollar. Token ve kalite israfı sıfıra iner.
- **Hatalı/Çöpe Atılan Veri İzleme (Failed Conversation UI)**: Hakem (Critic) tarafından düşük puan alıp çöpe atılan veriler artık silinmez. UI üzerinden (Tier 0 olarak) tüm "FAIL" alan konuşmaları, mantıksal çöküş noktalarını ve "Critic Feedback" bölümünü satır satır şeffaf bir şekilde inceleyebilirsiniz.
- **Dengeli Persona Dağılımı**: "Twitter Trolü" eğitimini önlemek için personallar dengelenmiştir: %30 Hostile/Troll, %40 Normal Şüpheci, %20 Eğitici/Öğrenci ve %10 Absürt/Komplo Teorisyeni.
- **Değişken Cevap Uzunlukları**: LLM'lerin bitmek bilmeyen paragraf yazma (500-word essay) sendromu aşılmış; %30 Kısa/Punchy (2-3 cümle), %40 Orta ve %30 Detaylı açıklama dengesi getirilmiştir.
- **Mantık Saldırıları (Logic Attacks)**: Kullanıcı ajanı; *Strawman*, *Ad Hominem*, *False Dilemma* gibi safsataları bilinçli kullanır; asistanın daha önce söylediği ile çeliştiğini iddia eden tuzaklar kurar (Contradiction Trap) ve sürekli kanıt ("Source?") talep eder.
- **Halüsinasyon Koruması (Citation Hedging)**: Asistan, kullanıcıdan gelen kaynak taleplerine karşılık sahte DOI veya dergi sayısı uydurmak yerine, "Citation Uncertainty" kuralı ile belirsizliği dürüstçe itiraf edip genel literatür bilgisi sunmaya (Hedging) zorlanır.
- **Contextual Memory Recall (Organik Hafıza)**: Asistan (Teacher), 2. ve 3. tur cevaplarında kullanıcının önceki konuşmalarında kullandığı spesifik bir kelimeye, hatalı bir analojiye veya safsataya doğrudan *alaycı olmayan bir gönderme* (Recall) yaparak, konuşmanın kopuk parçalar değil **derin bir multi-turn zincir** olduğunu kanıtlar. Bu özellik, modelin bağlam (context) takibini zirveye çıkarır.
- **Deep RLHF Analytics (Derin Analitik)**: Hakem (Critic) modeli artık sadece "Geçti/Kaldı" veya yüzeysel bir skor vermez. `reasoning` (Chain of Thought felsefesi), `detected_fallacies` (Kullanıcının yaptığı mantık hataları listesi), `assistant_counters` (Asistanın savunmaları) ve `factual_score` (Bilgi Doğruluğu) parametrelerini JSON olarak üretir ve dışa aktarılan `.jsonl` verisetine enjekte eder.
- **Token Kullanım Grafikleri**: Dashboard üzerinde son 14 günlük token harcamalarınızı gösteren interaktif bar tabloları eklenmiştir.

### 4 Ajan — 4 Görev

| Ajan | Görev | Model |
|---|---|---|
| **Research Agent** | Konu bulur, ChromaDB ile çeşitliliği garanti eder ve maliyeti loglar | Varsayılan (Gemini) |
| **Granular Teacher** | 8-12 mesajlık adversarial diyalog üretir. Dinamik uzunluk ve taktik uygular | Gemini (temp 0.8 / 0.3) |
| **Critic Agent** | Bağımsız LLM çağrısı ile kaliteyi 4 vektörde puanlar. Safsata kaçırmayı affetmez. | Groq/DeepSeek (temp 0.1) |
| **Orchestrator** | Bütçe ve kaynak koruması yapar, ajanları koordine eder | — |

---

## 📊 Streamlit Admin Dashboard

Tarayıcıdan `http://SUNUCU_IP:8501` adresine girerek erişilir.

| Sayfa | Özellik |
|---|---|
| **📊 Dashboard** | AI-Driven Öngörüler (Maliyet Tahmini), Toplam Üretim, **💰 Maliyet**, Tier kartları |
| **📈 Drift Monitor** | 7 günlük rolling PASS oranı, güven trendi ve çatışma türü histogramı |
| **💬 Conversations** | **🧪 CAL / 🚀 PROD** etiketli sohbetleri okuma ve Critic kararlarını inceleme |
| **🎯 Weekly Planner** | Research Agent'a hedef anahtar kelimeler ve öncelikler atama |
| **⚙️ Pipeline Control** | Üretimi Başlat/Durdur, Bekleme sürelerini değiştir ve Canlı Log izleme |
| **🤖 Models & Prices** | Veritabanı destekli Dinamik LLM sağlayıcı, Fiyat ve Free Tier Gecikme ayarları |
| **🔑 API Keys** | Tüm sağlayıcıların API anahtarlarını, Cooldown sağlık durumlarını ve günlük limitleri yönetme |
| **📥 Export Dataset** | **Tier 1, 2, 3** veya Domain filtreleri ile SFT formatlı `.jsonl` indirme |

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
│   ├── llm_client.py       # Multi-provider LLM client & Regex JSON Extractor
│   ├── graph.py             # LangGraph state machine (Adversarial Persona Loops)
│   ├── researcher.py        # Multi-Source Fetcher (Wikipedia Jump & ArXiv API)
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
- [x] Persona & Conflict Distribution Control (Phase 9)
- [x] Anti-Hallucination & Extended Reasoning (Phase 10)
- [ ] HuggingFace Datasets oto-push (Tier 1)
- [ ] Prompt A/B testing framework

---

## 📄 Lisans

MIT License
