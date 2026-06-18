# Sunum Planı — VAD Using VLMs

> Bu doküman, projede denenen tüm yöntemleri sunum için organize eden bir kılavuzdur.
> Her bölüm sunumda 1-3 slayda karşılık gelir.

---

## 1. Giriş ve Problem Tanımı

### 1.1. Proje Amacı
- **Visual Anomaly Detection (VAD)** — endüstriyel bir konveyör bant üzerindeki push-button bileşenlerinin görsel anomalilerini tespit etmek.
- Klasik VAD yöntemleri (one-class classification) yerine **Vision-Language Model (VLM)** tabanlı bir yaklaşım denenmiştir.

### 1.2. Senaryo (Use Case)
- Yeşil bir konveyör bant üzerinde, push-button bileşenleri **soldan sağa** doğru ilerlemekte.
- **Normal durum:** Butonun **kırmızı kapağı sağa**, **4-pinli kısmı sola** bakar.
- **Anomali durumu:** Yön ters çevrilmiş (kırmızı kapak solda) → **Reversed Orientation Anomaly**.
- Diğer anomaliler: butonun durması (jam), bant kenarından düşmesi, dönmesi (rolling).

### 1.3. Neden VLM?
- Geleneksel VAD modelleri her ortam için yeniden eğitilmek zorunda → maliyet ve veri ihtiyacı yüksek.
- VLM'ler **zero-shot** çalışabilir → kullanıcı doğal dilde "anomali" tanımı yapabilir.
- **AnyAnomaly (CVPR 2025)** paperı bu yaklaşımın etkili olduğunu göstermiş.

---

## 2. Proje Mimarisi — İki Ana Yaklaşım

Proje iki ayrı klasör altında **iki farklı yaklaşımı** birleştirir:

| Klasör | Yaklaşım | Model | Çalışma Yeri |
|---|---|---|---|
| `qwen_vl/` | **Cloud API** (DashScope) | Qwen3-VL-Plus | Sunucu (uzak) |
| `qwen_lighthing/` | **Local inference** (HuggingFace) | Qwen3-VL-8B-Instruct | GPU (lokal) |

> Sunum boyunca bu iki yaklaşımın **avantaj/dezavantajları** karşılaştırılmalıdır.

---

## 3. API Tabanlı Yaklaşımlar (`qwen_vl/`)

### 3.1. `test_api_2.py` — Tüm Video API'ye
- Video doğrudan **DashScope API**'ye gönderilir.
- Model tüm video klibini analiz eder.
- **Avantaj:** Basit, az kod.
- **Dezavantaj:**
  - Sol/sağ yön karışıklığı (model "red cap is on the right" diye halüsinasyon yapıyor)
  - Kısmi görünür frame'lerde yanlış değerlendirme
  - "Default to normal" bias'ı

### 3.2. `test_api_by_frame.py` — Temporal Grid (TC) Yaklaşımı
- **AnyAnomaly paperından ilham:** Temporal Context (TC) bileşeni
- Videodan **4 eşit aralıklı frame** çekilir → **2×2 grid image** oluşturulur.
- Bu grid tek bir görsel olarak modele gönderilir.
- Frame'lere etiket (Frame 1/2/3/4) ve bant yönü oku eklenir.

**Neden bu yaklaşım?**
- Model temporal akışı tek bir karede görüyor.
- Sol/sağ yön analizi daha güvenilir hale geliyor.
- Kısmi görünürlük problemi azalıyor.

### 3.3. Promptu Geliştirme Süreci (Iteration Log)
Sunumda **iterasyon hikâyesi** olarak anlatılabilir:

| Versiyon | Sorun | Çözüm |
|---|---|---|
| v1 — Genel prompt | Spesifik anomali tanımı yok | Specific normal/anomaly tanımı eklendi |
| v2 — Sol/sağ ifadesi | Model yön karıştırıyor | "Leading/trailing" terminolojisine geçildi |
| v3 — Kısmi görünürlük | Boş frame'lerde halüsinasyon | "Only analyze fully-visible frames" eklendi |
| v4 — Halen yanılma | Model yön bilmiyor | Grid image + frame etiketleri + yön oku |

### 3.4. API Yaklaşımının Limitleri
- API çağrısı başına **latency yüksek** (~5-15 sn)
- Maliyet token bazlı
- Model kararı tek seferlik → güvenilirlik düşük

---

## 4. Local Inference Yaklaşımı (`qwen_lighthing/`)

### 4.1. Genel Mimari — `test_with_persons.py`
- **Multi-pass voting** yaklaşımı: Aynı video birden fazla "persona" prompt'uyla 5-6 kez analiz edilir.
- Her persona farklı bir profesyonel bakış açısı temsil eder.
- Sonuçlar **majority voting** ile birleştirilir.
- Çıktı: Final verdict + confidence + JSON rapor.

**Neden voting?**
- VLM'ler tek seferlik kararlarda tutarsız olabilir.
- Farklı perspektifler birbirinin yanlışını dengeleyebilir.
- Confidence skoru elde edilir.

### 4.2. Persona Ensembling Mimarisi
- 20 farklı persona tanımlandı (operatörler, mühendisler, denetçiler vb.)
- **5 persona seçildi** (`VOTING_SUBSET`):
  - **P02** — Yeni operatör (kuralcı, naif)
  - **P05** — Makine mühendisi (geometrik bakış)
  - **P08** — Computer vision mühendisi (piksel bazlı)
  - **P15** — Kıdemli uzman (Toyota / Jidoka)
  - **P20** — Hibrit (kural + sezgi)
- Ek olarak 6. oy: **CONSENSUS_PROMPT** (tüm 20 persona'nın ortak özü)

---

## 5. ★ KRİTİK: `personas_specialized.py` vs `personas_complete.py` ★

> **Bu sunumun en önemli kısımlarından biri.** İki dosyanın farkı, projenin ana öğreniminin kanıtı.

### 5.1. Ortak Noktalar
- Her iki dosyada da **aynı 20 persona** vardır (P01–P20).
- Aynı `VOTING_SUBSET` (P02, P05, P08, P15, P20).
- Aynı `CONSENSUS_PROMPT` mantığı (küçük farklarla).
- Aynı domain bilgisi: red cap RIGHT = NORMAL, red cap LEFT = ANOMALY.

### 5.2. ★ Temel Fark: "Articulate-First" Format ★

#### `personas_specialized.py` (önceki versiyon)
- Her persona **kendi özgün output formatına** sahip.
- Bazıları **kompakt YES/NO** formatında.
- **Sorun:** Model, frame'leri gerçekten incelemeden direkt "NORMAL" sonucuna atlıyordu.
- Model **prior bias**'a göre karar veriyordu, görselleri analiz etmiyordu.

**Örnek (P02 — specialized):**
```
STEP 1: ...
STEP 2: ...
RESULT: ANOMALY DETECTED / NO ANOMALY DETECTED
```
→ Model "RESULT: NO ANOMALY" yazıp geçebiliyor, kanıt sunmadan.

#### `personas_complete.py` (geliştirilmiş versiyon)
- Her persona **önce frame frame gözlem yapmak** zorunda.
- **Sonra** her check için (A/B/C) verdict veriyor.
- **En son** final result.

**Örnek (P02 — complete):**
```
WHAT I SEE IN EACH FRAME:
  Entry  : [describe the button location and which side red is on]
  Middle : [describe the button location and which side red is on]
  Exit   : [describe the button location and which side red is on]

STEP A - ORIENTATION (based on my observations):
  - The red cap was on the [LEFT / RIGHT / UNCLEAR] side
  - Verdict for A: [PASS / FAIL / UNCLEAR]
...
RESULT: ANOMALY DETECTED / NO ANOMALY DETECTED
```
→ Model verdict'i **kendi gözlemlerine bağlamak** zorunda. Halüsinasyon zorlaşıyor.

### 5.3. ★ Öğrenilen Ders ★
> **"Force articulation BEFORE verdict"** prensibi.

LLM'lere "evet/hayır" soran formatlar **prior bias**'a yenilir.
Modeli **gözlemlerini sözel olarak ifade etmeye** zorlamak, halüsinasyonu büyük ölçüde azaltır.

Bu, projenin **en değerli teknik bulgusudur** ve sunumda ön plana çıkarılmalıdır.

### 5.4. Diğer Yapısal Farklar (özet)

| Özellik | specialized | complete |
|---|---|---|
| Output formatı | Çeşitli (her persona farklı) | Yapılandırılmış, birleşik |
| Frame-by-frame gözlem | Opsiyonel / yok | **Zorunlu** |
| Shared checklist | Yok (persona-specific) | `SHARED_CHECKLIST` (ortak A/B/C) |
| Articulation rule | Yok | `ARTICULATION_RULE` (zorunlu) |
| Anomali kategorileri | A: orientation | A: orientation, B: motion, C: belt position |
| Kullanım amacı | İlk versiyon, problemi gösterdi | Çözüm versiyonu |

---

## 6. AnyAnomaly Paperı ile İlişkilendirme

### 6.1. Paperdan Alınan Fikirler
- **Temporal Context (TC)** — `test_api_by_frame.py`'de uygulandı (2×2 grid).
- **Context-aware VQA** — Persona prompt'larında "describe before judging" yaklaşımı.
- **Scoring approach** (0.0–1.0) — API yaklaşımında denenmiştir.

### 6.2. Paperdan Uyarlanmayan / Uyarlanamayan Kısımlar
- **Position Context (PC)** — WinCLIP gerektirir, lokal CLIP modeli gerekli (yapılmadı).
- **Key Frame Selection Module (KSM)** — CLIP tabanlı, lokal model gerekli (basit eşit-aralıklı seçimle değiştirildi).

### 6.3. Bizim Katkımız
- **Persona ensembling** — paperda olmayan, multi-perspective voting.
- **Articulate-first prompt design** — halüsinasyonu engelleyen yapısal bir prompt değişikliği.

---

## 7. Yaşanan Zorluklar ve Çözümler

| Zorluk | Çözüm Denenen | Sonuç |
|---|---|---|
| Sol/sağ karışıklığı | "Leading/trailing" terminolojisi | Kısmi başarı |
| Sol/sağ karışıklığı | Frame etiketleri + yön oku | Kısmi başarı |
| Sol/sağ karışıklığı | OpenCV color detection (önerildi) | **Henüz uygulanmadı, gelecek iş** |
| Default-to-normal bias | Articulate-first prompt | **Önemli iyileşme** |
| Tek seferlik karar tutarsızlığı | 5-persona voting | **Confidence iyileşti** |
| Kısmi görünürlük | "Only analyze fully-visible frames" | Kısmi başarı |
| Model halüsinasyonu | Çoklu perspektif + CONSENSUS prompt | **Önemli iyileşme** |

---

## 8. Sonuçlar ve Karşılaştırma

### 8.1. Yaklaşımların Karşılaştırması (sunumda tablo)

| Kriter | API + Video | API + TC Grid | Local + Voting |
|---|---|---|---|
| Latency | Orta | Orta | Yüksek |
| Maliyet | Token bazlı | Token bazlı | GPU sahipliği |
| Doğruluk | Düşük | Orta | **Yüksek** |
| Tutarlılık | Düşük | Orta | **Yüksek** |
| Setup karmaşıklığı | Düşük | Düşük | **Yüksek** |

### 8.2. Önerilen Final Pipeline
1. **Local inference** + **complete personas** + **5-voter majority**
2. Düşük confidence durumlarında ek voter (CONSENSUS prompt) ekle
3. Çok kritik vakalar için **OpenCV color detection** ile cross-check

---

## 9. Gelecek Çalışmalar

- **OpenCV ile renk tabanlı orientation tespiti** — VLM yerine deterministik backup
- **Position Context (PC)** uygulaması — WinCLIP entegrasyonu
- **CLIP-based KSM** — eşit aralıklı yerine en alakalı frame seçimi
- **Daha geniş anomali tipleri** (renk farkı, fiziksel hasar, vb.)
- **Edge deployment** — local model'in optimize edilmesi (örn. Qwen3-VL-3B veya quantization)

---

## 10. Sunum İçin Öneriler

### Slayt sırası önerisi:
1. Giriş + problem (1-2 slayt)
2. Use-case demo (görseller + video) (1 slayt)
3. AnyAnomaly paperı kısa özeti (1 slayt)
4. İki yaklaşım: API vs Local (1 slayt — mimari diagram)
5. API yaklaşımı + iterasyon hikâyesi (2 slayt)
6. Local yaklaşımı + voting mimarisi (1 slayt)
7. **★ Specialized vs Complete farkı ★** (2 slayt — bu en kritik kısım)
8. Articulate-first prensibi (1 slayt — vurgulanmalı)
9. Karşılaştırma tablosu (1 slayt)
10. Sonuç + gelecek iş (1 slayt)

### Demo önerileri:
- Sunumda **iki video** göster: biri normal, biri anomali (kırmızı kapak solda).
- Her iki video için **specialized vs complete** çıktılarını yan yana göster.
- "Specialized" persona'nın frame'i incelemeden "NORMAL" dediği bir örnek bulunup gösterilirse en güçlü etki yaratır.

### Anahtar mesajlar (sunumun "take-aways"):
1. **VLM'ler zero-shot anomali tespitinde umut verici ama promptlamaya çok hassas.**
2. **"Önce gözlem, sonra karar" prensibi LLM halüsinasyonunu büyük ölçüde azaltıyor.**
3. **Multi-perspective voting tek-seferlik kararlardan daha güvenilir.**
4. **Hibrit yaklaşımlar (VLM + klasik CV) en pratik çözüm olabilir.**
