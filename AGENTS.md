# AGENTS.md — VAD Using VLMs (Graduation Project)

## Proje Özeti

Endüstriyel konveyör bantlarında **Video Anomaly Detection (VAD)** gerçekleştirmek için
**Vision-Language Models (VLM)** API'larını kullanan bir araştırma projesi.
Eğitim gerektirmeyen (zero-shot / few-shot) yaklaşım benimseniyor.

**Model**: Qwen-VL (`qwen-vl-plus`, `qwen-vl-max`) via DashScope API  
**Environment**: `conda activate grad2_env` (Python 3.10)  
**API Key**: `.env` → `DASHSCOPE_API_KEY`

---

## Dataset: IPAD

`/home/yunus/projects/vad_using_one_shot_learning/dataset/IPAD_dataset/`

IPAD (Industrial Process Anomaly Detection, 2024) — endüstriyel VAD için benchmark dataset.
Her senaryo `testing/frames/<clip_id>/` (JPG) + `test_label/<clip_id>.npy` (0/1) içeriyor.

| Senaryo | İçerik | Anomali Türleri |
|---------|--------|-----------------|
| **R01** | Kırmızı push-button konveyör | ANGLE ANOMALY (pin yönü), COLOR ANOMALY |
| **S08** | Kırmızı/mavi kutu sıralama sistemi | SORTING ERROR, CLOGGING |

Ground truth: `.npy` dosyasında her frame için `0.0` (normal) veya `1.0` (anomali).

---

## Mevcut Denemeler (qwen_vl/)

| Dosya | Yöntem | Açıklama |
|-------|--------|----------|
| `test_api.py` | **Zero-shot, tek görüntü** | R01 için tek frame → 3 farklı prompt denenmiş (basit, CoT, attribute-based) |
| `test_fewshot_api.py` | **Few-shot (1-shot)** | Referans normal frame + test frame birlikte gönderilip karşılaştırılıyor |
| `test_sequence_api.py` | **Multi-frame sequence** | S08 için 8 frame ayrı image olarak gönderiliyor, sliding window ile |
| `test_video_api.py` | **Frame→Video clip** | R01 için seçilen frameler OpenCV ile MP4'e çevrilip base64 video olarak gönderiliyor |
| `test_video_describe.py` | **Video betimleme** | Model'in clip'i nasıl yorumladığını anlamak için tanımlayıcı sorgulama |
| `video_understanding.py` | **Ham video gönderimi** | Tam video dosyası base64 olarak gönderiliyor |

### Kritik Parametreler (`test_video_api.py` / `test_sequence_api.py`)
- `SAMPLE_STEP`: Her N. frame alınır (R01: 5, S08: 20)
- `SEQUENCE_LEN`: Sliding window boyutu (R01: 10, S08: 8)
- `SKIP_FIRST_N`: Başlangıç frame'leri atla
- `VIDEO_FPS`: Oluşturulan clip'in FPS'i (R01: 4)

---

## Bilinen Sorunlar ve Dikkat Edilmesi Gerekenler

1. **Payload boyutu**: Base64 video gönderimi > 20 MB için API reddedebilir.
   → `test_video_api.py`'daki `frames_to_video` fonksiyonu geçici MP4 oluşturur, sonra siler.

2. **API key normalizasyonu**: DashScope key'i `sk-` prefix gerektiriyor.
   → `_normalize_api_key()` fonksiyonu her test dosyasında tekrar ediyor.

3. **Ground truth hizalaması**: Frame indeksleri `.npy` array indeksleriyle birebir eşleşmeli.
   → `get_sequence_ground_truth()`: sekans içinde herhangi bir frame anomaliyse tüm sekans anomali sayılır.

4. **Model seçimi**: `qwen-vl-plus` daha hızlı/ucuz, `qwen-vl-max` video ve ince detaylar için daha iyi.

5. **Prompt determinizmi**: Model yanıtları aynı input için her zaman aynı değil.
   → Sonuçlar tekrar çalıştırmada değişebilir; istatistiksel değerlendirme için birden fazla run gerekir.

---

## Prompt Mühendisliği — Öğrenilen Dersler

- **PROMPT (basit)**: Tek paragraf açıklama + "Reply with exactly one of" → model bazen format dışına çıkıyor.
- **PROMPT2**: Sadece renk soruyor → yetersiz.
- **PROMPT3 (CoT)**: Step 1 (gözlem) + Step 2 (kural tabanlı karar) formatı → en tutarlı sonuçlar.
  ```
  Step 1 - Observe: What color is the cap? Which direction do the pins point?
  Step 2 - Classify using ordered rules.
  Reply in exact format: Cap color: / Pins: / Result:
  ```
- **Few-shot**: Referans normal görüntü eklemek, özellikle ince renk farklarında yardımcı oluyor.

---

## Sonraki Adımlar — Araştırma Yönleri

Aşağıdaki fikirler 2024-2025 literatürüne dayanmaktadır.
Bunlar uygulanmamış, deneysel önerilerdir.

### 1. Prompt Ayrıştırma (Anomaly-Specific Prompts)
Her anomali türü için ayrı, odaklı prompt:
- Renk için: "What is the exact color of the cap/button top?"
- Açı için: "Which direction do the connector pins point? (left/right/up/forward/diagonal)"
- İki ayrı sonucu birleştirerek nihai karar ver.
- Referans: *Unlocking VLMs for VAD via Fine-Grained Prompting* (arXiv 2510.02155)

### 2. Ensemble Prompting
Aynı sekans için 2-3 farklı prompt varyantı çalıştır, çoğunluk oyuyla karar ver.
→ Tek seferlik yanlış cevapların etkisini azaltır.

### 3. Daha İyi Frame Seçimi (Key-Frame Selection)
- Düzgün aralıklarla frame almak yerine, hareketi veya değişimi en iyi temsil eden frame'leri seç.
- Basit yaklaşım: her N frame içinden en yüksek piksel farkı olan frame'i al.

### 4. Qwen2.5-VL Geçişi
Mevcut `qwen-vl-max` yerine Qwen2.5-VL denemesi:
- 20+ dakika video desteği, daha iyi temporal reasoning.
- DashScope'ta `qwen-vl-max-latest` veya `qwen2.5-vl-max` ile test edilebilir.

### 5. VERA / Sorgulama Tabanlı Yaklaşım (CVPR 2025)
Tek büyük prompt yerine, modele önce birkaç odaklı soru sor:
1. "Is the button red?" → YES/NO
2. "Do the pins point left?" → YES/NO
3. Bu yanıtları kural tabanlı bir karar ağacından geçir.
- Referans: *VERA: Explainable VAD via Verbalized Learning* (CVPR 2025)

### 6. Basit Değerlendirme Pipeline'ı
Şu an her test dosyası kendi accuracy'sini hesaplıyor ama sistematik değil.
- Tüm clip'ler için bir döngü yaz, sonuçları CSV'e kaydet.
- Precision / Recall / F1 hesapla (sadece accuracy yanıltıcı olabilir, sınıflar dengesizse).

---

## İlgili Makaleler

| Makale | Konu | Link |
|--------|------|------|
| IPAD Dataset (2024) | Kullandığımız dataset | arXiv 2404.15033 |
| Unlocking VLMs for VAD via Fine-Grained Prompting (2024) | Prompt optimizasyonu | arXiv 2510.02155 |
| VERA: Explainable VAD (CVPR 2025) | Sorgulama tabanlı anomali tespiti | CVPR 2025 |
| Towards Zero-Shot AD with MLLMs (CVPR 2025) | Multi-stage reasoning | CVPR 2025 |
| Quo Vadis, Anomaly Detection? (2024) | LLM/VLM ile VAD genel bakış | arXiv 2412.18298 |
| AnomalyCLIP (ICLR 2024) | Zero-shot CLIP tabanlı anomali tespiti | GitHub: zqhang/AnomalyCLIP |
| Qwen2.5-VL Technical Report (2025) | Model mimarisi | arXiv 2502.13923 |

---

## Proje Yapısı

```
vad_using_VLMs_graduation_project/
├── AGENTS.md               ← bu dosya
├── README.md               ← kurulum talimatları
├── requirements.txt
├── .env                    ← API key (git'e commit edilmez)
├── .env.example
├── data/
│   ├── *.jpg               ← R01 örnek frame'leri
│   ├── ornek_video*.mp4    ← çeşitli FPS/çözünürlük test videoları
│   ├── r01_clip06_angle_anomaly.mp4
│   └── r01_clip09_normal.mp4
└── qwen_vl/
    ├── test_api.py          ← tek frame, zero-shot (R01)
    ├── test_fewshot_api.py  ← few-shot, 2 frame karşılaştırma (R01)
    ├── test_sequence_api.py ← multi-frame sequence (S08)
    ├── test_video_api.py    ← frame→video clip (R01)
    ├── test_video_describe.py ← betimleme testi
    └── video_understanding.py ← ham video gönderimi
```
