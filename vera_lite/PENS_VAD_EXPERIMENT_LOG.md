# VERA-lite — Kalem (Pens) Dataset Deney Günlüğü

**Tarih:** 2026-06-09
**Model:** `qwen3.6-plus` (DashScope, intl endpoint)
**Ortam:** WSL Ubuntu-22.04 + `grad2_env` venv (`vera_lite/run_wsl.sh` yardımcı script)
**Amaç:** Mevcut VERA-lite iş akışını (R01 push-button sahnesi) yeni **kalem konveyör** veri setine uyarlamak ve anomali tespiti yapmak.

---

## 1. Veri Seti

`data/pens/` altında 14 ham video (telefonla çekim). Özellikler:
- **Çözünürlük:** 2160×3840 (4K, dikey) · **FPS:** 30 · **Süre:** ~17-27 sn · **Boyut:** 100-170 MB

**Sahne (rehber + modelin describe çıktısından):** Açık ahşap tezgah üzerinde **lineer ray + vidalı mil (lead screw)** düzeneği; üstünde kayan **beyaz dikdörtgen levha** soldan sağa hareket ediyor ve **3 identical kalem** taşıyor. (Not: sahne bir "yeşil konveyör bandı" değil; model tutarlı şekilde lineer ray düzeneği görüyor.)

### Anomali türleri (13 `pens_*` klibi)
| Klip | Sınıf | Beklenen |
|------|-------|----------|
| `pens_normal`, `_2`, `_3` | Normal | NORMAL |
| `pens_missing_cap`, `_2` | Bir kalemin kapağı yok | ANOMALY |
| `pens_wrong_direction`, `_2` | Bir kalem 180° ters | ANOMALY |
| `pens_wrong_color` | Bir kalem farklı renk | ANOMALY |
| `pens_no_pen` | Kalem eksik (sayı < 3) | ANOMALY |
| `pens_conveyor_stops` | Bant durur/donar | ANOMALY |
| `pens_conveyor_stops_and_continues` | Bant duraklar sonra devam | ANOMALY |
| `pens_temporal_anomaly` | Kalemler kendi başına oynar (görünmeyen dış etki) | ANOMALY |
| `pens_wrong_direction_and_missing_cap` | Birleşik (ters + kapaksız) | ANOMALY |

---

## 2. Önişleme — Video Küçültme

**Neden:** 4K@30fps videolar DashScope için çok büyük (ret riski + aşırı `video_tokens` maliyeti).

**Ne yaptık:** `vera_lite/compress_videos.py` ile çözünürlüğü **2160×3840 → 720×1280** indirdik (9:16 korundu, **FPS'e dokunulmadı** — kullanıcı tercihi). Çıktı: `data/pens_small/`.

**Sonuç:** Boyut 100-170 MB → **~5.5-8.4 MB** (~18× küçülme). Anomali ipuçları (kapak, uç yönü, renk, boşluk) korundu.

---

## 3. Adım 1 — Describe (önyargısız betimleme)

**Neden:** `SYSTEM_CONTEXT`'i körlemesine yazmak yerine, **modelin sahneyi gerçekte nasıl algıladığını** öğrenmek. Sınıflandırma yok, sadece nötr gözlem.

**Ne yaptık:** `vera_lite/test_qwen3.6.py`'daki R01-özgü prompt'u kalem-nötr hale getirdik (renk/kapak/yön/sayı/hareketi açık uçlu soran). `normal`, `missing_cap`, `wrong_color` kliplerinde çalıştırdık. (Çıktılar: `vera_lite/runs/desc_pens_*.txt`)

**Öğrenilenler:**
- Model 3 kalemi doğru **sayıyor**; sahneyi lineer ray olarak görüyor.
- **Renk algısı zayıf:** kalemleri "grimsi/şeffaf" görüyor, "mavi" demiyor → mutlak renk yerine **göreli ("üçü aynı mı")** tanım gerekli.
- **Kapak belirsizliği:** "sökülebilir kapak mı sabit tutuş mu" emin değil → kapağı **kalemleri birbiriyle kıyaslayarak** sorgulamak gerekli.
- Anomalileri (eksik kapak, renk farkı) describe modunda **doğru yakaladı** → göreli kıyas stratejisi sağlam.

---

## 4. Adım 2 — Prompt + Sorular + Manifest

**Ne yaptık:**
- `vera_lite/prompts.py` `SYSTEM_CONTEXT` → kalem sahnesine yeniden yazıldı; **göreli kıyas** mantığı, "renk mutlak değil göreli", "sayımı levha tam görünürken yap", giriş/çıkış artefaktlarını yok say.
- `vera_lite/guiding_questions.json` (**v2**) → 5 soru, her biri bir anomali türüne denk (sayı / yön / renk / kapak / hareket).
- `vera_lite/dataset_manifest.json` → 13 klip etiketlendi (`expected` + `anomaly_type`).

---

## 5. Metodolojik Düzeltme — Label Leakage (Etiket Sızıntısı)

**Sorun:** Orijinal kod, beklenen etiketi (`expected`) learner prompt'una sızdırıyordu ("ama körü körüne kopyalama" uyarısıyla). Bu, dürüst accuracy ölçümünü geçersiz kılar.

**Çözüm:** `vera_lite/run_iteration.py` **kör mod (blind)** varsayılan yapıldı — learner etiketi **görmüyor**; etiket yalnızca değerlendirme + optimizer için tutuluyor (`--reveal-label` ile ablasyon mümkün). Ayrıca:
- **Otomatik metrikler:** accuracy + precision/recall/F1 + confusion (TP/TN/FP/FN) → `summary.json`.
- **Retry/backoff** (`dashscope_client.py`): geçici 503/throttle hatalarında 5 denemeli exponential backoff.

---

## 6. Kör Değerlendirme Sonuçları (video modu)

| Soru seti | Accuracy | Kaçırılanlar (hep false negative, 0 FP) |
|-----------|:--------:|------------------------------------------|
| **v2** (`runs/blind/20260609_114621`) | **0.77** (10/13) | wrong_orientation_02, motion_stop_02, motion_temporal_01 |
| **v3** (`runs/blind/20260609_121427`) | **0.92** (12/13) | motion_temporal_01 |

**Optimizer katkısı:** v2 sonuçlarına bakıp soruları iyileştirdi. Gerekçesi: *"VLM'ler mantıksal olumsuzlamada (no/without/none) zorlanır; soruları olumlu/doğrudan ifadeye çevir."* → **v3** (`guiding_questions_v3.json`): olumlu ifade + "kenardaki kalem dahil" + "kısa 1-3 sn duraklama dahil" + **Q6 (kalem göreli hareketi)**.

**Sonuç:** VERA döngüsü çalıştı — soruları iyileştirerek **0.77 → 0.92**. False positive boyunca **0** (model normalleri aşırı tetiklemiyor).

---

## 7. Çözülemeyen Klip — `motion_temporal_01`

**Anomali:** Sahnenin ortasında kalemler **görünmeyen bir dış etkiyle kendi başına kayıyor**; bant normal hareket ediyor, sebep kadrajda yok.

**Teşhis:** Bu klip hem v3 learner hem **açık uçlu describe** tarafından kaçırıldı — model "kusursuz normal sahne" diyor. Yani sorun **prompt ifadesi değil**, VLM'in **seyrek kare örnekleme algı limiti** (bant zaten hareket ettiği için kalemlerin göreli kaymasını ayırt edemiyor).

---

## 8. Çözüm Denemeleri

### 8a. Yoğun kare (dense-frame) — `test_dense_frames.py`
**Fikir:** Modelin iç örneklemesi yerine, biz N kareyi zaman sıralı **ayrı görüntü** olarak gönderelim; prompt **kalemler arası göreli boşluğa** odaklansın (bandın L→R hareketinden bağımsız sinyal).

- İlk deneme (tüm klip, 16 kare): temporal → **ANOMALY (0.9)** ✓ ama normal → **ANOMALY (FP!)** ✗ — bandın kalemleri çıkışta kadraj dışına taşıması "göreli hareket" sanıldı.
- Düzeltme: **orta pencere [0.15–0.70]** örnekleme + prompt'a "çıkış normaldir" notu. → temporal **ANOMALY**, normal **NORMAL** ✓✓

### 8b. Video modu + `fps` parametresi
**Fikir (kullanıcı önerisi):** Kareyi elle çıkarmak yerine, videoyu gönderirken DashScope'un `fps` (kare örnekleme oranı) parametresini artır.

- **Bulgu:** `{"video": ..., "fps": 4}` → `video_tokens` 10100 → **20198** (2×). Parametre çalışıyor.
- **Ama:** Model temporal anomaliyi yine **kaçırdı** (NORMAL). Video modunda yüksek fps tek başına yetmiyor — kritik olan kare sayısı değil, **ayrık numaralı kare + göreli-boşluk prompt'u**.

### 8c. "Neden bitmiyor?" — Thinking budget sorunu
**Bulgu:** `qwen3.6-plus` bir **düşünen (thinking) model**. Yoğun görsel girdide kontrolsüz muhakeme yapıp çıktı bütçesini tüketiyor (`reasoning_tokens=15684`, `text_tokens=1`) → **boş JSON** → koşu çöküyordu.

**Çözüm:** `thinking_budget` parametresi (DashScope) → muhakeme N token'da kesilip model **zorla cevaba** geçiyor. `thinking_budget=1500-2000` ile temiz JSON, hızlı bitiş. ("Boş çıktı / takılma" sorunu kalıcı çözüldü.)

### 8d. Başka modeller
- `qwen-vl-max-latest` → **403 AccessDenied** (API key'inde etkin değil).
- `qwen3.7-max` → **400** (video girdisini kabul etmiyor).
- → Erişilebilir tek video modeli **`qwen3.6-plus`**.

---

## 9. Tam Yoğun-Kare Koşusu (16 kare, [0.15–0.70], thinking_budget)

`runs/blind_frames/20260609_160454` — **11/13 = 0.85**, 0 FP.

| Klip | Video v3 (0.92) | Yoğun-kare (0.85) |
|------|:---:|:---:|
| `motion_temporal_01` | ❌ | ✅ **kazanıldı** |
| `wrong_orientation_02` | ✅ | ❌ **kaybedildi** |
| `motion_stop_02` | ✅ | ❌ **kaybedildi** |
| diğer 10 | ✅ | ✅ |

**Takas (trade-off):** Yoğun-kare modu temporal anomaliyi kazandı ama orta-pencere kırpması ([0.15–0.70]) **kenardaki ters kalemi** ve **geç duraklamayı** kırptı. Yani **tek bir temsil 13/13 yapmıyor**; video ve yoğun-kare farklı anomali türlerini yakalıyor.

---

## 10. Çıkarımlar

1. **VERA döngüsü işe yarıyor:** soruları iyileştirerek 0.77 → 0.92, 0 false positive.
2. **VLM temporal zayıflığı doğrulandı:** tek-video çağrısı (yüksek fps dahil) ince intra-object hareketi kaçırıyor — literatürle uyumlu, tez için değerli bulgu.
3. **Sunum biçimi > kare sayısı:** temporal anomaliyi çözen şey, ayrık numaralı kare + göreli-boşluk prompt'u idi.
4. **Thinking model yönetimi şart:** yoğun görsel girdide `thinking_budget` olmadan model boş cevap veriyor.
5. **Temsil takası:** video modu (kenar-yön, duraklama) vs yoğun-kare (intra-object hareket) — farklı güçler.

---

## 11. Sonraki Adımlar

1. **Pencereyi genişlet** (0.05–0.95) + "çıkış normaldir" notu → kenar-yön + geç duraklama geri gelir mi, normaller FP olmadan korunur mu? (tek ucuz koşu)
2. **Ensemble (çok-görüşlü):** video v3 **VE** yoğun-kare; "biri ANOMALY derse ANOMALY". Birleşim muhtemelen 13/13. (2× çağrı, metodolojik olarak şık)
3. **Uyarı:** 13 kliplik pilot sette 13/13'ü tam bu kliplere ayar yaparak elde etmek **overfitting**; asıl değer "hangi yöntem hangi anomali türünü yakalıyor" analizi. Büyük partiye (45 klip) geçmeden önce dikkat.

---

## 12. Eklenen/Değişen Kod (özet)

| Dosya | Değişiklik |
|-------|-----------|
| `compress_videos.py` | 4K → 720×1280 küçültme (yeni) |
| `run_wsl.sh` | WSL grad2_env çalıştırma yardımcısı (yeni) |
| `video_frames.py` | Yoğun kare çıkarma (data URI), paylaşılan (yeni) |
| `test_dense_frames.py` | Yoğun-kare + göreli-boşluk probu (yeni) |
| `prompts.py` | `SYSTEM_CONTEXT` kalem sahnesi; anomali enum; kare-modu notu |
| `guiding_questions.json` (v2) / `guiding_questions_v3.json` | Kalem soruları / optimizer + manuel iyileştirme |
| `dataset_manifest.json` | 13 klip etiketleri |
| `run_iteration.py` | Kör mod (varsayılan), metrikler, `--frames/--fps/--thinking-budget/--max-tokens` |
| `run_learner.py` | Kare modu, `fps`, thinking parametreleri, parse-fallback |
| `dashscope_client.py` | Retry/backoff, generation kwargs geçişi |

**Koşu çıktıları:** `vera_lite/runs/` (video kör: `blind/`, yoğun-kare: `blind_frames/`, describe: `desc_*.txt`, tanılama: `diag_*.json`).
