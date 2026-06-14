# Rapor Taslağı — Persona Yaklaşımı & VERA-lite (akış + tablolar)

> Çalışma dili Türkçe taslak; tez gövdesi İngilizce ise sonradan çevrilecek.
> Sayılar 2026-06-13 itibarıyla gerçek koşu çıktılarından.

---

## Bölüm X — Yaklaşım B: Çok-Persona Çoğunluk Oylaması

### X.1 Motivasyon
Tek bir prompt ve tek bir çıkarım, VLM'in rastgele varyansına ve "her şey normal" yanlılığına açık.
Fikir: aynı sahneyi **farklı bakış açılarından** sorgulayan birden çok "denetçi personası" çalıştırıp
**çoğunluk oyuyla** karar vermek (self-consistency / çok-ajan ensemble). Yerel açık-ağırlık model
**Qwen3-VL-8B-Instruct** (transformers/torch) üzerinde çalışır — API değil, veri hattı dışarı çıkmaz.

### X.2 Yöntem
- **20 persona**: kıdemli/yeni operatör, vardiya amiri, makine/kalite/CV/robotik mühendisi, bakım,
  denetçi, lean uzmanı, Toyota uzmanı, muhafazakâr/agresif denetçi, eğitmen, hibrit… Hepsi **aynı
  anomali kuralları** (checklist), farklı **ses / eşik / format**.
- **Oylama alt kümesi**: 5 persona + 1 "konsensüs" prompt = **6 oy** → `majority_vote` → nihai karar +
  güven (oy yüzdesi: ≥%80 HIGH, ≥%60 MEDIUM, altı LOW).
- **articulate-first kuralı**: model önce kareleri (giriş/orta/çıkış) betimler, *sonra* karar verir.

### X.3 Üç Persona Tipi (ablasyon)
Aynı oylama iskeleti, farklı **checklist genişliği**:

| Tip | Dosya | Her oy verenin yaptığı | Mantık |
|-----|-------|------------------------|--------|
| **specialized** | `personas_specialized.py` | Tek odak (ör. sadece yön / sadece hareket) | Uzmanlar komitesi (iş bölümü) |
| **complete** | `personas_complete.py` | Tüm A/B/C checklist (yön + hareket + bant konumu) | Artıklık (her oy tam denetim) |
| **orientation** | `personas_orientation.py` | Sadece yön sorusu (kırmızı kapak sağda mı/solda mı) | Minimal / birincil anomaliye odak |

### X.4 Temel Bulgu — "Articulate-first"
Kompakt YES/NO çıktı formatı **false negative** üretiyordu: model kareleri gerçekten incelemeden
"normal"e kayıyordu. Çıktıyı **önce kare-kare betimleme → sonra karar** olacak şekilde zorlayınca bu
yanlılık bastırıldı. (Bu bulgu VERA'daki "soruyu alt-sorulara ayır" ilkesiyle aynı kökten — *modeli
gözlemi dile dökmeye zorlamak doğruluğu artırıyor*.)

### X.5 Durum / Sınırlılık
Persona hattı **R01 push-button** sahnesine ve yerel Qwen3-VL-8B'ye özgü kuruldu. Etiketli, çok-klipli
**nicel değerlendirme henüz yok** (eldeki tek rapor bir örnek videoda demo koşusu). Adil sayısal kıyas
için personaları etiketli bir sette (R01 veya kalem) koşmak → **gelecek çalışma**.

---

## Bölüm Y — Yaklaşım C: VERA-lite (tam akış)

### Y.1 Kurulum
- Model: `qwen3.6-plus` (DashScope, intl endpoint), ortam WSL + `grad2_env`.
- Pipeline: guiding questions + **learner** (NORMAL/ANOMALY + skor) + **optimizer** (soruları iyileştirir).
- Veri: 13 klip kalem konveyör (`data/pens_small/`, 4K→720×1280), 3 normal + 10 anomali (kapak, yön,
  renk, sayı, hareket-durması, kalem-oynaması/temporal, birleşik).

### Y.2 Metodolojik Düzeltme — Kör Değerlendirme
Orijinal kod beklenen etiketi learner prompt'una **sızdırıyordu** (label leakage). Düzeltildi: **kör mod
varsayılan** (learner etiketi görmez; etiket yalnızca değerlendirmede). Tüm aşağıdaki sayılar kördür.

### Y.3 Akış 1 — Soruları İyileştirme (VERA döngüsü)

| Soru seti | Accuracy (kör, tek koşu) | FP | Kaçırdığı |
|-----------|:---:|:---:|---|
| v2 | 0.77 (10/13) | 0 | orient2, stop2, temporal |
| **v3** (optimizer + manuel) | **0.92 (12/13)** | 0 | temporal |

Optimizer'ın gerekçesi: olumsuz ifadeler (no/without) VLM'i zorluyor → **olumlu/doğrudan** sorular.
v3 ayrıca "kenar kalemi dahil", "kısa duraklama dahil" ve Q6 (kalem göreli hareketi) ekledi.

### Y.4 Çözülemeyen Tek Klip — Temporal Anomali
`motion_temporal_01` (kalemler kendi başına kayıyor, bant normal). Hem v3 learner hem **açık-uçlu
describe** kaçırdı → **prompt değil, VLM seyrek-kare algı limiti**.

### Y.5 Akış 2 — Girdi Temsili Denemeleri

| Temsil | Accuracy | FP | Not |
|--------|:---:|:---:|-----|
| Video (v3, varsayılan fps) | 0.92 (12/13) | 0 | temporal'i kaçırır |
| Yoğun-kare [0.15–0.70], 16 kare | 0.85 (11/13) | 0 | temporal'i **yakalar**, orient2+stop2'yi kaçırır |
| Yoğun-kare geniş [0.05–0.95] | 0.62 (8/13) | 1 | çıkış artefaktı FP doğurdu |
| Temporal grid 3×3 (4-klip probu) | — | var | temporal✓, orient2✗, stop2✗, normal **FP** |
| Video + fps=8 + v3 | 0.85 (11/13) | 0 | yüksek fps orient2'yi bozdu |
| **Ensemble (video v3 ⋁ dense16)** | **1.00 (13/13)** | **0** | iki görüşün birleşimi |

**Bulgu:** Hiçbir tek temsil 13/13 yapmıyor; **continuous (video) ↔ discrete (kare/grid) takası içsel**.
Ensemble, iki yöntemin **ayrık kör noktalarını** birleştirir; ikisi de 0 FP olduğu için birleşim FP üretmez.

`video_tokens` (örnekleme kontrolü kanıtı): default ~10.100 → fps4 ~20.198 → fps8 ~40.394.

### Y.6 Akış 3 — Tekrarlanabilirlik (varyans)
Tek koşu yanıltıcı; her ayar tekrar koşuldu (kör):

| Ayar | n | Ortalama Accuracy | std | Toplam FP | Her zaman kaçan | Gözlem |
|------|:-:|:---:|:---:|:---:|---|--------|
| baseline (sınırsız muhakeme) | 3 | **0.846 (11/13)** | 0.000 | 1 | temporal | diğer kaçış orient2↔normal_02 |
| tb2000 | 3 | **0.846 (11/13)** | 0.000 | 0 | temporal | orient2 |

**Çıkarım:** tek-koşu "0.92" şanslı bir çekilişti; gerçek değer **0.846±0.000** (n=3). `temporal` 6/6 kaçtı
(deterministik limit). Borderline klipler (orient2, normal_02) zıplıyor. baseline 3 koşuda 1 FP üretti.

### Y.7 Akış 4 — Hız (thinking_budget)
qwen3.6-plus düşünen model; muhakeme çıktı bütçesini tüketiyor (bir klip 13.394 token "düşündü").
`thinking_budget` muhakemeyi sınırlar:

| Ayar | sn/klip (n=3 ort.) | Accuracy (n=3) | Not |
|------|:---:|:---:|-----|
| baseline (sınırsız) | 85.2 | 0.846±0.000 | yavaş uç klipler (bir klip 13.394 token) |
| **tb2000** | **49.9** | 0.846±0.000 | **~%41 hızlı, accuracy aynı, 0 FP** |
| tb1000 | 39.5* | ~0.85* | ~%53 hızlı, borderline daha oynak (*tek koşu) |

**Bulgu:** `thinking_budget=2000` accuracy'yi bozmadan ~%41 hızlandırır (tekrarlı ölçümle kanıtlı). Not: fps **artırmak**
hızı düşürür + accuracy'yi bozar (fps=8 → orient2 kaybı), yani hız için yanlış yön.

### Y.8 Özet Bulgular
1. VERA döngüsü soruları iyileştirir (v2→v3).
2. VLM **temporal/intra-object hareket** zayıflığı tutarlı ve doğrulandı (5/5).
3. **Sunum biçimi > kare sayısı / ham fps.**
4. **Ensemble** ayrık kör noktaları birleştirir (0 FP korunur).
5. Tek-koşu accuracy gürültülü → **tekrarlı ölçüm şart** (~0.85±0.00).
6. **thinking_budget** = güvenli hız kaldıracı (~%30, accuracy sabit).

---

## Ortak Temalar (A↔B↔C — Tartışma bölümü)
- **Modelin "NORMAL" yanlılığı**: persona (articulate-first) ve VERA (soru ayrıştırma) aynı sorunu çözdü.
- **Oylama/ensemble robustluğu**: persona çoğunluk oyu ↔ VERA video+dense ensemble.
- **Sunum biçimi belirleyici**: describe-önce, ayrık kare, göreli sorular.

## Kıyaslanabilirlik Uyarısı (rapora yaz)
A/B/C farklı model + sahne kullandı → birebir sayısal kıyas adil değil; "yöntem yolculuğu" olarak sun,
"üçünü aynı sette koşmak" = gelecek çalışma.
