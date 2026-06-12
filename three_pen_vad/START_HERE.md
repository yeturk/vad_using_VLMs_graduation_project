# Three Pen VAD Pilot Summary

Bu dosya, `three_pen_vad/` klasorundeki yeni calismaya baslamak icin okunmasi
gereken ana ozet dosyasidir. Projenin bu asamasi, IPAD videolarindaki kalite
sorunundan sonra cekilen gercek conveyor videolari uzerinde VLM tabanli video
anomaly detection denemesidir.

## 1. Kisa Amac

Amacimiz, egitim yapmadan Qwen gibi bir Vision-Language Model kullanarak gercek
bir conveyor uzerindeki 3 kalemlik duzende anomali tespiti yapmaktir.

Pilot asama once yalnizca iki sinifla basladi:

- `NORMAL`: Beyaz carrier uzerinde 3 kalem var ve ucunde de siyah kapak benzeri
  parca gorunuyor.
- `MISSING_CAP`: 3 kalemden biri siyah kapak benzeri parcaya sahip degil ve
  bunun yerine acikta sivri yazma ucu gorunuyor.

Daha sonra `WRONG_ORIENTATION` sinifi eklendi:

- `WRONG_ORIENTATION`: Kalemlerden biri digerlerine gore ters, upside-down,
  tilted ya da diagonal duruyor. Kalemde cap-like marker olabilir, fakat marker
  normal ucta degildir veya kalemin acisi digerleriyle uyumlu degildir.

Su an `stuck` gibi diger anomali turleri hala dahil edilmedi. Strateji, yeni
anomali turlerini tek tek ekleyip prompt davranisini gozlemlemek ve her adimda
sonuclari belgelemektir.

## 2. Teorik Arka Plan

Bu calisma `docs/VERA_paper.md` icindeki VERA fikrinden esinlenir. VERA'nin ana
fikri model agirliklarini egitmek degil, prompt icindeki yonlendirici sorulari
iyilestirmektir.

Paper'daki tam akisin ozeti:

```text
video/frame samples + guiding questions
-> learner VLM karar verir
-> optimizer VLM hatalari inceleyip guiding questions'i iyilestirir
-> learned questions inference sirasinda kullanilir
-> segment skorlarindan frame-level anomaly score uretilir
```

Bu repodaki `three_pen_vad` hatti bunun daha kucuk bir "VERA-lite" uyarlamasidir:

```text
processed video
-> description-only Qwen call
-> normal sahne tanimi ve guiding questions tasarimi
-> learner Qwen call
-> NORMAL / MISSING_CAP karari ve aciklama
```

Frame-level scoring, segment retrieval, Gaussian smoothing ve optimizer dongusu
bu pilot asamada uygulanmadi.

## 3. Veri Duzeni

Ham pilot videolar:

```text
data/grad2_pilot/videos/
```

Ilk 5 video:

```text
normal_01.mp4
normal_02.mp4
normal_03.mp4
missing_cap_01_left_pen.mp4
missing_cap_02_middle_pen.mp4
```

Daha sonra eklenen videolar:

```text
wrong_orientation_01_middle_pen.mp4
wrong_orientation_02_right_pen.mp4
combined_01_missing_cap_left_wrong_orientation_right.mp4
```

Processed videolar:

```text
data/grad2_pilot/processed/
```

Preprocessing ayari:

```text
original resolution kept: 2160x3840
source fps: about 30 FPS
target fps: 20 FPS
trim: first 2 seconds and last 2 seconds removed
```

Processed dosyalar yaklasik 13.5-14.2 saniye ve 29-32 MB araligindadir.
Boyutun 100 MB seviyesinden 30 MB seviyesine dusmesinin ana sebebi yalnizca FPS
dusurme degildir. Video yeniden encode edildi ve OpenCV `mp4v` daha dusuk
bitrate ile sikistirdi. Trim ve FPS dusurme de buna ek katkidir.

Buyuk video dosyalari `.gitignore` ile disarida tutulur.

## 4. Kod Duzeni

Yeni hat:

```text
three_pen_vad/
  README.md
  START_HERE.md
  dataset_manifest.json
  prompts.py
  preprocess_video.py
  describe_video.py
  run_learner.py
  runs/
```

Dosyalarin rolleri:

- `preprocess_video.py`: 4K cozunurlugu koruyarak 20 FPS ve 2s trim ile
  processed video uretir.
- `describe_video.py`: Qwen'e siniflandirma yaptirmadan sadece videoda ne
  gordugunu anlattirir.
- `prompts.py`: Description-only prompt, learner prompt ve guiding questions'i
  tutar. Promptlar Ingilizcedir.
- `run_learner.py`: Qwen ile learner prompt'unu calistirir. `--prompt-version`
  ile V1 veya V2 secilebilir. Calisma suresi ve token kullanimini JSON'a yazar.
- `dataset_manifest.json`: 5 pilot videonun id, path, expected label ve
  aciklamalarini tutar.

Eski `vera_lite/` klasoru push-button/IPAD deneyi olarak kalir. Yeni gercek
conveyor ve 3 kalem hatti `three_pen_vad/` altindadir.

## 5. Runs Klasoru Duzeni

`three_pen_vad/runs/` klasoru uc asamaya ayrildi:

```text
three_pen_vad/runs/
  01_description_only/
  02_learner_with_expected_label_leakage/
  03_learner_no_expected_label/
  04_learner_wrong_orientation/
  05_learner_combined/
```

### 01_description_only

Qwen'e "normal/anomaly karari verme, sadece ne gordugunu anlat" denilen
ciktilardir.

Bu asamada Qwen su ortak gozlemleri yakaladi:

- Gercek lineer hareket/conveyor benzeri duzenek var.
- Beyaz dikdortgen carrier plate var.
- Metal raylar ve central lead screw gorunuyor.
- Carrier soldan saga hareket ediyor.
- Normal videolarda 3 kalem var.
- Kalemler paralel, dikey gorunumlu ve carrier'a gore sabit.
- Normal videolarda uc kalemde de siyah cap-like parca gorunuyor.
- Missing cap videolarinda eksik kapagi dogru yerde fark ediyor.

Onemli prompt dersi:

`normal_02` description-only ciktisinda model giris/cikis sirasindaki kismi
gorunurlugu "stop-motion or digital editing" gibi yorumladi. Bu nedenle learner
prompt'una su kural eklendi:

```text
Analyze fully visible middle frames for pen count and cap visibility.
Do not treat partial visibility during entry or exit as an anomaly.
Sequential appearance or disappearance at the frame borders is expected.
```

### 02_learner_with_expected_label_leakage

Ilk learner denemesinde `--expected` degeri prompt icine yaziliyordu:

```text
Expected label: NORMAL
```

Bu metodolojik olarak hataliydi. Model gercek etiketi prompt'ta gordugu icin
dogru cevaplar label leakage riski tasiyordu.

Bu run'lar arsiv olarak tutuldu ama temiz deney sonucu olarak kullanilmamali.

### 03_learner_no_expected_label

Bu temiz deneydir. `expected` degeri prompt'a verilmez. Sadece JSON sonucunda
metadata olarak saklanir.

No-leak learner sonuclari:

| Video | Expected | Predicted | Missing cap pen | Score | Confidence | Time | Total tokens |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `normal_01` | `NORMAL` | `NORMAL` | `none` | 0.0 | HIGH | 47.00s | 9339 |
| `normal_02` | `NORMAL` | `NORMAL` | `none` | 0.0 | HIGH | 59.16s | 9981 |
| `normal_03` | `NORMAL` | `NORMAL` | `none` | 0.0 | HIGH | 43.24s | 10008 |
| `missing_cap_01_left_pen` | `MISSING_CAP` | `MISSING_CAP` | `left_first` | 1.0 | HIGH | 88.63s | 11899 |
| `missing_cap_02_middle_pen` | `MISSING_CAP` | `MISSING_CAP` | `middle_second` | 1.0 | HIGH | 39.54s | 9892 |

Sonuc:

```text
correct predictions: 5 / 5
normal samples: 3 / 3 correct
missing cap samples: 2 / 2 correct
```

Bu final performans iddiasi degildir. Sadece ilk pilot sanity check olarak
yorumlanmalidir.

### 04_learner_wrong_orientation

`WRONG_ORIENTATION` sinifi prompt V1'e eklendi. Bu asamada once sadece
`wrong_orientation_01_middle_pen` icin description-only call yapildi. Model orta
kalemin cap-like parcasinin ustte oldugunu ve sol/sag kalemlerden farkli
yonlendigini fark etti.

Sonra iki wrong-orientation videosu learner ile test edildi:

| Video | Expected | Predicted | Wrong orientation pen | Type | Confidence | Time | Total tokens |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| `wrong_orientation_01_middle_pen` | `WRONG_ORIENTATION` | `WRONG_ORIENTATION` | `middle_second` | `reversed` | HIGH | 61.00s | 10602 |
| `wrong_orientation_02_right_pen` | `WRONG_ORIENTATION` | `WRONG_ORIENTATION` | `right_third` | `tilted` | HIGH | 129.40s | 14143 |

Onemli gozlem:

```text
wrong_orientation_01 reasoning tokens: 1515
wrong_orientation_02 reasoning tokens: 5057
```

Raw JSON uzunluklari benzer olmasina ragmen ikinci videoda reasoning token
sayisi cok artti. Bu, sureyi sadece soru sayisinin belirlemedigini gosterir.
Model bazen gorsel karar icin daha uzun ic akil yurutme yapabilir.

### 05_learner_combined

`combined_01_missing_cap_left_wrong_orientation_right` videosu describe
ettirilmeden dogrudan V1 learner ile test edildi. Videoda sol kalemde
`MISSING_CAP`, sag kalemde ise `WRONG_ORIENTATION` olmasi bekleniyordu.

V1 sonucu:

```text
Predicted: MISSING_CAP
missing_cap_pen: left_first
wrong_orientation_pen: none
wrong_orientation_type: none
confidence: HIGH
score: 0.95
elapsed: 83.08s
input/output/total/video tokens: 8588 / 2562 / 11150 / 7724
```

Degerlendirme:

- V1, sol kalemdeki missing cap anomalysini dogru yakaladi.
- V1, sag kalemdeki wrong-orientation anomalysini kacirdi.
- Bu sonuc, tek dominant anomaly kararinin combined anomaly videolari icin
  yetersiz olabilecegini gosterdi.
- Bu nedenle V2 prompt tasarimi per-pen attribute extraction yaklasimina
  gecirildi.

### 07_learner_v1_new_anomalies

Color anomaly ve temporal stuck videolari once eski V1 prompt ile dogrudan test
edildi. Bu testte amac, V1'in yeni anomaly turlerini hic prompt guncellemeden
yakalayip yakalamadigini gormekti.

Sonuclar:

| Video | Expected anomaly | Predicted | Confidence | Time | Total tokens |
| --- | --- | --- | --- | ---: | ---: |
| `color_anomaly_01_left_pen_blue` | color anomaly | `NORMAL` | HIGH | 160.69s | 15076 |
| `temporal_stuck_01_carrier_stuck` | temporal stuck | `NORMAL` | HIGH | 31.55s | 10279 |

Color anomaly degerlendirmesi:

- Model farki gordu: "one blue, two black" dedi.
- Q2'de sol kalemin clear/white cap-like parca tasidigini belirtti.
- Ancak promptta `COLOR_ANOMALY` sinifi olmadigi icin bunu normal kabul etti.

Temporal stuck degerlendirmesi:

- Model durmayi gordu: carrier'in soldan girip pens fully in view durumunda
  durdugunu yazdi.
- Ancak promptta "stopping/stalling is anomaly" kurali olmadigi icin bunu normal
  kabul etti.
- Daha sonra temporal stuck icin description-only call denendi, fakat 180
  saniyeyi astigi icin sonlandirildi ve cikti dosyasi uretmedi.
- Prompt guncellemesi bu nedenle learner evidence'ina dayandirildi: modelin
  "stopping with the pens fully in view" ifadesi temporal anomaly kuralina
  baglandi.

Kok neden:

```text
Bu iki hata algi hatasi degil, karar kurali eksikligidir.
```

Bu nedenle V1 prompt 8 soruyu asmadan genisletildi:

- `COLOR_ANOMALY` eklendi.
- `TEMPORAL_STUCK` eklendi.
- Redundant cap/orientation sorulari birlestirilerek soru sayisi 8'de tutuldu.
- Temporal stuck icin early/middle/late frame karsilastirmasi eklendi:
  carrier birkac saniye fully visible halde sabit kalirsa `TEMPORAL_STUCK`
  sayilacak.

Guncellenmis V1 ile tekrar test:

| Video | Expected anomaly | Predicted | Key field | Confidence | Time | Total tokens |
| --- | --- | --- | --- | --- | ---: | ---: |
| `color_anomaly_01_left_pen_blue` | `COLOR_ANOMALY` | `COLOR_ANOMALY` | `color_anomaly_pen=left_first` | HIGH | 75.92s | 11645 |
| `temporal_stuck_01_carrier_stuck` | `TEMPORAL_STUCK` | `TEMPORAL_STUCK` | `temporal_anomaly=carrier_stuck` | HIGH | 62.04s | 11051 |

Temporal stuck retry evidence:

```text
Q7: NO -- The carrier moves from left to right initially but stops completely
around 00:05 and remains stationary until the end of the video.
```

Sonuc:

```text
Color anomaly ve temporal stuck, V1 prompt kapsam genisletmesi sonrasi dogru
yakalanmistir. Temporal stuck icin kritik iyilestirme, early/middle/late frame
pozisyon karsilastirmasini acikca istemek oldu.
```

## 6. Learner Prompt V1 Ozeti

Learner prompt V1'in ana fikri:

- Sahneyi gercek conveyor ve beyaz carrier olarak tanimla.
- `NORMAL`, `MISSING_CAP`, `WRONG_ORIENTATION`, `COLOR_ANOMALY` ve
  `TEMPORAL_STUCK` ayrimini yap.
- Orta frame'lerde tam gorunen kalemlere bak.
- Gidis/giris kismi gorunurlugu anomali sayma.
- Dark/black cap-like parcayi normal gorsel marker olarak kullan.
- Missing cap icin acikta sivri yazma ucunu ara.
- Wrong orientation icin ters, upside-down, tilted veya diagonal kalemi ara.
- Color anomaly icin digerlerinden belirgin farkli renk veya material appearance
  ara.
- Temporal stuck icin carrier'in inspection view icinde stop/stall/get stuck
  olmasini ara.

Guiding questions:

```text
1. In fully visible middle frames, are exactly three pens visible on the white carrier?
2. Do all pens match the expected normal appearance: dark body and dark cap-like part at the lower end?
3. Does any pen show a clearly different color or material appearance, such as blue, transparent, or white parts?
4. Does any pen lack the normal cap-like marker or show an exposed pointed writing tip?
5. Are all pens similarly oriented, with no reversed, upside-down, tilted, or diagonal pen?
6. Are the cap-like markers on the same end of all visible pens?
7. Compare early, middle, and late frames: does the carrier keep changing position without stopping, stalling, or getting stuck?
8. Do the pens remain stable relative to the carrier without sliding, falling, or unexpected disappearance?
```

Output JSON semasi:

```json
{
  "question_answers": [
    {
      "question_id": 1,
      "answer": "YES / NO / UNCLEAR",
      "evidence": "short visual evidence"
    }
  ],
  "verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / COLOR_ANOMALY / TEMPORAL_STUCK / UNCLEAR",
  "missing_cap_pen": "none / left_first / middle_second / right_third / unclear",
  "wrong_orientation_pen": "none / left_first / middle_second / right_third / unclear",
  "wrong_orientation_type": "none / reversed / tilted / unclear",
  "color_anomaly_pen": "none / left_first / middle_second / right_third / unclear",
  "temporal_anomaly": "none / carrier_stuck / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the verdict"
}
```

V1 limitation:

```text
V1 works well for one dominant anomaly, but can miss a second anomaly in a
combined video because the output schema is centered around one verdict and a
small set of global yes/no questions.
```

## 7. Learner Prompt V2 Tasarimi

V2'nin amaci soru sayisini artirmak degil, sorularin kalitesini artirmaktir.
V1'de 8 global guiding question varken V2'de 5 daha yuksek seviyeli soru vardir.

V2 yaklasimi:

```text
global yes/no questions
-> per-pen attribute extraction
-> anomaly list
-> primary verdict
```

V2 guiding questions:

```text
1. In fully visible middle frames, are exactly three pens visible and stable on the white carrier?
2. For each pen, report cap marker status, cap end, exposed writing tip, and orientation.
3. Which pens, if any, have missing-cap evidence?
4. Which pens, if any, have wrong-orientation evidence?
5. Does the carrier motion look normal enough that the object inspection is reliable?
```

V2 output schema:

```json
{
  "question_answers": [
    {
      "question_id": 1,
      "answer": "YES / NO / UNCLEAR",
      "evidence": "short visual evidence"
    }
  ],
  "pen_observations": {
    "left_first": {
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    },
    "middle_second": {
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    },
    "right_third": {
      "cap_marker": "present / absent / unclear",
      "cap_end": "top / bottom / none / unclear",
      "writing_tip_exposed": "yes / no / unclear",
      "orientation": "normal / reversed / tilted / unclear",
      "evidence": "short visual evidence"
    }
  },
  "detected_anomalies": [
    {
      "type": "MISSING_CAP / WRONG_ORIENTATION",
      "pen": "left_first / middle_second / right_third",
      "evidence": "short visual evidence"
    }
  ],
  "primary_verdict": "NORMAL / MISSING_CAP / WRONG_ORIENTATION / MULTIPLE_ANOMALIES / UNCLEAR",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the detected anomalies"
}
```

V2'nin beklenen artilari:

- Combined anomaly videolarinda birden fazla anomaly raporlayabilir.
- Her kalem icin ayri attribute cikardigi icin modelin tek anomalye
  kilitlenme riski azalir.
- Soru sayisi 8'den 5'e dustugu icin prompt daha kompakt hale gelir.
- `detected_anomalies` listesi ileride evaluation icin daha kullanislidir.

V2'nin riskleri:

- JSON semasi daha ayrintili oldugu icin text output token sayisi artabilir.
- Model bazi durumlarda attribute tablosunu tutarli dolduramayabilir.
- `primary_verdict` ile `detected_anomalies` arasinda celiski cikabilir.
- Sureyi azaltacagi garanti degildir; video zorlugu ve modelin reasoning token
  davranisi hala belirleyici olabilir.

V2'yi calistirma mantigi:

```bash
python -m three_pen_vad.run_learner \
  --prompt-version v2 \
  --video data/grad2_pilot/processed/combined_01_missing_cap_left_wrong_orientation_right_4k_20fps_trim2s.mp4 \
  --expected MISSING_CAP \
  --out three_pen_vad/runs/06_learner_v2_per_pen/manual_v2_combined_01_missing_cap_left_wrong_orientation_right_learner.json
```

Beklenen V2 davranisi:

```text
primary_verdict: MULTIPLE_ANOMALIES
detected_anomalies includes:
- MISSING_CAP on left_first
- WRONG_ORIENTATION on right_third
```

## 8. Token ve Sure Gozlemleri

No-leak learner run'larinda:

```text
average elapsed time: about 55.5 seconds
average total tokens: about 10.2k
video tokens: about 7.7k-8.3k
```

`missing_cap_01_left_pen` en pahali run oldu:

```text
elapsed: 88.63s
total tokens: 11899
output tokens: 2931
```

V1 wrong-orientation run'lari:

```text
wrong_orientation_01: 61.00s, total tokens 10602
wrong_orientation_02: 129.40s, total tokens 14143
```

Combined V1 run:

```text
combined_01: 83.08s, total tokens 11150
```

Combined V2 per-pen run:

```text
combined_01: 189.07s, total tokens 17450
input/output/video tokens: 8769 / 8681 / 7724
reasoning/text output tokens: 8151 / 530
primary_verdict: MULTIPLE_ANOMALIES
detected anomalies: MISSING_CAP left_first, WRONG_ORIENTATION right_third
```

V2 combined videoda evidence kalitesini belirgin sekilde iyilestirdi ve iki
anomaliyi de yakaladi. Ancak sure ve reasoning token maliyeti cok artti. Gorunen
JSON uzun degildi; asil maliyet `reasoning_tokens` alanindan geldi. Bu nedenle
V2 fikri dogru olsa da pratik akista simdilik V1 ile devam edilecek.

Muhtemel neden: Soldaki eksik kapak icin model daha uzun reasoning uretti.

## 8. Yapilabilir Prompt ve Maliyet Iyilestirmeleri

Bu fikirler su anda uygulanmadi. Sonraki prompt/optimizer asamalarinda
denenebilir:

- `v2_lite`: V2'nin per-pen attribute extraction fikrini koru ama
  `question_answers` alanini cikart.
- Evidence alanlarini kisalt: Her evidence cumlesi en fazla 8-12 kelime olsun.
- Prompt'a "Use direct visual inspection. Do not provide step-by-step reasoning."
  benzeri bir kural ekle.
- DashScope/Qwen API tarafinda reasoning/thinking davranisini kapatmaya veya
  sinirlamaya yarayan resmi bir parametre var mi arastir.
- V1'i anomaly-detection odakli kullan: Bu asamada amac en az bir anomaly
  sinyali yakalamak; anomaly tipinin kusursuz ayrimi ikinci oncelik.
- Optimizer'i yeni anomaly turleri V1 sorulariyla yakalanmazsa devreye al:
  ozellikle color anomaly ve temporal stuck icin hangi soru eksik kaldiysa
  optimizer'dan yeni ama kompakt guiding question onerileri istenebilir.
- Combined anomaly evaluation icin ayri bir metric dusun: Tek verdict yerine
  "any anomaly detected" ve "all anomaly types detected" sonuclari ayrilabilir.

## 9. Optimizer Gerekli Miydi?

Bu asamada optimizer'a gerek kalmadi.

Neden:

- Description-only asamada model sahneyi yeterince iyi anladi.
- Learner prompt insan tarafindan, description-only ciktisindan elde edilen
  derslerle tasarlandi.
- No-leak learner 5 videonun tamaminda dogru karar verdi.
- Evidence kalitesi de dogru gorsel gerekceye dayaniyordu.

Optimizer daha sonra gerekli olabilir:

- Yeni anomali turleri eklendiginde.
- Model yanlis karar verdiginde.
- Guiding questions bir anomaly turunu fazla dar veya fazla genis yakaladiginda.
- `wrong_orientation` ve `stuck` gibi daha zor vakalarda prompt belirsizlesirse.

## 10. Calistirma Komutlari

Preprocess:

```bash
conda activate grad2_env
python -m three_pen_vad.preprocess_video
```

Description-only:

```bash
python -m three_pen_vad.describe_video \
  --video data/grad2_pilot/processed/normal_01_4k_20fps_trim2s.mp4 \
  --out three_pen_vad/runs/01_description_only/manual_normal_01_description.txt
```

No-leak learner:

```bash
python -m three_pen_vad.run_learner \
  --video data/grad2_pilot/processed/normal_01_4k_20fps_trim2s.mp4 \
  --expected NORMAL \
  --out three_pen_vad/runs/03_learner_no_expected_label/manual_noleak_normal_01_learner.json
```

Not: `--expected` artik prompt'a girmez. Sadece JSON sonucunda evaluation
metadata olarak saklanir.

## 11. Testler

Mevcut unit testler:

```bash
conda activate grad2_env
python -m unittest tests/test_three_pen_vad.py
```

Bu testler sunlari kontrol eder:

- Processed video dosya adi formatini.
- Trim window hesaplamasini.
- Description prompt'un observation-only olmasini.
- Manifest'in 5 pilot item icermesini.
- Learner prompt V1'in `NORMAL / MISSING_CAP / WRONG_ORIENTATION` kapsaminda
  olmasini.
- Learner prompt V2'nin per-pen attribute extraction semasini istemesini.
- `expected` label'in learner prompt'a sizmadigini.
- Run summary'nin elapsed time ve token bilgilerini dogru cikarmasini.

## 12. Siradaki Is

Bir sonraki asamada yeni anomali turleri eklenecek.

Onerilen sira:

1. V1 prompt ile yeni anomaly videolarini test et.
2. Color anomaly icin once dogrudan learner calistir; description-only'i sadece
   gerekirse kullan.
3. Temporal stuck anomaly icin dogrudan learner calistir; V1 hareket sorusu
   yeterli sinyal veriyor mu bak.
4. Eger V1 yeni anomaly videolarinda tamamen `NORMAL` derse optimizer veya
   kompakt V1 soru guncellemesi tartis.
5. V2-lite ve reasoning azaltma fikirlerini daha sonra maliyet optimizasyonu
   olarak ele al.

`stuck` anomalisi temporal oldugu icin color anomaly'ye gore daha zor olabilir.
Missing cap, wrong orientation ve color anomaly daha cok statik/gorsel farklara
dayanirken stuck hareketin durmasini veya beklenen conveyor akisinin
bozulmasini anlamayi gerektirir.
