# Three Pen VAD Pilot Summary

Bu dosya, `three_pen_vad/` klasorundeki yeni calismaya baslamak icin okunmasi
gereken ana ozet dosyasidir. Projenin bu asamasi, IPAD videolarindaki kalite
sorunundan sonra cekilen gercek conveyor videolari uzerinde VLM tabanli video
anomaly detection denemesidir.

## 1. Kisa Amac

Amacimiz, egitim yapmadan Qwen gibi bir Vision-Language Model kullanarak gercek
bir conveyor uzerindeki 3 kalemlik duzende anomali tespiti yapmaktir.

Bu pilot asamada yalnizca iki sinif ele alindi:

- `NORMAL`: Beyaz carrier uzerinde 3 kalem var ve ucunde de siyah kapak benzeri
  parca gorunuyor.
- `MISSING_CAP`: 3 kalemden biri siyah kapak benzeri parcaya sahip degil ve
  bunun yerine acikta sivri yazma ucu gorunuyor.

Diger anomali turleri, ornegin `wrong_orientation` ve `stuck`, bu asamada
bilerek dahil edilmedi. Adim adim ilerlemek icin once `NORMAL` vs
`MISSING_CAP` ayrimi test edildi.

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
- `run_learner.py`: Qwen ile `NORMAL / MISSING_CAP / UNCLEAR` karari alir.
  Calisma suresi ve token kullanimini JSON'a yazar.
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

## 6. Learner Prompt Ozeti

Learner prompt'un ana fikri:

- Sahneyi gercek conveyor ve beyaz carrier olarak tanimla.
- Sadece `NORMAL` ve `MISSING_CAP` ayrimini yap.
- Diger anomali turlerini henuz raporlama.
- Orta frame'lerde tam gorunen kalemlere bak.
- Gidis/giris kismi gorunurlugu anomali sayma.
- Siyah cap-like parcayi normal gorsel marker olarak kullan.
- Missing cap icin acikta sivri yazma ucunu ara.

Guiding questions:

```text
1. In fully visible middle frames, are exactly three pens visible on the white carrier?
2. Does each pen show a visible black cap-like part at its lower end?
3. Does any pen show an exposed pointed writing tip where the black cap-like part should be?
4. Are the three pens parallel and similarly oriented on the carrier?
5. Does the carrier move smoothly from LEFT to RIGHT while the pens remain stable relative to it?
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
  "verdict": "NORMAL / MISSING_CAP / UNCLEAR",
  "missing_cap_pen": "none / left_first / middle_second / right_third / unclear",
  "anomaly_score": 0.0,
  "confidence": "LOW / MEDIUM / HIGH",
  "reason": "one or two sentences explaining the verdict"
}
```

## 7. Token ve Sure Gozlemleri

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

Muhtemel neden: Soldaki eksik kapak icin model daha uzun reasoning uretti.

## 8. Optimizer Gerekli Miydi?

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

## 9. Calistirma Komutlari

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

## 10. Testler

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
- Learner prompt'un sadece `NORMAL / MISSING_CAP` kapsaminda olmasini.
- `expected` label'in learner prompt'a sizmadigini.
- Run summary'nin elapsed time ve token bilgilerini dogru cikarmasini.

## 11. Siradaki Is

Bir sonraki asamada yeni anomali turleri eklenecek.

Onerilen sira:

1. `wrong_orientation` icin once 1-2 video sec.
2. Description-only ciktisini al.
3. Modelin ters kalemi nasil gordugunu incele.
4. Learner prompt kapsam tasarimini tartis.
5. `NORMAL / MISSING_CAP / WRONG_ORIENTATION` gibi yeni kapsama gecmeden once
   prompt'u birlikte onayla.
6. Ancak onaydan sonra kodu guncelle.

`stuck` anomalisi daha temporal oldugu icin daha sonra ele alinmali. Missing cap
gorsel/statik bir anomaliydi; stuck ise hareketin durmasini anlamayi gerektirir.

