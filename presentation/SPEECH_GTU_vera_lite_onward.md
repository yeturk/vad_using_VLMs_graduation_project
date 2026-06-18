# Speaker Script — VERA-lite and onward (GTU Final Presentation)

Professional English speaking notes for **slides 9–17** of
`GTU_graduate_project_presentation_template (4).pptx`.
Approx. total: **8–9 minutes**. Delivery tips: speak slowly on the numbers,
pause before each "but / however", and look up at the audience on every transition.

---

## Slide 8 — VERA: the academic foundation (~70 s)

"Everything I am about to show is built on one paper, so let me give it proper credit first. This is **VERA — Verbalized Learning for Video Anomaly Detection**, by Ye, Liu and He, published at **CVPR 2025**.

Its central insight is simple but powerful. If you ask a vision-language model a vague question — *'is there any anomaly here?'* — you only get around **53 to 65 percent AUC**. But if you give it **concrete, learned guiding questions**, performance improves dramatically — and, importantly, **with the model weights completely frozen**.

The method rests on three ideas. **First, a frozen VLM** — there is no fine-tuning and no external reasoning module; the model is adapted *purely through the prompt*. **Second, verbalized learning** — the guiding questions are treated as *learnable parameters*: a learner model answers them, and an optimiser model rewrites them from the learner's mistakes, using only coarse, video-level labels. **Third, coarse-to-fine scoring** — segment-level scores are combined with scene context and then smoothed into frame-level scores; this is what gives VERA state-of-the-art results on the UCF-Crime and XD-Violence benchmarks.

Now, those benchmarks are about violence and crime in surveillance footage. Our setting — an industrial line — is different: we don't need frame-level scores, we need a clear verdict per clip. So we keep VERA's **learner–optimiser question-learning core**, but simplify everything around it to a single clip-level verdict. We call that **VERA-lite** — and that is what I'll walk through next."

---

## Slide 9 — The VERA-lite pipeline (~75 s)

"Having seen what VERA proposes, let me now walk you through our own adaptation, which we call **VERA-lite**. It has four stages.

It begins with a set of **guiding questions** — concrete, yes-or-no questions, one for each family of anomaly we care about: the number of pens, their colour, their orientation, whether each cap is present, whether the belt keeps moving, and whether the pens stay fixed relative to the board. These questions are the heart of the method; in the spirit of VERA, we treat them as *learnable parameters*, not as a fixed prompt.

In the second stage, the **learner** — a frozen vision-language model — watches the clip, answers every question YES or NO, and, crucially, must cite the *visual evidence* for each answer. It then emits a single structured verdict for the clip.

The third stage is the **optimiser**. It reads the cases the learner got wrong and rewrites the questions — sharper, clearer wording — without ever touching the model's weights. We ran this loop once, taking our questions from version two to version three.

And the fourth principle, the one we take most seriously, is **blind evaluation**: the learner never sees the ground-truth label. Early on we found a subtle label leak; once we fixed it, our first *honest* accuracy was 0.77. After a single pass of the optimiser, it rose to **0.92**. So the loop genuinely works — and the next slide shows exactly what it changed."

---

## Slide 10 — Inside the optimiser (~60 s)

"This is, for me, the most interesting finding of the whole project. When the optimiser analysed the learner's mistakes, it effectively *diagnosed the model for us*. Its observation was that vision-language models **struggle with logical negation**.

Concretely: our version-two questions were phrased as negative constraints — for example, *'are there no missing pens and no empty gaps?'*. The model would often skim past the negation and default to 'normal'. The optimiser rewrote these as **positive verification** — *'are there exactly three pens in a single row?'*. Same content, but now the model has to actively *confirm a positive fact* rather than rule out a negative one.

That single linguistic shift removed the blind spots on the count, colour and belt-stop cases — and it is precisely what moved us from 0.77 to 0.92. It's a clean illustration that, with a frozen model, the wording of the prompt is not cosmetic — **the wording is the algorithm**."

---

## Slide 11 — Inside a verdict (~60 s)

"Beyond accuracy, the property we care most about for a factory is **explainability** — and here it becomes concrete. For every clip, the model does not just return a label; it returns this structured **JSON**.

For each guiding question you get a YES or NO *together with the visual evidence* the model used. Then the clip-level verdict, the anomaly type, a score, and a confidence level.

Take the wrong-colour clip shown here. Five of the six questions are YES. But question three is a **NO** — *'the leftmost pen is blue while the others are black'* — and that single NO is what drives the verdict to ANOMALY, type *wrong_color*, with HIGH confidence. An operator on the line can read this in seconds and see *precisely* why the alarm was raised. The decision is **auditable** — not a black box."

---

## Slide 12 — Experiments: three representations (~75 s)

"Now, a verdict is only as good as what the model actually *sees* — so we experimented with **three ways of feeding the clip** to the model.

The first is **continuous video**: we send the whole clip and let the model sample frames internally. This scored **12 out of 13**. Its weakness is that it samples sparsely, so it misses our temporal case, where the pens shift while the belt looks perfectly normal.

The second is **dense frames**: we extract a set of evenly-spaced frames and send them as separate images. This *catches* the temporal case — but at **11 out of 13** it misses some subtler static cues.

The third was an idea we were genuinely hopeful about: the **temporal grid**, where we tile the frames into a single montage image and send it in one pass. This was a **negative result** — it inherited the same blind spots as dense frames, and even added a false positive.

The important conclusion is that this trade-off is **intrinsic**. It is not about the *layout* of the frames; it is the fundamental difference between a *continuous* representation, which captures edges and belt-stops, and a *discrete* one, which captures intra-object motion. We confirmed this across three discrete variants — and **no single representation reaches 13 out of 13**."

---

## Slide 13 — The solution: ensemble (~50 s)

"If no single view is complete, the natural move is to **combine** them — and here we benefited from a principled piece of luck. The key observation is that the blind spots of the two views are **disjoint**. Continuous video misses intra-object motion; dense frames miss some edge and belt-stop cases — but the two never fail on the *same* clip.

So we use a simple **OR-ensemble**: if *either* the video view or the dense-frame view flags a clip, we call it an anomaly. The per-clip matrix on the right makes it visual — every individual row has a red cell, but their union, the bottom row, is **fully green**. The ensemble solves every clip in the dataset, with **zero false alarms**."

---

## Slide 14 — Results (~75 s)

"Let me bring the numbers together. The ensemble reaches **13 out of 13 with zero false positives**. But I want to be careful and honest about the single-view figures, so let me explain the **0.846** and the speed result.

The model we use, *qwen3.6-plus*, is a **reasoning model** — it 'thinks' before it answers. On harder inputs it once spent over **fifteen thousand reasoning tokens** on a single clip, which pushed latency to about **85 seconds** and occasionally caused timeouts.

We capped its reasoning budget at two thousand tokens. That brought latency down to about **50 seconds per clip — a 41 percent reduction — with no loss of accuracy**. And it gave us an unexpected bonus: **determinism**. Before the cap, the verdict on two clips would flip from run to run; afterwards, that dropped to *zero*.

That determinism is also why I quote **0.846** rather than 0.92. The 0.92 was a single *lucky* run. The honest, repeatable number — the same 11 out of 13 in *every* run — is 0.846 for a single view, and **13 out of 13 for the ensemble**."

---

## Slide 15 — Limitations (~70 s)

"No project is complete without an honest account of its limits — and we have six.

**First**, the dataset is small and single-scene — 13 clips on one rig — so the results are indicative, not statistically conclusive.

**Second**, the underlying model is non-deterministic; we had to cap the reasoning budget to stabilise it, and the true single-view accuracy sits below the best single run.

**Third**, no single representation catches everything — the perfect score genuinely requires an ensemble of two API calls.

**Fourth**, it is not real-time: around 50 seconds per clip, and it depends on a paid cloud API and an internet connection, which raises cost and data-privacy questions for a real factory.

**Fifth**, our model comparison was limited — several alternatives were simply inaccessible to us, so the results are tied to one vendor's model version.

**And sixth**, the optimiser sees labels during question-tuning on a very small set, so there is some risk of overfitting the questions to these specific clips.

We believe naming these clearly is part of doing the work honestly."

---

## Slide 16 — Conclusion & future work (~70 s)

"To conclude. We set out to ask whether **explainable video anomaly detection is possible without any labelled training data** — and our answer is yes: it is viable for industry.

To summarise what we showed: a frozen vision-language model, guided by learned questions, detects fine-grained line anomalies with **zero training**; every verdict is a **human-readable, auditable** JSON; a **video-or-dense ensemble reaches 13 out of 13** with no false alarms; and a simple reasoning-budget cap makes the system both **faster and deterministic**.

For future work, the most promising direction is a **cascade**: a cheap classical detector running in real time, with the expensive VLM invoked *only* on the segments it flags — addressing both latency and cost. Beyond that: a **larger, multi-scene dataset** for statistically solid numbers; an **on-premise or open VLM** to remove the cloud and privacy concerns; and **automating the two-view ensemble** into a single call."

---

## Slide 17 — References / closing (~15 s)

"These are the references that informed our work. Thank you very much for your attention — we would be glad to take your questions."

---

### Quick-reference numbers (for Q&A)
- Honest accuracy after fixing label leak: **0.77** → after optimiser **0.92** (single run)
- Repeatable single-view: **0.846 (11/13)** · Ensemble: **13/13, 0 false positives**
- Latency: **85.2 s → 49.9 s (−41%)** with `thinking_budget = 2000`
- Determinism: verdict flips across 3 runs **2 → 0**
- Video: 12/13 (misses temporal) · Dense: 11/13 (catches temporal) · Grid: negative
- Model: **qwen3.6-plus** (reasoning model), via DashScope API
