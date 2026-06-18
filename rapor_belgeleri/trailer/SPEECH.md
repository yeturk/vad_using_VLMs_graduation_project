# Presentation Script (~3.5 minutes)

Simple B1 English. One block per slide. Times are rough.

---

## Slide 1 — Title (~20s)
Hello everyone. Our project is about **video anomaly detection in production and assembly lines**.
The goal is simple: a camera watches a conveyor belt, and a computer says if something looks wrong.
The special part is that we do this **without training any model**. We only use a Vision-Language Model and good prompts.
I am Alper, this is Yunus, and our advisor is Assistant Professor Habil Kalkan.

## Slide 2 — Problem & Motivation (~30s)
First, why is this hard? Classic computer vision models have three big problems in a factory.
One: they need **thousands of defect images**, but in a good factory defects are very rare.
Two: every time the product changes, you must **train the model again**.
Three: they are a **black box** — they say "anomaly", but they cannot explain *why*.
Our idea is different. We do not train a detector. We give a frozen Vision-Language Model a description of the scene and the rules, and we ask it to decide. No training, and a full explanation.

## Slide 3 — Dataset Journey (~35s)
We worked with data in two steps.
At the start, we used a public benchmark called **IPAD**. It shows a small button on a conveyor belt.
But the videos were only 256 by 256 pixels. That is too low quality to see small details like color or angle.
So we decided to **build our own dataset**. We recorded a conveyor belt with three pens, in 4K, with a fixed camera.
We created **seven different anomaly types** — for example wrong color, wrong direction, a missing pen, or the belt stopping.

## Slide 4 — Approach 1: Multi-Persona Ensemble (~30s)
Our first method was tested on the IPAD button videos.
One single prompt was not stable. Sometimes it was right, sometimes wrong.
So we used a trick called **multi-persona voting**. We create many "inspector" characters.
Each one looks at the same scene, but with a different point of view — like an operator, an engineer, or a vision expert.
Then they **vote**. The majority decides. This made the system much more stable.

## Slide 5 — The VERA Paper (~30s)
Our second method is based on a paper called **VERA**, from CVPR 2025.
Its main idea is very useful. If you ask the model a vague question like "is there an anomaly?", the result is poor.
But if you ask **concrete, learned questions**, the result is much better — and the model stays frozen.
VERA also learns these questions automatically: one model answers, another model rewrites the questions when there are mistakes.
We took this core idea and made a simpler version for the factory.

## Slide 6 — Approach 2: VERA-lite (~30s)
We call our version **VERA-lite**.
A **learner** model watches the video and answers our guiding questions with a clear yes or no.
An **optimizer** model reads the mistakes and improves the questions — without changing the model weights.
One important point: the model **never sees the correct label**. This is a fair, blind test.
With better questions, our accuracy went up from 0.77 to 0.92.

## Slide 7 — Inside a Verdict (~25s)
This is what makes our system trustworthy.
For every clip, the model returns a **structured answer**. For each question, it gives yes or no, plus the visual evidence.
In this example, one pen is blue while the others are black. So question three is "NO".
That single "NO" creates the final decision: anomaly, type "wrong color".
An operator can read this and understand exactly *why*.

## Slide 8 — Representation & Ensemble (~30s)
We also found an interesting problem.
When we send the **full video**, the model misses small movements, because it only looks at a few frames inside.
When we send **dense frames**, it catches the movement, but misses some other cases.
So the two methods have **different blind spots**.
The solution is simple: we **combine** them. If either one says anomaly, it is an anomaly. Together, they reach 13 out of 13.

## Slide 9 — Results & Conclusion (~30s)
Finally, our results.
The model we use is a reasoning model, so it was sometimes slow. We added a thinking limit. This made it **41% faster** and also **deterministic** — the same input always gives the same answer.
The honest, repeatable accuracy is **0.846**, and the ensemble reaches **100%** with **zero false alarms**.
In conclusion: explainable anomaly detection **without any training data** is possible for industry.
In the future, we want a cheap detector first, and the VLM only on suspicious moments, for real-time use.
Thank you for listening.
