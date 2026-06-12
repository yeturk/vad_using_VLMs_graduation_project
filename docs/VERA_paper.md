# VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models

**Muchao Ye¹\*  Weiyang Liu²  Pan He³**

¹ The University of Iowa  ² Max Planck Institute for Intelligent Systems, Tübingen  ³ Auburn University

¹ muye@uiowa.edu  ² weiyang.liu@tuebingen.mpg.de  ³ pan.he@auburn.edu  \* Corresponding Author

🔗 https://vera-framework.github.io

> arXiv:2412.01095v3 [cs.AI] 31 Mar 2025

---

## Abstract

The rapid advancement of vision-language models (VLMs) has established a new paradigm in video anomaly detection (VAD): leveraging VLMs to simultaneously detect anomalies and provide comprehendible explanations for the decisions. Existing work in this direction often assumes the complex reasoning required for VAD exceeds the capabilities of pretrained VLMs. Consequently, these approaches either incorporate specialized reasoning modules during inference or rely on instruction tuning datasets through additional training to adapt VLMs for VAD. However, such strategies often incur substantial computational costs or data annotation overhead. To address these challenges in explainable VAD, we introduce a verbalized learning framework named **VERA** that enables VLMs to perform VAD without model parameter modifications. Specifically, VERA automatically decomposes the complex reasoning required for VAD into reflections on simpler, more focused guiding questions capturing distinct abnormal patterns. It treats these reflective questions as learnable parameters and optimizes them through data-driven verbal interactions between learner and optimizer VLMs, using coarsely labeled training data. During inference, VERA embeds the learned questions into model prompts to guide VLMs in generating segment-level anomaly scores, which are then refined into frame-level scores via the fusion of scene and temporal contexts. Experimental results on challenging benchmarks demonstrate that the learned questions of VERA are highly adaptable, significantly improving both detection performance and explainability of VLMs for VAD.

---

## 1. Introduction

Video anomaly detection (VAD) aims to automatically identify unexpected and abnormal events in video sequences, with broad applications ranging from autonomous driving [2] to industrial manufacturing [31]. While achieving good performance in VAD is essential, providing clear explanations for detected anomalies is even more crucial.

To this end, our work primarily focuses on explainable VAD, which requires both comprehensive visual understanding and the ability to generate human-interpretable predictions. The rapid advancement of vision-language models (VLMs) [7, 18, 21, 58] enables us to address both requirements through their strong visual reasoning and language interaction capabilities. As multi-modal architectures that effectively combine the reasoning capabilities from large language models (LLMs) [4] and the visual understanding capabilities from pretrained vision encoders [8], VLMs are particularly well-suited for VAD for they can offer explainable predictions that clearly illustrate the rationale behind specific anomalies, making the results more interpretable to users. Recent research on VAD has consequently focused on how to effectively leverage the power of pretrained VLM. As shown in Fig. 1, existing approaches aim to address the misalignment problem between VLMs' pretraining tasks and the VAD requirements through either additional reasoning modules or instruction tuning (IT):

- **One line of research introduces external LLMs to assist frozen VLMs to reason in VAD** [46, 52]. It uses VLMs to caption what they see given a video, and the descriptions are then passed to an external LLM, *e.g.*, GPT-4 [1], to reason whether an anomaly occurs.
- **Another line of research, instead, expands VLMs to generate explainable prediction via IT** [26, 55]. This research line creates additional VAD datasets with frame-level annotations and leverages exemplary instructions to fine-tune the VLM, enabling it to detect anomalies and generate human-interpretable explanations.

![Figure 1](figs/fig1.png)

***Figure 1.*** *VERA renders frozen VLMs to describe and reason with learnable guiding questions learned from coarsely labeled data. (Existing pipeline 1 uses external LLMs to assist frozen VLMs in reasoning; existing pipeline 2 utilizes instruction tuning to make VLMs integrated VAD systems; VERA employs learnable guiding questions to elicit the reasoning of frozen VLMs. Legend: → Inference, ⇢ Training, ❄ Frozen Module, 🔥 Learnable Module, 📋 Extra Annotation.)*

**Key Observations and Research Question.** While prior research demonstrates the potential of applying VLMs to VAD, we identify that this new paradigm is hindered by a shared critical issue: the use of additional reasoning modules or fine-grained labeled datasets incurs significant computational cost either in the inference or training phases. First, decoupling a VAD system into a frozen VLM and an extra LLM introduces more overhead in inference, because it separates the description generation and reasoning processes. Secondly, although IT-based methods enable VLMs to effectively integrate description and reasoning for VAD, they require additional manpower and computational resources for annotating and finetuning on fine-grained labeled instruction datasets, which is time-consuming and not scalable for large-scale datasets. In light of this, we investigate the following unexplored yet important question:

> *Can we enable a frozen VLM to integrate description and reasoning for VAD without instruction tuning?*

**Our Approach.** This research question is nontrivial because the reasoning ability of a frozen VLM is limited in general visual tasks, and it struggles to handle complex reasoning tasks like VAD, which requires the understanding of subtle, context-dependent outliers. To illustrate, Table 1 shows that prompting frozen VLMs with simple VAD questions used in existing works leads to unsatisfactory results. Thus, instruction-tuning a VLM seems necessary to make it responsive to specific instructional cues and capture delicate visual variations. In this paper, we question the necessity of such an operation and propose a principled approach to tailor frozen VLMs for VAD.

Specifically, our solution is guided by the intuition that the reasoning ability of VLMs for VAD will improve if we find questions with suitable and concrete description of abnormal patterns rather than with abstract and general words like "anomaly" to prompt them. Our idea is to iteratively refine anomaly descriptions from abstract ones (*e.g.*, "is there any anomaly?") to detailed, specific characterizations.

Driven by such insight, we propose a framework, termed **VERA**, to explore verbalized learning (VL) for VAD. This framework considers the practical constraint that it is suboptimal to manually write down VAD guiding questions across VLMs, so it introduces a data-driven learning task to identify suitable anomaly-characterization questions containing concrete abnormal patterns for the frozen VLM using coarsely labeled datasets, eliminating the need for IT. Specifically, in the training phase, VERA treats the questions guiding the reasoning of VLMs in VAD as learnable parameters, improving them based on the verbal feedback from an optimizer VLM on the performance of a learner VLM on an intermediate VAD subtask—binary video classification for each video in the VAD training set. This design is both efficient and appropriate for VAD, as it accounts for video-specific properties like temporality while relying solely on provided coarse video-level labels. After that, considering the large scale of video frames, VERA assigns a fine-grained anomaly score for each frame in a coarse-to-fine manner in the inference phase. First, VERA generates segment-level anomaly scores by querying VLMs with the learned guiding questions. Next, VERA improves the initial score by incorporating scene context into each segment score via ensembling. Finally, VERA outputs frame-level scores by fusing temporal context via Gaussian smoothing and frame-level position weighting.

***Table 1.*** *Instructing a frozen VLM (InternVL2-8B [7]) with simple questions to perform VAD yields poor AUC on UCF-Crime [32] dataset.*

| VAD Question for InternVL2-8B | AUC (%) |
|---|---|
| "Describe the video and is there any anomaly?" [26] | 53.05 |
| "Are there any abnormal events in the video?" [55] | 65.03 |

**Contributions.** To sum up, our contributions are:

- To our knowledge, we present the first approach, that is, VERA, to adapt frozen VLMs as an integrated system for VAD by learning detailed anomaly-characterization questions in prompts that decompose anomalies into concrete and recognizable patterns. VERA learns them directly from coarsely labeled datasets, eliminating the need for IT or external reasoning modules.
- We introduce an effective VL-based algorithm for VLMs in VAD, allowing direct adaptation without modifying model parameters. With coarse labeled VAD datasets only, our approach obtains good guiding questions in VAD by relying on the verbal interaction between learner and optimizer VLMs in verbalized training. Additionally, we design a coarse-to-fine strategy to derive frame-level anomaly scores from verbally learned guiding questions in VAD, integrating both scene and temporal contexts for better VAD performance and reasoning.
- The learned guiding questions from VERA are expressed in natural languages, providing a unified method to encode and transfer prior VAD knowledge seamlessly to other datasets or VLMs. In challenging VAD datasets like UCF-Crime [32] and XD-Violence [42], VERA achieves state-of-the-art explainable VAD performance and enjoys good generalization ability across models and datasets.

---

## 2. Related Work

**Video Anomaly Detection.** VAD is the task of localizing frames that contain abnormal events in a given video. This task is challenging for anomalies cover a broad scope of events like accidents and criminal activities while training sets only offer coarse annotations. Modern VAD methods are based on deep neural networks (DNNs) for their superiority and are going through a paradigm shift in using VLMs:

1. **Early DNNs for VAD are task-specific**, which often employ unsupervised (including one-class) or weakly supervised (WS) learning techniques for training. Most unsupervised learning methods [23, 25, 37, 38, 48, 56] train DNNs on frame reconstruction/prediction tasks to establish representation spaces for normal/abnormal videos. WS learning methods [5, 27, 32, 44, 47, 53] leverage both normal and abnormal videos to train a feature extractor that distinguishes anomalies from normalcy, typically using multiple instance learning [32] objectives.
2. **Recent VAD methods adopt VLMs** due to their remarkable success across core vision tasks [12, 21, 28, 33]. Early research [26, 46, 52, 55] has leveraged VLMs to generate textual descriptions of detected anomalies to enhance prediction explainability for VAD. However, current approaches incur high processing demands from external LLMs or require substantial effort and cost for fine-tuning on additional datasets, which are computationally inefficient in training or inference. Our work reduces the processing overhead by adapting frozen VLMs for VAD without model parameter modification or extra reasoning modules via learnable guiding questions, which elicit superior reasoning from frozen VLMs and significantly boost their performance in VAD.

**Verbalized Learning for VLMs.** The designed VL framework is inspired by a recent technique called verbalized machine learning (VML) [45]. The main idea of VML is to use LLMs to approximate functions and learn the verbal rules and descriptions of performing specific tasks, which casts traditional machine learning tasks as language-based learning tasks. This approach regards the language expressions that define classification rules and other task-specific criteria as learned parameters, and optimize them in a data-driven fashion through interactions between a learner and an optimizer modeled by LLMs or VLMs. However, the VML framework is limited to tasks involving regression on scalar values or classification for static images. A similar idea has also been explored in a concurrent method, TextGrad [49], which integrates the process of incorporating textual feedback from LLMs for improving prompts in PyTorch and further proves its effectiveness in coding, question answering, and optimization in chemistry and medicine. Compared to existing works, our work pioneers VL for the VAD task and video data, which remains unsolved for previous VL frameworks focus on static-data tasks and cannot handle the challenges of temporality and scene dynamics in videos. Specifically, VERA introduces a new learning paradigm for VAD: generating effective questions that encapsulate key abnormal patterns in videos to elicit the reasoning ability from VLMs for explainable VAD. Additionally, VERA works for any VAD dataset and supports WS learning. Unlike previous WS methods, VERA only needs to learn concise text but not millions of parameters, so the training is lightweight.

---

## 3. The VERA Framework

Our approach adapts VLMs to detect video anomalies without additional reasoning modules or IT. We now formulate the VAD task and detail the design of VERA.

### 3.1. Problem Formulation

**Video Anomaly Detection.** Let $V$ be a video with $F$ frames, represented as $V = \{I_i\}_{i=1}^{F}$, where $I_i$ is the $i$-th frame ($1 \le i \le F$). Our objective is to locate and detect the start and end of anomalous events within $V$. In standard labeling, any frame associated with an anomaly is labeled as 1, and normal frames are labeled as 0. Therefore, the ground truth label sequence for $V$ is $Y = [y_1, \dots, y_F]$, where $y_i \in \{0, 1\}$ represents the fine-grained label for $I_i$. We aim to use a frozen VLM, $f_{\text{VLM}}$, to generate anomaly score predictions across all frames, $\hat{Y} = [\hat{y}_1, \dots, \hat{y}_F]$, where $\hat{y}_i \in [0, 1]$ is a continuous anomaly score for $I_i$.

**Available Training Data for VAD.** Typically, VAD datasets only provide coarsely labeled training sets [23, 25, 32, 42]. We denote a VAD training set as $\mathcal{D} = \{(V^{(j)}, Y^{(j)})\}_{j=1}^{N}$, where $N$ is the total number of training videos, $V^{(j)}$ represents the $j$-th video ($1 \le j \le N$) and $Y^{(j)}$ is the corresponding video-level label. $Y^{(j)} = 1$ if $V^{(j)}$ contains any anomaly defined by the dataset annotators, *e.g.*, abuse or arson activities, and $Y^{(j)} = 0$ if $V^{(j)}$ has no anomalies. For $V^{(j)}$, we suppose it contains $F_j$ frames and denote the frames sequence as $V^{(j)} = \{I_i^{(j)}\}_{i=1}^{F_j}$, where $I_i^{(j)}$ is the $i$-th frame ($1 \le i \le F_j$) in $V^{(j)}$.

### 3.2. Training in VERA

**Training Objective.** We aim to learn guiding questions that break down a complex and ambiguous concept (i.e., what is an "anomaly") into a set of identifiable anomalous patterns to unlock reasoning capabilities within frozen VLMs for VAD tasks. Those patterns vary among datasets, making manually designed descriptions ineffective for generalization. To address this, we propose a general VL framework shown in Fig. 2 to generate the desired guiding questions. We denote the guiding question set as $\mathbf{Q} = \{q_1, \dots, q_m\}$, where $q_i$ is the $i$-th question ($1 \le i \le m$) and $m$ is the number of questions. The training framework considers $\mathbf{Q}$ as the learnable parameters, which are optimized through verbal interaction between a learner and an optimizer, modeled by VLMs through leveraging their ability to follow instructions with given prompts.

**Training Data.** The training data for learning $\mathbf{Q}$ consist of paired sampled video frames and video-level labels. Sampling is necessary because the amount of video frames is so huge that we cannot compute with every frame. We explore three types of sampling strategies and find that uniform sampling [54] yields the best results. That is, with any video $V^{(j)} \in \mathcal{D}$, we first calculate the interval between sampled frames as $l = \lfloor F_j / S \rfloor$, where $S$ is the number of sampled frames, and $\lfloor \cdot \rfloor$ (floor) denotes rounding down to the nearest integer. Given $l$, the uniformly sampled frames from $V^{(j)}$ are represented by $\tilde{V}^{(j)} = [I_1^{(j)}, I_{l+1}^{(j)}, \dots, I_{(S-1)\cdot l + 1}^{(j)}]$. The label used for training is $Y^{(j)}$ only, resulting in training data pairs $\{(\tilde{V}^{(j)}, Y^{(j)})\}_{j=1}^{N}$ for VERA.

![Figure 2](figs/fig2.png)

***Figure 2.*** *The overall training pipeline in VERA aims to optimize VAD guiding questions iteratively. In each iteration $t$, the optimization is verbalized by providing verbal instructions for the learner and optimizer to follow. They will generate predictions and new guiding questions, respectively.*

> **Learner Prompt Template θ (excerpt from Fig. 2):**
> *You are the model. **Model Description:** You are designed to do binary classification. The input is a sequence of video frames for identifying whether there is an anomaly in the video; you need to output the class label, i.e., an integer in the set {0, 1}. 0 represents normal video, and 1 represents abnormal video. Please answer the prompt questions. **Prompt Questions:** Answer the following questions based on what you see from the video frames and provide an explanation in one sentence. [Guiding Questions $\mathbf{Q}_t$]. Based on the analysis above, please conclude your answer to "Is there any anomaly in the video?" in "Yes, there is an anomaly" or "No, there is no anomaly". **Input:** [Video Frames $\tilde{V}^{(j)}$]. Please give your output strictly in the following format:*
> - *Answers to Prompt Questions: [Provide your analysis by answering the questions listed in Prompt Questions.]*
> - *Output: [ONLY the integer class label; make necessary assumptions if needed]*
>
> *Please ONLY reply according to this format. Don't give me any other words.*

> **Optimizer Prompt Template ψ (excerpt from Fig. 2):**
> *You are the optimizer for a model. Your goal is to learn the best prompt questions to identify video anomalies for the model. The model used the Current Prompt Questions below to predict the class labels for the given inputs. You are given the target labels. Please optimize the Current Prompt Questions for better prediction. **Inputs:** [A batch of input video frames $V_{\text{batch}}$]. **Model Description:** (same binary-classification description as above). **Current Prompt Questions:** (current $\mathbf{Q}_t$). **The Model Predictions:** [A batch of predictions $\hat{Y}_{\text{batch}}$ made by the learner]. **The Targets:** [A batch of video-level ground truths $Y_{\text{batch}}$ for the input video frames]. If the model is doing well, you can keep using the current prompt questions. However, if the model is not performing well, please update the model by improving upon the "Current Prompt Questions", which should result in lower classification error both on the current and the next batch of i.i.d. data. Limit your "New Prompt Questions" to be no more than five questions! Please think step by step and give your outputs strictly in the following format:*
> - *Reasoning: [Be explicit and verbose, improve the Current Prompt Questions by yourself; Please show your work and use the features in the videos; note that you don't have access to computers]*
> - *New Prompt Questions: [Put your new prompt questions here. The questions MUST be based on the features in the input videos. Please limit the number of questions to be at most five!]. Please ONLY reply according to this format. Don't give me any other words.*
>
> *Example guiding questions shown in Fig. 2:*
> 1. *Are there any unusual movements of the vehicles in the video frames?*
> 2. *Are there any unusual behaviors of the people in the video frames?*
> 3. *Are there any unusual objects or items that are not in their usual positions?*
> 4. *Are there any objects or people that appear out of place in the scene?*
> 5. *Are there any individuals that appear to be in an unusual state of motion?*

**Updating Q via Learner and Optimizer.** Since $\mathbf{Q}$ are verbal expressions for specific anomaly patterns, VERA inherits the idea of VML [45] in training: optimizing language-based parameters by verbal communication between a learner agent $f_{\text{learner}}$ and an optimizer agent $f_{\text{opt}}$, rather than by numerical optimization algorithms like Adam [16]. W.l.o.g., we take an arbitrary iteration $t$ when implementing the complete algorithm (detailed in Supplementary Material) for illustration. We denote any LLM-based model as $f(x; \phi)$ where $x$ represents the input data, and $\phi$ denotes the natural language instructions for $f$ to follow, which is considered as learnable parameters in our VL framework. Specifically, $\mathbf{Q}$ contains parameters to be learned in VERA. As depicted in Fig. 2, in each iteration $t$, the learner agent $f_{\text{learner}}^{(t)}$ is modeled by the frozen VLM $f_{\text{VLM}}(\cdot)$ used for VAD with a specific prompt template $\theta$ that guide $f_{\text{VLM}}(\cdot)$ to conduct a learning task by pondering on current guiding questions $\mathbf{Q}_t$. We denote the learner agent as $f_{\text{learner}}^{(t)}(x) = f_{\text{VLM}}(x; (\theta, \mathbf{Q}_t))$, where $x$ is the input in a learning task, and $\mathbf{Q}_t$, the learnable guiding questions applied in each iteration $t$, constitutes the core parameters that distinguish the learner between iterations. Meanwhile, we introduce an optimizer $f_{\text{opt}}^{(t)}$ to assess the quality of the predictions of the learner and to optimize $\mathbf{Q}_t$. W.l.o.g., we use the same frozen VLM $f_{\text{VLM}}$ to model the optimizer. As demonstrated in Fig. 2, we provide another specific prompt template $\psi$ for the learner to follow to optimize $\mathbf{Q}_t$, so we denote the optimizer agent as $f_{\text{opt}}^{(t)}(z) = f_{\text{VLM}}(z; (\psi, \mathbf{Q}_t))$, where $z$ is its input and $\psi$ is the instruction to improve $\mathbf{Q}_t$. It is important to note that $f_{\text{learner}}^{(t)} \neq f_{\text{opt}}^{(t)}$ because $f_{\text{learner}}^{(t)}$ follows $(\theta, \mathbf{Q}_t)$ to conduct a learning task, while $f_{\text{opt}}^{(t)}$ follows $(\psi, \mathbf{Q}_t)$ to refine $\mathbf{Q}_t$.

**Learning Task for $f_{\text{learner}}$.** The learner executes the "forward pass" and outputs a prediction. Recall that we only use the original coarsely labeled information for training. Thus, we design a binary classification task for $f_{\text{learner}}$, which accounts for the temporal nature of video data, the sparsity of anomalies, and the weak supervision in VAD datasets. In this task, the job of the learner $f_{\text{learner}}$ is to produce a binary classification prediction $\hat{Y}^{(j)}$ to determine whether there is an anomaly in the video based on the sampled frames $\tilde{V}^{(j)}$. As shown in Fig. 2, we explain the task in natural language in the "Model Description" section in $\theta$. Guiding questions $\mathbf{Q}_t$ are inserted in the "Prompt Questions" section in $\theta$ to elicit reasoning of the VLM. This template design is based on the prompt structures used in VML, with targeted modifications to help the learner effectively address this WS learning task. Given $\theta$ and a sampled frame set $\tilde{V}^{(j)}$, the learner will output a prediction as

$$\hat{Y}^{(j)} = f_{\text{learner}}^{(t)}(\tilde{V}^{(j)}), \tag{1}$$

where $\hat{Y}^{(j)} = 1$ if the learner thinks there is an anomaly after skimming across the sampled frames $\tilde{V}^{(j)}$ and reasoning through the guiding questions $\mathbf{Q}_t$, and otherwise, $\hat{Y}^{(j)} = 0$.

**Optimization Step in $f_{\text{opt}}$.** The optimizer executes the "backward pass" to update the questions $\mathbf{Q}_t$ via a mini-batch (batch size is $n$). Suppose the visual input in a batch is $V_{\text{batch}} = [\tilde{V}_{\text{batch}}^{(1)}, \dots, \tilde{V}_{\text{batch}}^{(n)}]$ and the corresponding ground truths are $Y_{\text{batch}} = [Y_{\text{batch}}^{(1)}, \dots, Y_{\text{batch}}^{(n)}]$. The learner generates prediction as $\hat{Y}_{\text{batch}} = [\hat{Y}_{\text{batch}}^{(1)}, \dots, \hat{Y}_{\text{batch}}^{(n)}]$ with the current questions $\mathbf{Q}_t$ by Eq. (1). The optimizer will output a new set of questions $\mathbf{Q}_{t+1}$ by following the prompt $\psi$ with batched data. We denote the optimization step as

$$\mathbf{Q}_{t+1} = f_{\text{opt}}^{(t)}(V_{\text{batch}}, \hat{Y}_{\text{batch}}, Y_{\text{batch}}), \tag{2}$$

where $\mathbf{Q}_{t+1}$ is a new set of guiding questions constructed from $f_{\text{opt}}^{(t)}$ owing to its text generation and instruction following abilities after reading $\psi$.

### 3.3. Inference in VERA

During training, we denote the one with the largest validation accuracy as $\mathbf{Q}^*$. In inference, given $\mathbf{Q}^*$, VERA yields fine-grained anomaly score $\hat{Y}$ for a test video $V$ via a coarse-to-fine process shown in Fig. 3.

![Figure 3](figs/fig3.png)

***Figure 3.*** *VERA computes anomaly scores with $\mathbf{Q}^*$ in three steps.*

**Step 1: Initial Anomaly Scores via Learned Guiding Questions.** We divide the video into segments and analyze each segment independently first. Following [52], we perform equidistant frame sampling within $V$ to obtain the set of segment centers $\mathcal{C} = \{I_1, I_{d+1}, \dots, I_{(h-1)\cdot d + 1}\}$, where $d$ is the interval between centers and $h = \lfloor F/d \rfloor$ is the total number of segments. For each center frame $I_{(u-1)\cdot d + 1}$ ($1 \le u \le h$), we define a 10-second window around it as the $u$-th segment, within which we uniformly sample 8 frames. We denote the sampled frame set in the $u$-th segment as $V_u$. Next, we input $V_u$ in $f_{\text{VLM}}$ with the prompt $(\theta, \mathbf{Q}^*)$ to get the initial score

$$\tilde{y}_u = f_{\text{VLM}}(V_u; (\theta, \mathbf{Q}^*)), \tag{3}$$

where $\tilde{y}_u = 1$ if $f_{\text{VLM}}$ thinks the segment contains an anomaly after reasoning via $\mathbf{Q}^*$ with $V_u$, and otherwise, $\tilde{y}_u = 0$. By repeating Eq. (3) for each segment, we have a segment-level initial anomaly score set $\tilde{Y} = [\tilde{y}_1, \dots, \tilde{y}_h]$.

**Step 2: Ensemble Segment-Level Anomaly Scores with Scene Context.** Note that the scores derived above only examine a short moment in a long video without considering any context. To resolve it, we refine the initial segment-level score by incorporating scene context—defined as preceding and following segments that contain similar elements, such as actors and background, to those in the current segment. We measure the relevance between different video segments by the cosine similarity of their feature representations [22], extracted by a pretrained vision feature extractor $g$, *e.g.*, ImageBind [10]. For the $u$-th segment $V_u$, its similarity with any segment $V_w$ ($1 \le w \le h$) is $\text{sim}(u, w) = \cos\left(\frac{e_u \cdot e_w}{\|e_u\| \cdot \|e_w\|}\right)$, where $\cos$ denotes the cosine function, and $e_u = g(V_u)$ and $e_w = g(V_w)$ represent their features. Let $\kappa_u = [\kappa_u^{(1)}, \dots, \kappa_u^{(K)}]$ denote the indices of the top-$K$ segments similar to $V_u$. We refine the anomaly score by

$$\bar{y}_u = \sum_{i=1}^{K} \tilde{y}_{\kappa_u^{(i)}} \cdot \frac{\exp\big(\text{sim}(u, \kappa_u^{(i)})/\tau\big)}{\sum_{j=1}^{K} \exp\big(\text{sim}(u, \kappa_u^{(j)})/\tau\big)}, \tag{4}$$

where $\bar{y}_u$ is an ensemble of initial scores of top-$K$ video segments relevant to $V_u$. Here, the initial score of each retrieved segment is weighted by a factor derived from the cosine similarity and normalized by the Softmax function (with $\tau$ as the temperature hyperparameter). Accordingly, scenes with greater similarity are assigned higher weights, making the ensemble score a more comprehensive reflection of anomalies with the video context. By applying Eq. (4) for all segments, we obtain $\bar{Y} = [\bar{y}_1, \dots, \bar{y}_h]$.

**Step 3: Frame-level Anomaly Scoring with Temporal Context.** Given $\bar{Y}$, we aim to incorporate temporal context to capture how events evolve over time when computing frame-level anomaly scores, for the abnormality of an event often depends on the timing and progression of observed activities. To detail, we first apply Gaussian smoothing [11] to aggregate local temporal context into the segment-level anomaly scores. We denote the Gaussian kernel (suppose the filter size is $\omega$) as $G(p) = \exp\left(\frac{-p^2}{2\sigma_1^2}\right)$ where $p$ is the distance from the kernel center and $\sigma_1$ is the variance. We update segment-level scores as $\bar{\Gamma} = \bar{Y} * G = [\bar{\gamma}_1, \dots, \bar{\gamma}_h]$, where $*$ is the convolution operation. Next, we integrate global temporal context by position weighting. With $\bar{\Gamma}$, we flatten it into frame-level scores by assigning the score $\bar{\gamma}_u$ to each frame in the $u$-th segment, *i.e.*, $[I_{(u-1)\cdot d + 1}, \dots, I_{u\cdot d}]$. We denote the frame-level score sequence after flattening as $[\rho_1, \dots, \rho_F]$. We then apply the Gaussian function to encode position weights as $w(i) = \exp\left(\frac{-(i-c)^2}{2\sigma_2^2}\right)$, where $i$ ($1 \le i \le F$) is any frame index, $c = \lfloor F/2 \rfloor$ is the center frame index, and $\sigma_2$ is the variance. The anomaly score for the $i$-th frame is:

$$\hat{y}_i = w(i) \cdot \rho_i. \tag{5}$$

This operation scales the score $\rho_i$, diminishing the anomaly score for frames near the beginning and end of the event. This helps better capture the temporal progression of anomalies: the score gradually increases as the anomaly reaches its peak and decreases afterward. The final scores is denoted as $\hat{Y} = [\hat{y}_1, \dots, \hat{y}_F]$ after applying Eq. (5).

**Explainable VAD by VERA.** When using template $\theta$ embedded with $\mathbf{Q}^*$ to compute $\hat{Y}$, we ask the VLM to "provide an explanation in one sentence" when reasoning, and VLM will explain the anomaly score it assigns based on $\mathbf{Q}^*$.

---

## 4. Experiments and Results

In this section, we present an evaluation of VERA as follows, addressing key questions of interest including: **(Q1)** Does it enhance the effectiveness of frozen VLMs in VAD? **(Q2)** Is its design reasonable and well-structured? **(Q3)** How well does it generalize across different scenarios?

### 4.1. Experimental Settings

**Datasets.** We conduct experiments on two large-scale VAD datasets: (1) **UCF-Crime** [32] collected from surveillance videos with 13 types of anomalies and 290 (140 abnormal) test videos (2.13 minutes long on average). (2) **XD-Violence** [42] with 6 anomaly categories and 800 (500 abnormal) test videos (1.62 minutes long on average).

**Metrics.** Following approaches in [52, 55], we mainly evaluate VAD performance using the Area Under the Curve (AUC) of the frame-level Receiver Operating Characteristic (ROC) curve, as it provides a comprehensive measure of model performance across all thresholds.

**Baselines.** We categorize baselines into non-explainable approaches and explainable ones as [55] does. Non-explainable ones are obtained by WS learning [6, 9, 15, 17, 19, 32, 36, 41–43, 50, 51, 57] and unsupervised learning [13, 25, 34, 35, 37, 38]. These non-explainable approaches cannot provide language-based explanations for VAD. For explainable approaches, we use LAVAD [52], Holmes-VAD [55], and VADor [26] as representatives of Pipeline 1 and Pipeline 2 shown in Fig. 1. It should be noted that [46] does not report performance on UCF-Crime and XD-Violence. Additionally, we include zero-shot (ZS) VAD by frozen VLMs designed by [52] as baselines.

**Implementation of VERA.** In our experiments, we choose a small VLM, InternVL2-8B [7], as the backbone $f_{\text{VLM}}$ for building VERA by default, if not otherwise specified. We also explore other backbones, such as Qwen2-VL-7B [40] and InternVL2-40B [7] for ablation. We train $\mathbf{Q}$ for no more than 10 epochs, with a validation accuracy calculated every 100 iterations to determine $\mathbf{Q}^*$. We set $n$ as 2, $S$ as 8, and $m$ as 5 for training. The initial questions $\mathbf{Q}_0$ is *"1. Is there any suspicious person or object that looks unusual in this scene? 2. Is there any behavior that looks unusual in this scene?"*, inspired by previous VAD methods [13, 43], which assume anomalies appear with unusual appearance or motions.

### 4.2. Comparison to State-of-the-art Methods

We address Q1 by empirically comparing VERA to existing VAD methods. First, in Table 2, VERA achieves the highest AUC among explainable VAD methods on UCF-Crime, outperforming Holmes-VAD and VADor (without IT, as reported in their papers) in a fair comparison. Importantly, unlike these methods, VERA does not need to modify the model parameters, demonstrating its suitability to directly adapt VLM to the VAD task with minimal training requirements. Moreover, VERA surpasses LAVAD by 6% in AUC on UCF-Crime, uniquely integrating both description and reasoning capabilities in VAD. Compared to non-explainable methods, VERA achieves AUC performance that is comparable to one of the top-performing methods, CLIP-TSA, on UCF-Crime, while offering the additional advantage of explainable predictions.

***Table 2.*** *AUC (%) on UCF-Crime. No IT is used for Holmes-VAD and VADor.*

| Method | AUC |
|---|---|
| **Non-explainable VAD Methods** | |
| Wu et al. [42] | 82.44 |
| OVVAD [43] | 86.40 |
| S3R [41] | 85.99 |
| RTFM [36] | 84.30 |
| MSL [19] | 85.62 |
| MGFN [6] | 86.98 |
| SSRL [17] | 87.43 |
| CLIP-TSA [15] | 87.58 |
| Sultani et al. [32] | 77.92 |
| GCL [51] | 79.84 |
| GCN [57] | 82.12 |
| MIST [9] | 82.30 |
| CLAWS [50] | 83.03 |
| DYANNET [35] | 84.50 |
| Tur et al. [37] | 66.85 |
| GODS [38] | 70.46 |
| **Explainable VAD Methods** | |
| LAVAD [52] | 80.28 |
| Holmes-VAD [55] | 84.61 |
| VADor [26] | 85.90 |
| ZS CLIP [52] | 53.16 |
| ZS IMAGEBIND-I [52] | 53.65 |
| ZS IMAGEBIND-V [52] | 55.78 |
| LLAVA-1.5 [20] | 72.84 |
| **VERA** | **86.55** |

Similar advantages are also observed in Table 3 for XD-Violence. Considering multiple factors, including performance, training efficiency, system integration, and explainability, VERA stands out as a promising pipeline for VLMs in VAD.

***Table 3.*** *AUC (%) on XD-Violence.*

| Method | AUC |
|---|---|
| **Non-Explainable VAD Methods** | |
| Hasan et al. [13] | 50.32 |
| Lu et al. [25] | 53.56 |
| BODS [38] | 57.32 |
| GODS [38] | 61.56 |
| RareAnom [34] | 68.33 |
| **Explainable VAD Methods** | |
| LAVAD [52] | 85.36 |
| ZS CLIP [52] | 38.21 |
| ZS IMAGEBIND-I [52] | 58.81 |
| ZS IMAGEBIND-V [52] | 55.06 |
| LLAVA-1.5 [20] | 79.62 |
| **VERA** | **88.26** |

### 4.3. Ablation Studies

We perform necessary ablation studies on UCF-Crime to answer both Q2 and Q3 for a comprehensive evaluation on our design choices.

**Frame Sampling Strategy in Training.** We compare three frame sampling strategies for obtaining each $\tilde{V}^{(j)}$ in training: uniform sampling, random sampling, and TSN sampling (random sampling from equally divided segments). Table 4 shows that uniform sampling performs the best (with batch size $n = 2$ and $S = 8$). This is because uniform sampling preserves the temporal structure and maintains consistent motion patterns throughout the long video, making it easier for VLMs to understand the video and update $\mathbf{Q}$.

***Table 4.*** *Sampling strategies explored in VERA training.*

| Strategy | AUC (%) |
|---|---|
| Random [3] | 83.63 |
| TSN [39] | 82.63 |
| Uniform [54] | **86.55** |

**How to Obtain Guiding Questions Q for VLM.** As seen in Table 5, if the guiding questions are not incorporated into the VLM prompt, the AUC will drop largely to 78.81%, confirming the need to use simpler and more focused questions to provoke reasoning in the VLMs for VAD. Meanwhile, if we use manually written questions ($\mathbf{Q}_0$), the performance is suboptimal with an 81.15% AUC, which shows the need to use VL to find guiding questions. Lastly, if we only input batched predictions $\hat{Y}_{\text{batch}}$ and ground truths $Y_{\text{batch}}$ without inputting $V_{\text{batch}}$ in the optimizer, the $\mathbf{Q}$ updated in this way will dumb the VLMs and make it have a low AUC. Thus, inputting video frames as Eq. (2) does is necessary to learn good $\mathbf{Q}$.

***Table 5.*** *The way we obtain guiding questions affects AUC substantially.*

| Question Type | AUC (%) |
|---|---|
| No questions | 78.81 |
| Manually written questions by human | 81.15 |
| Learned questions w/o iteratively inputting $V_{\text{batch}}$ in Eq. (2) | 78.06 |
| Iteratively learned questions (used in VERA) | **86.55** |

**Number of Questions $m$.** As shown in Fig. 4, when $m$ is set to 1, the reasoning is limited to a single perspective, resulting in a lower AUC. As $m$ increases up to 5, the model captures more comprehensive anomaly patterns, leading to improved AUC. However, increasing $m$ beyond 5 yields no significant gains. Therefore, we set $m$ to 5 by default in VERA, if not otherwise specified.

![Figure 4](figs/fig4.png)

***Figure 4.*** *Effect of the number of guiding questions on AUC. (AUC values: 0 questions = 78.81; m=1 → 77.31; m=3 → 83.76; m=5 → 86.55; m=7 → 85.79.)*

**Coarse-to-Fine Anomaly Score Computation.** We also validate the anomaly score computation by VERA. Table 6 shows the AUC is 76.10% when using the flattened initial score obtained in Step 1, and leveraging retrieved segments in Step 2 significantly boosts the AUC to 84.53%, highlighting the effectiveness of incorporating ensemble scores based on scene context. Meanwhile, smoothing and weighting in Step 3 further improves the AUC by around 1% each, verifying the benefit of integrating temporal context.

***Table 6.*** *Ablation study of each step in VERA's inference.*

| Operation | AUC (%) |
|---|---|
| Initial (Step 1) | 76.10 |
| Initial + Retrieval (Step 2) | 84.53 (+8.43) |
| Initial + Retrieval + Smoothing (Step 3) | 85.48 (+0.95) |
| Initial + Retrieval + Smoothing + Weighting (Step 3) | 86.55 (+1.07) |

**Generalizability Test.** We further examine the generalizability of VERA across different model sizes, VLM architectures, and datasets to address Q3.

First, we apply VERA to InternVL2-40B, a larger model in the InternVL2 family compared to InternVL2-8B. As shown in Table 7, InternVL2-40B achieves effective AUC performance, slightly exceeding that of InternVL2-8B, indicating that VL in VERA enables models of various scales to identify a $\mathbf{Q}$ suitable for their reasoning capabilities. Additionally, we also evaluate the transferability of $\mathbf{Q}$ across different scales and observe an interesting phenomenon: the $\mathbf{Q}$ learned by InternVL2-8B remains effective for InternVL2-40B, but not vice versa. This is likely because the $\mathbf{Q}$ learned by the smaller model is readily interpretable by the larger model, whereas the $\mathbf{Q}$ derived from the larger model is more complex in syntactic structure and does not align well with the reasoning framework of the smaller model.

***Table 7.*** *AUC (%) across model sizes (rows = $f_{\text{VLM}}$; columns = Source of $\mathbf{Q}$).*

| $f_{\text{VLM}}$ \ Source of Q | InternVL2-8B | InternVL2-40B |
|---|---|---|
| InternVL2-8B | **86.55** | 80.43 |
| InternVL2-40B | 85.24 | **86.72** |

Secondly, we select a different VLM, Qwen2-VL-7B [40], as the backbone for VERA. As shown in Table 8, while the AUC achieved with Qwen2-VL-7B is lower than that with InternVL2-8B, the VL in VERA remains effective, allowing it to outperform notable baselines such as LAVAD [52]. However, a notable gap exists when transferring $\mathbf{Q}$ across different model architectures in Table 8. Developing a universal $\mathbf{Q}$ that can effectively elicit reasoning capabilities across various VLM structures would be a promising direction for future research.

***Table 8.*** *AUC (%) across architectures (rows = $f_{\text{VLM}}$; columns = Source of $\mathbf{Q}$).*

| $f_{\text{VLM}}$ \ Source of Q | InternVL2-8B | Qwen2-VL-7B |
|---|---|---|
| InternVL2-8B | **86.55** | 81.37 |
| Qwen2-VL-7B | 79.60 | **82.64** |

Lastly, we observe that the transferability of $\mathbf{Q}$ depends on the training dataset. From Table 9, we observe that transferring $\mathbf{Q}$ learned from UCF-Crime to XD-Violence results in a smaller performance drop compared to the reverse case. This suggests the source dataset is crucial to the transferability of $\mathbf{Q}$ across datasets.

***Table 9.*** *AUC (%) across datasets (rows = Dataset evaluated; columns = Source of $\mathbf{Q}$).*

| Dataset \ Source of Q | UCF-Crime | XD-Violence |
|---|---|---|
| UCF-Crime | **86.55** | 80.42 |
| XD-Violence | 86.26 | **88.26** |

### 4.4. Qualitative Results and Case Studies

W.l.o.g., we take one video on UCF-Crime to illustrate the explainability brought by the learned $\mathbf{Q}^*$ qualitatively (on UCF-Crime $\mathbf{Q}^*$ is *"1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior? 2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function? 4. Is there any visible damage or unusual movement in the video that indicates an anomaly? 5. Are there any unusual sounds or noises in the video that suggest an anomaly?"*).

As shown in Fig. 5, the main anomaly in this video is that a man tries to steal money from the washing machines in a laundromat and is arrested after being found by the police. In the selected 6 main video segments, the frozen VLM with VERA's learned questions is able to explain the scene by closely following the detailed anomaly characterization of the five learned guiding questions. W.l.o.g., we take the first 3 segments in Fig. 5 for instance to closely compare the caption quality with LAVAD, a representative baseline. As shown in Fig. 6, VERA's captions include both precise descriptions (bold text) and reasoning (text in purple) about anomalies, while LAVAD's captions contain only plain descriptions. This difference owes to VERA's learned guiding questions, which transform VLM's thinking and phrasing.

![Figure 5](figs/fig5.png)

***Figure 5.*** *Given $\mathbf{Q}^*$ by VERA, the frozen VLM (InternVL2-8B) will reason and explain the scene based on it. For illustration, we take as an example the video "Arrest007_x264" from UCF-Crime and include 6 scenes here. The complete anomaly scores are shown in Fig. 8.*

![Figure 6](figs/fig6.png)

***Figure 6.*** *Qualitative comparison between VERA and LAVAD.*

A more interesting advantage of VERA is that it allows humans to further interact with VLMs because it retains the general question-answering ability of pretrained VLMs. This is because VERA does not require finetuning of the VLM backbone weights. Although finetuning VLMs with parameter-efficient methods like [14, 24, 29] is easy and computationally tractable, instruction-tuned models still inevitably lose the flexibility to handle general questions (due to catastrophic forgetting), as they are trained to respond to certain queries with fixed answer styles. In contrast, as shown in Fig. 7, the learned $\mathbf{Q}^*$ can steer reasoning in a frozen VLM while allowing it to flexibly answer open-ended (like follow-up or counterfactual) questions, which is an important ability lost in IT-based models.

![Figure 7](figs/fig7.png)

***Figure 7.*** *VERA can take open-ended questions and interact with humans. (Example: for segment ① of the laundromat video, VERA concludes there is no anomaly; for a follow-up question "This is a laundromat with washing machines. Why is this behavior normal?" it explains the actions are consistent with a laundromat setting; for a counterfactual question "what if the man is stealing from the machine?" it infers that stealing would be an anomaly because it is illegal and inconsistent with typical laundromat behavior.)*

Moreover, as shown in Fig. 8, owing to the proposed coarse-to-fine anomaly scoring, the anomaly score dynamics from VERA well represent the actual real-time anomaly level in this video and gradually increases to nearly 1 when the man is being arrested. This result verifies that VERA allows VLMs to effectively identify anomalies with a holistic model, reducing the manpower and computational overhead for VAD.

![Figure 8](figs/fig8.png)

***Figure 8.*** *Anomaly scores generated by VERA (with InternVL2-8B) in "Arrest007_x264" from UCF-Crime. (The score rises toward 1.0 during the part where the suspect is taken away by the police.)*

---

## 5. Concluding Remarks

We propose a novel pipeline, VERA, which can effectively elicit the reasoning ability from VLMs to perform explainable VAD without additional computation overhead. This is done through an effective and novel application of verbalized machine learning [45] to VLM. In training, VERA obtains the guiding questions detailing anomaly patterns through the verbal interaction between the learner and the optimizer agents. In inference, VERA uses them to enhance VLMs for identifying anomalies and compute frame-level anomaly scores in a coarse-to-fine process. Experimental results validate the effectiveness of the VERA framework in achieving state-of-the-art explainable VAD performance.

---

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. **Gpt-4 technical report.** arXiv preprint arXiv:2303.08774, 2023.

[2] Daniel Bogdoll, Maximilian Nitsche, and J Marius Zöllner. **Anomaly detection in autonomous driving: A survey.** In CVPR Workshops, 2022.

[3] Meinardus Boris, Batra Anil, Rohrbach Anna, and Rohrbach Marcus. **The surprising effectiveness of multimodal large language models for video moment retrieval.** arXiv preprint arXiv:2406.18113, 2024.

[4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. **Language models are few-shot learners.** In NeurIPS, 2020.

[5] Junxi Chen, Liang Li, Li Su, Zheng-jun Zha, and Qingming Huang. **Prompt-enhanced multiple instance learning for weakly supervised video anomaly detection.** In CVPR, 2024.

[6] Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton Fok, Xiaojuan Qi, and Yik-Chung Wu. **Mgfn: Magnitude-contrastive glance-and-focus network for weakly-supervised video anomaly detection.** In AAAI, 2023.

[7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. **Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks.** In CVPR, 2024.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. **An image is worth 16x16 words: Transformers for image recognition at scale.** In ICLR, 2021.

[9] Jia-Chang Feng, Fa-Ting Hong, and Wei-Shi Zheng. **Mist: Multiple instance self-training framework for video anomaly detection.** In CVPR, 2021.

[10] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. **Imagebind: One embedding space to bind them all.** In CVPR, 2023.

[11] Rafael C Gonzalez. **Digital image processing.** Pearson education india, 2009.

[12] Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. **Regiongpt: Towards region understanding vision language model.** In CVPR, 2024.

[13] Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. **Learning temporal regularity in video sequences.** In CVPR, 2016.

[14] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. **Lora: Low-rank adaptation of large language models.** In ICLR, 2021.

[15] Hyekang Kevin Joo, Khoa Vo, Kashu Yamazaki, and Ngan Le. **Clip-tsa: Clip-assisted temporal self-attention for weakly-supervised video anomaly detection.** In ICIP, 2023.

[16] Diederik P Kingma. **Adam: A method for stochastic optimization.** arXiv preprint arXiv:1412.6980, 2014.

[17] Guoqiu Li, Guanxiong Cai, Xingyu Zeng, and Rui Zhao. **Scale-aware spatio-temporal relation learning for video anomaly detection.** In ECCV, 2022.

[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. **Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.** In ICML, 2023.

[19] Shuo Li, Fang Liu, and Licheng Jiao. **Self-training multi-sequence learning with transformer for weakly supervised video anomaly detection.** In AAAI, 2022.

[20] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. **Improved baselines with visual instruction tuning.** In CVPR, 2024.

[21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. **Visual instruction tuning.** In NeurIPS, 2024.

[22] Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. **Deep hyperspherical learning.** In NeurIPS, 2017.

[23] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. **Future frame prediction for anomaly detection–a new baseline.** In CVPR, 2018.

[24] Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, et al. **Parameter-efficient orthogonal finetuning via butterfly factorization.** In ICLR, 2024.

[25] Cewu Lu, Jianping Shi, and Jiaya Jia. **Abnormal event detection at 150 fps in matlab.** In ICCV, 2013.

[26] Hui Lv and Qianru Sun. **Video anomaly detection and explanation via large language models.** arXiv preprint arXiv:2401.05702, 2024.

[27] Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, and Hanwang Zhang. **Unbiased multiple instance learning for weakly supervised video anomaly detection.** In CVPR, 2023.

[28] Sarah Pratt, Ian Covert, Rosanne Liu, and Ali Farhadi. **What does a platypus look like? generating customized prompts for zero-shot image classification.** In ICCV, 2023.

[29] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Schölkopf. **Controlling text-to-image diffusion by orthogonal finetuning.** In NeurIPS, 2023.

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. **Learning transferable visual models from natural language supervision.** In ICML, 2021.

[31] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. **Towards total recall in industrial anomaly detection.** In CVPR, 2022.

[32] Waqas Sultani, Chen Chen, and Mubarak Shah. **Real-world anomaly detection in surveillance videos.** In CVPR, 2018.

[33] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and YingCong Chen. **Hawk: Learning to understand open-world video anomalies.** arXiv preprint arXiv:2405.16886, 2024.

[34] Kamalakar Vijay Thakare, Debi Prosad Dogra, Heeseung Choi, Haksub Kim, and Ig-Jae Kim. **Rareanom: A benchmark video dataset for rare type anomalies.** Pattern Recognition, 140:109567, 2023.

[35] Kamalakar Vijay Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, and Ig-Jae Kim. **Dyannet: A scene dynamicity guided self-trained video anomaly detection network.** In WACV, 2023.

[36] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro. **Weakly-supervised video anomaly detection with robust temporal feature magnitude learning.** In ICCV, 2021.

[37] Anil Osman Tur, Nicola Dall'Asen, Cigdem Beyan, and Elisa Ricci. **Unsupervised video anomaly detection with diffusion models conditioned on compact motion representations.** In International Conference on Image Analysis and Processing, 2023.

[38] Jue Wang and Anoop Cherian. **Gods: Generalized one-class discriminative subspaces for anomaly detection.** In ICCV, 2019.

[39] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. **Temporal segment networks: Towards good practices for deep action recognition.** In ECCV, 2016.

[40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. **Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution.** arXiv preprint arXiv:2409.12191, 2024.

[41] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. **Self-supervised sparse representation for video anomaly detection.** In ECCV, 2022.

[42] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. **Not only look, but also listen: Learning multimodal violence detection under weak supervision.** In ECCV, 2020.

[43] Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, and Yanning Zhang. **Open-vocabulary video anomaly detection.** In CVPR, pages 18297–18307, 2024.

[44] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. **Vadclip: Adapting vision-language models for weakly supervised video anomaly detection.** In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6074–6082, 2024.

[45] Tim Z Xiao, Robert Bamler, Bernhard Schölkopf, and Weiyang Liu. **Verbalized machine learning: Revisiting machine learning with language models.** arXiv preprint arXiv:2406.04344, 2024.

[46] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. **Follow the rules: reasoning for video anomaly detection with large language models.** arXiv preprint arXiv:2407.10299, 2024.

[47] Zhiwei Yang, Jing Liu, and Peng Wu. **Text prompt with normality guidance for weakly supervised video anomaly detection.** In CVPR, 2024.

[48] Muchao Ye, Xiaojiang Peng, Weihao Gan, Wei Wu, and Yu Qiao. **Anopcn: Video anomaly detection via deep predictive coding network.** In ACM international conference on multimedia, 2019.

[49] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. **Textgrad: Automatic "differentiation" via text.** arXiv preprint arXiv:2406.07496, 2024.

[50] Muhammad Zaigham Zaheer, Arif Mahmood, Marcella Astrid, and Seung-Ik Lee. **Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection.** In ECCV, 2020.

[51] M Zaigham Zaheer, Arif Mahmood, M Haris Khan, Mattia Segu, Fisher Yu, and Seung-Ik Lee. **Generative cooperative learning for unsupervised video anomaly detection.** In CVPR, 2022.

[52] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. **Harnessing large language models for training-free video anomaly detection.** In CVPR, 2024.

[53] Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, and Ming-Hsuan Yang. **Exploiting completeness and uncertainty of pseudo labels for weakly supervised video anomaly detection.** In CVPR, 2023.

[54] Hang Zhang, Xin Li, and Lidong Bing. **Video-llama: An instruction-tuned audio-visual language model for video understanding.** In EMNLP, 2023.

[55] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, and Nong Sang. **Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm.** arXiv preprint arXiv:2406.12235, 2024.

[56] Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, and Jianxin Liao. **Multi-scale video anomaly detection by multi-grained spatio-temporal representation learning.** In CVPR, 2024.

[57] Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H Li, and Ge Li. **Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection.** In CVPR, 2019.

[58] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. **Minigpt-4: Enhancing vision-language understanding with advanced large language models.** In ICLR, 2024.

---

# Appendix

**Table of Contents**

- **A. Training in VERA**
  - A.1. Algorithm
  - A.2. Details for Iterative Update by the Optimizer
- **B. Additional Experiments and Results**
  - B.1. Comparison to the State-of-the-art Methods on XD-Violence Measured by AP
  - B.2. Discussion on Generalizability of Used Questions
  - B.3. Hyperparameters in Training
  - B.4. Hyperparameters in Inference and Sensitivity Test
  - B.5. Additional Qualitative Results & Case Studies
- **C. Further Discussion on Limitations**

In this supplementary material, we first include more details on training in VERA (Sec. A) and additional experimental results (Sec. B). To specify:

- In Sec. A, we provide the pseudocodes and details on the initialization, the learner prompt template, and the optimizer prompt template for the training process in Sec. A.1. After that, we discuss the optimization process of the learned questions by the optimizer in Sec. A.2.
- In Sec. B, we first include comparison results with the state-of-the-art methods on XD-Violence measured by AP in Sec. B.1. We also discuss other good properties of VERA, including the good generalizability of the learned questions for different scenarios and the insensitivity of VERA regarding hyperparameters in Sec. B.2 and Sec. B.4, respectively. Finally, we include additional case studies with normal and abnormal videos in Sec. B.5.

We also include a further discussion on the limitations of VERA for future research exploration in Sec. C.

---

## A. Training in VERA

### A.1. Algorithm

We show the complete iterative training process of VERA in pseudocodes in Algorithm 1. It is an iterative process of using the learner to output binary prediction for each sample in a mini-batch and asking the optimizer to update the guiding questions after collecting the batched data. Meanwhile, we have a small validation set (10% samples randomly drawn from the original training set) for deciding the $\mathbf{Q}^*$ used for testing. We want to further detail on certain elements in Algorithm 1 as follows.

> **Algorithm 1: Optimizing Guiding Questions in VAD by VERA during Training**
>
> **Inputs:** Training data pairs $\mathcal{D}_{\text{train}} = \{(\tilde{V}^{(j)}, Y^{(j)})\}_{j=1}^{N}$, iteration number $P$, initial guiding questions $\mathbf{Q}_0$, learner $f_{\text{learner}}$, optimizer $f_{\text{opt}}$, learner prompt template $\theta$, optimizer prompt template $\psi$, validation set $\mathcal{D}_{\text{val}} = \{(\tilde{V}^{(j)}_{\text{val}}, Y^{(j)}_{\text{val}})\}_{j=1}^{\eta}$, period for validation $\mu$, batch size $n$.
>
> **Output:** Optimal guiding questions $\mathbf{Q}^*$.
>
> 1. Set iteration counter $t \leftarrow 1$;
> 2. Set $\mathbf{Q}^* \leftarrow \mathbf{Q}_0$, test $\mathbf{Q}_0$ on validation set $\mathcal{D}_{\text{val}}$ and compute its validation accuracy as $\text{Acc}^*$;
> 3. **while** $t \le P$ **do**
> 4. &nbsp;&nbsp;&nbsp;&nbsp;*# Conduct the learning task with a mini-batch by the learner*
> 5. &nbsp;&nbsp;&nbsp;&nbsp;Randomly sample a batch without repetition from $\mathcal{D}_{\text{train}}$ with a visual input batch $V_{\text{batch}} = [\tilde{V}^{(1)}_{\text{batch}}, \dots, \tilde{V}^{(n)}_{\text{batch}}]$ and ground truths $Y_{\text{batch}} = [Y^{(1)}_{\text{batch}}, \dots, Y^{(n)}_{\text{batch}}]$;
> 6. &nbsp;&nbsp;&nbsp;&nbsp;**for** $1 \le j \le n$ **do**
> 7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Obtain a prediction $\hat{Y}^{(j)}_{\text{batch}}$ for $\tilde{V}^{(j)}_{\text{batch}}$ from $f_{\text{learner}}$ with prompt $(\theta, \mathbf{Q}_t)$ by Eq. (1) as $\hat{Y}^{(j)}_{\text{batch}} = f_{\text{learner}}^{(t)}(\tilde{V}^{(j)}_{\text{batch}})$;
> 8. &nbsp;&nbsp;&nbsp;&nbsp;**end**
> 9. &nbsp;&nbsp;&nbsp;&nbsp;*# Update the guiding questions with the batched data by the optimizer*
> 10. &nbsp;&nbsp;&nbsp;&nbsp;Input the batched prediction $\hat{Y}_{\text{batch}} = [\hat{Y}^{(1)}_{\text{batch}}, \dots, \hat{Y}^{(n)}_{\text{batch}}]$ with $V_{\text{batch}}$ and $Y_{\text{batch}}$ into the optimizer for obtaining a new set of guiding questions by Eq. (2) as: $\mathbf{Q}_{t+1} = f_{\text{opt}}^{(t)}(V_{\text{batch}}, \hat{Y}_{\text{batch}}, Y_{\text{batch}})$;
> 11. &nbsp;&nbsp;&nbsp;&nbsp;*# Compute the validation accuracy with the learned guiding questions periodically*
> 12. &nbsp;&nbsp;&nbsp;&nbsp;$t \leftarrow t + 1$;
> 13. &nbsp;&nbsp;&nbsp;&nbsp;**if** $t \bmod \mu = 0$ **then**
> 14. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test $\mathbf{Q}_t$ on the validation set $\mathcal{D}_{\text{val}}$ and compute the validation accuracy $\text{Acc}_t$;
> 15. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\text{Acc}_t > \text{Acc}^*$ **then**
> 16. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\mathbf{Q}^* \leftarrow \mathbf{Q}_t$;
> 17. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\text{Acc}^* \leftarrow \text{Acc}_t$;
> 18. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**
> 19. &nbsp;&nbsp;&nbsp;&nbsp;**end**
> 20. **end**
> 21. **Return** $\mathbf{Q}^*$;

**Initial $\mathbf{Q}_0$.** The initial guiding questions $\mathbf{Q}_0$ are *"1. Is there any suspicious person or object that looks unusual in this scene? 2. Is there any behavior that looks unusual in this scene?"*. These two questions are manually written and inspired by previous VAD methods, which assume anomaly as something or somebody with unusual appearance or motions [13, 43]. This set of questions is also the "manually written questions by human" in Table 5, which is suboptimal in guiding frozen VLMs to detect anomalies. The key idea of training is to use VL to iteratively update $\mathbf{Q}$ given a suboptimal $\mathbf{Q}_0$.

**Learner Prompt Template $\theta$.** We detail the design of $\theta$ as follows. As shown in Fig. 2, the learner prompt template $\theta$ includes four sections, *i.e.*, Model Description, Prompt Questions, Input, and Output Formatting. To specify:

- **Model Description:** This section introduces the learning task, providing the learner with the necessary background knowledge to understand the objective. It clarifies what the learner is expected to predict based on the given visual input data.
- **Prompt Questions:** This section presents a general prompt to guide the learner's reasoning process. Specific prompts, denoted as $\mathbf{Q}_t$, will be inserted here to facilitate reasoning within a frozen VLM.
- **Input:** This section simply stores the visual tokens. When the VLM reads this, it will correlate the read text with the visual inputs.
- **Output Formatting:** The last section in $\theta$ mainly provides information on output formats to ensure that VLMs think through the given questions $\mathbf{Q}_t$ and output a prediction in a format easy for post-processing in computers.

**Optimizer Prompt Template $\psi$.** As shown in Fig. 2, the optimizer prompt template includes seven sections, *i.e.*, Instruction, Inputs, Model Description, Current Prompt Questions, Model Predictions & Targets, and Optimization Instruction:

- **Instruction:** The prompt template begins with an introduction outlining the responsibilities of the optimizer, clearly stating that its primary task is to optimize the guiding questions provided.
- **Inputs:** This section is used to attach the batched visual data for the reference of the optimizer.
- **Model Description:** The learning task of the learner is reiterated here for the information of the optimizer.
- **Current Prompt Questions:** The guiding questions used by the learner in the current iteration are shown here for the reference of the optimizer.
- **Model Predictions & Targets:** The batched numerical predictions and the ground truths are shown here for $f_{\text{opt}}$. These two inputs can tell the optimizer how well the learner does in the learning task on the mini-batch data.
- **Optimization Instruction:** The final section includes the instruction to ask the optimizer to think step by step with all the information above and output a new set of prompt questions with the required format.

![Figure 10](figs/fig10.png)

***Figure 10.*** *We take the guiding questions $\mathbf{Q}$ learned from the 100th iteration to the 700th iteration for illustration purpose. During the updating process, the optimizer gradually concretizes anomaly patterns that can be applied to different scenarios in a concise expression.*

The figure shows the evolution of the learned question sets across iterations:

**Learned Questions after 100 Iterations**
1. Are there any vehicles in the video frames that are not typically found on a street, such as a motorcycle in a store or a backpack in a parking lot?
2. Are there any people in the video frames who are interacting with objects in a manner that is not typical for a street setting, such as handling a large object in a store or carrying a bag in a parking lot?
3. Are there any people in the video frames who are standing or sitting in a manner that seems suspicious or out of place, such as sitting on the ground in a store or standing in a parking lot?
4. Are there any people in the video frames who are wearing clothing that is not typical for a street setting, such as wearing a uniform in a parking lot or wearing protective gear in a store?
5. Are there any people in the video frames who are in unusual positions or postures, such as sitting on the ground or standing in a way that is not typical for a store setting?

**Learned Questions after 300 Iterations**
1. Are there any people in the video frame who are not engaging in typical activities, such as loitering or not interacting with the surroundings?
2. Are there any objects in the video frame that are not consistent with the typical environment, such as unusual equipment or structures?
3. Are there any sudden or unusual movements in the video frame that are not typical for the environment, such as equipment malfunctions or unexpected actions?
4. Are there any people in the video frame who are not following the normal flow of activities, such as standing in unusual places or engaging in suspicious behaviors?
5. Are there any objects in the video frame that are not interacting with the environment in a typical manner, such as being placed in unusual locations or not being used for their intended purposes?

**Learned Questions after 500 Iterations**
1. Are there any individuals in the video frames who are carrying unusual or unexpected items?
2. Are there any objects in the video frames that are not typically found in that environment and are being used in an unusual or unexpected way?
3. Are there any individuals in the video frames who are interacting with objects in an unusual or unexpected way?
4. Are there any individuals in the video frames who are performing actions that are not typical for that environment?
5. Are there any unusual or unexpected activities occurring in the video frames, such as interactions between objects or individuals that are not typical for that environment?

**Learned Questions after 700 Iterations**
1. Are there any frames where the presence of a specific object (e.g., unusual item, unexpected tool) is not typical for the scene?
2. Are there any frames where the arrangement of people or objects is unusual or unsafe?
3. Are there any frames where an individual is carrying an object that is not typical for the scene?
4. Are there any frames where an individual is interacting with an object in an unusual manner?
5. Are there any frames where the overall environment or setting is not consistent with normal conditions?

### A.2. Details for Iterative Update by the Optimizer

In training, we assess the quality of the learned guiding questions by the accuracy of the validation set. We show the validation accuracy from different questions $\mathbf{Q}_t$ obtained every 100 iterations (mini-batches) in Fig. 9. In the duration of up to 5000 iterations in training, the observed plot in Fig. 9 contains three oscillations, each consisting of an increase in validation accuracy followed by a decrease. The increase represents that the optimizer VLM gradually finds better questions for the binary classification learning task when it sees more batched data, which shows the optimizer can understand its responsibility well and find better questions effectively. Meanwhile, we note that verbal optimization may not always lead to an increase. This is probably because the optimization is completely verbalized, and the VLM will have an inertial thinking behavior like humans, which gets the optimizer stuck in the wrong direction and makes it continue the optimization in a direction that is not beneficial. As a result, this causes the validation accuracy to decrease sometimes. Despite that, because of the guidance provided by the optimizer prompt template $\psi$, the optimizer can overcome its pitfalls in thinking and find good guiding questions in a new direction, which leads to an increase in validation accuracy afterward. This is an interesting phenomenon due to the distinction between verbal learning and traditional numerical optimization algorithms, and it will be a promising future direction to reduce the time in overcoming pitfalls in thinking for VLMs during VL.

![Figure 9](figs/fig9.png)

***Figure 9.*** *The validation accuracy given different learned guiding questions from each iteration. The graph is smoothed with moving average (window size 5) for better readability.*

In addition, w.l.o.g., we take learned questions from the 100th iteration to the 700th iteration (which are within the first epoch) for illustration to show the process of updating $\mathbf{Q}$ by the optimizer in Fig. 10. First, as the optimizer sees more videos, it tries to make the questions focus on a more general setting. For example, the questions in the 100th iteration focus on "street" and "store" scenes. After more iterations, the questions become more generalizable for a general environment and focus on the elements that cause anomalies. Additionally, the anomalous pattern descriptions become more diverse as the optimization continues. To illustrate, in the beginning, the questions mostly pay attention to the humans, objects, and their interaction. In later iterations, the optimizers gradually summarize some previous questions into one and raise questions considering the overall environment ($\mathbf{Q}_5$ from the 700th iteration). Therefore, the VL framework proposed in this paper is effective in finding a diverse set of guiding questions for VAD that apply to general cases, which can elicit the reasoning of a frozen VLM in VAD.

---

## B. Additional Experiments and Results

### B.1. Comparison to the State-of-the-art Methods on XD-Violence Measured by AP

The comparison results regarding average precision (AP), *i.e.*, the area under the frame-level precision-recall curve, on XD-Violence are shown in Table 10. Compared to AUC, AP focuses on measuring the ability to identify the positive class (anomaly), while AUC measures how well a method separates anomaly and normalcy in general. We provide the analysis of the results as follows.

***Table 10.*** *AP (%) on XD-Violence. † indicates VAD methods are trained on entire training frames. No IT is used for Holmes-VAD.*

| Method | AP |
|---|---|
| **Non-Explainable VAD Methods** | |
| Wu et al.† [42] | 78.64 |
| OVVAD [43] | 66.53 |
| S3R† [41] | 80.26 |
| RTFM† [36] | 77.81 |
| MSL† [19] | 78.58 |
| MGFN† [6] | 80.11 |
| CLIP-TSA† [15] | 82.19 |
| **Explainable VAD Methods** | |
| Holmes-VAD† [55] | 84.96 |
| LAVAD [52] | 62.01 |
| ZS CLIP [52] | 17.83 |
| ZS IMAGEBIND-I [52] | 27.25 |
| ZS IMAGEBIND-V [52] | 25.36 |
| LLAVA-1.5 [20] | 50.26 |
| **VERA** | **70.54** |

Firstly, under such a distinct property of AP, as pointed out by [43], methods trained on the whole training set and utilizing all frames will enjoy advantages when measuring VAD performance by AP. As a result, CLIP-TSA and Holmes-VAD, two methods using the whole training frames, attain the highest AP in the category of non-explainable and explainable VAD, respectively. We acknowledge there is a gap between VERA and these two methods under AP on XD-Violence, which is understandable because they use the whole training frames to improve the ability to find anomalies of classifiers. To illustrate, in training VERA only samples 8 frames for each video and only uses 0.19% total frames (31,632 out of 16,378,527) for training on XD-Violence. Thus, our training is dramatically light compared to the methods like CLIP-TSA and Holmes-VAD in Table 10. With fewer frames used for training, VERA unavoidably achieve lower AP (which only considers positive cases) compared to those that have more, for it relies on fewer training data. In addition, we want to point out that judging the VAD performance solely by AP on XD-Violence can be biased. This is because the ratio of positive frames in XD-Violence (23.07%) in test videos is overly higher than other datasets like UCF-Crime (7.92%), which is unrealistic because the anomaly is sparse in the real world [32]. Given that, only focusing on the comparison in AP on XD-Violence would amplify the bias in VAD performance evaluation, and we recommend taking into consideration other factors like training costs and the comprehensive ability of distinguishing anomaly and normality by the methods in evaluation.

Secondly, among the methods (OVVAD, LAVAD, ZS CLIP, ZS IMAGEBIND, and LLAVA-1.5) that does not use full frames for training, VERA achieves the best AP in this fair comparison, surpassing the second best method in the Explainable VAD category (LAVAD) over 8.53%, which showcases the effectiveness of using learned guiding question to prompt frozen VLMs for VAD.

To conclude, it is unfair to only judge VAD performance by AP on XD-Violence without considering the training costs and the relatively imbalanced frame distribution in test videos. Considering all factors into consideration, VERA is a favorable method used for VAD in detecting anomalies.

### B.2. Discussion on Generalizability of Used Questions

During the optimization of $\mathbf{Q}$, because of the randomness involved in this process, the optimizer may output certain guiding questions that only focus on one specific surrounding. We find an interesting phenomenon on VLMs in VAD that guiding questions related to a specific scenario yield inferior VAD performance compared to the general questions in both general cases and specific cases.

To illustrate, we take two sets of specific questions obtained on UCF-Crime for analysis. The first example is a set of guiding questions $\mathbf{Q}_{\text{traffic}}$ that only ask the VLM to consider anomalies related to the traffic as follows:

1. Are there any vehicles or people violating traffic rules?
2. Are there any accidents or near-accidents occurring?
3. Are there any objects or people obstructing the normal flow of traffic?
4. Are there any unusual or unexpected behaviors from pedestrians or drivers?
5. Are there any emergency vehicles or personnel present?

The second example is another set of guiding questions $\mathbf{Q}_{\text{store}}$ that only ask the VLM to identify anomalies in a store setting, which includes questions like:

1. Are there any individuals loitering or behaving suspiciously inside the store?
2. Is there any unusual activity inside the store, such as tampering with items or attempting to enter restricted areas?
3. Are there any signs of forced entry or damage to the store's entrance?
4. Are there any individuals present who seem to be watching or waiting for something specific inside the store?
5. Are there any interactions between individuals inside the store that appear suspicious or out of the ordinary?

Thus, $\mathbf{Q}_{\text{traffic}}$ and $\mathbf{Q}_{\text{store}}$ focuses on the specific anomalies of traffic accidents and shoplifting, respectively, while the $\mathbf{Q}^*$ that we find focuses on general cases and includes the following questions:

1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?
2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?
3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?
4. Is there any visible damage or unusual movement in the video that indicates an anomaly?
5. Are there any unusual sounds or noises in the video that suggest an anomaly?

The comparison results of $\mathbf{Q}^*$, $\mathbf{Q}_{\text{traffic}}$, and $\mathbf{Q}_{\text{store}}$ in detecting anomalies in general cases (all testing videos on UCF-Crime), traffic scenes (testing videos from the Traffic Accident category on UCF-Crime), and the store scenes (testing videos from the Shoplifting category on UCF-Crime) are shown in Table 11. It indicates that $\mathbf{Q}^*$ performs the best in both general cases and two specific cases like in traffic and store scenes. This is because the overly specific definition of anomalies like $\mathbf{Q}_{\text{traffic}}$ and $\mathbf{Q}_{\text{store}}$ makes it harder for a VLM to classify one clip into an anomaly and leads to more false negatives in its prediction given those specific questions, which degrades the performance. Therefore, we recommend using general questions like the ones shown in $\mathbf{Q}^*$ in frozen VLMs for VAD.

***Table 11.*** *General guiding questions outperform specific ones measured by AUC (%) on UCF-Crime. Specific questions are not tested on other specific scenarios, which is indicated by a slash (/).*

| Questions | All | Traffic | Store |
|---|---|---|---|
| $\mathbf{Q}^*$ | **86.55** | **70.43** | **72.58** |
| $\mathbf{Q}_{\text{traffic}}$ | 82.59 | 67.53 | / |
| $\mathbf{Q}_{\text{store}}$ | 76.67 | / | 44.84 |

### B.3. Hyperparameters in Training

**Batch Size and Sampled Frame Number.** Key hyperparameters that need to be set in training are the batch size $n$ and the number of sampled frames $S$ for each video $V^{(j)}$ in the VL framework. They are correlated because they determine the total number of frames for the optimizer to skim and provide feedback as $S \cdot n$. Considering memory constraints when implementing VLMs on GPUs, we set $S \cdot n = 16$ in training. We further explore the trade-off between $S$ and $n$ given the constraints for input frames to decide $S$ and $n$. The results are shown in Table 12. If the batch size $n$ is 1 with $S = 16$, the learned questions cannot be generalized due to the limited video sample in the batch which leads to a suboptimal AUC, and it takes longer to train for VERA. Meanwhile, if we set $n$ as large numbers like 4 or 8 (with $S = 4$ or $S = 2$), the learned questions are suboptimal too because relatively few sampled frames generally lack the temporality for the optimizer to look into the details and conceive good questions. Thus, setting $n$ to 2 and $S$ to 8 is in default in this paper, which strikes the balance between training efficiency and effectiveness.

***Table 12.*** *The choice of batch size and sampling frames affects the effectiveness of the learned guiding questions in VAD. The results are obtained by InternVL2-8B as VERA's backbone.*

| Batch Size | Sampled Frames | AUC (%) |
|---|---|---|
| $n = 1$ | $S = 16$ | 81.53 |
| $n = 2$ | $S = 8$ | **86.55** |
| $n = 4$ | $S = 4$ | 83.19 |
| $n = 8$ | $S = 2$ | 79.91 |

### B.4. Hyperparameters in Inference and Sensitivity Test

**Hyperparameters in Inference.** During inference, in Step 1, following [52], the interval between each segment center $d$ is 16 frames. In Step 2, we use ImageBind [10] as the feature extractor in computing segment similarity as [52] does, and the number of retrieved segments $K$ depends on the total number of segments $h$ in each test video $V$. Setting $K$ to $(0.1 \cdot h)$ to $(0.15 \cdot h)$ is generally good. We set $K$ to $(0.1 \cdot h)$ for UCF-Crime and to $(0.15 \cdot h)$ for XD-Violence. The temperature $\tau$ in the Softmax function is set to 10 for both datasets in Eq. (4). In Step 3, due to the properties of datasets, we set the filter size $\omega$ of $G(p)$ to 15 and $\sigma_1$ to 10 for UCF-Crime, while setting $\omega$ to 30 and $\sigma_1$ to 30 for XD-Violence. For position weighting, we set $c = \lfloor F/2 \rfloor$ and $\sigma_2 = \lfloor F/2 \rfloor$ for both datasets to make sure the position weight covers the whole video sequence. W.l.o.g., we test the sensitivity of the VAD performance of VERA regarding hyperparameters on UCF-Crime.

**Sensitivity Test for $K$.** As shown in Table 13, as the number of retrieved segments increases from 0 to $0.15 \cdot h$, the AUC gradually increases from 85.21% to 86.61%. Meanwhile, if we randomly select $0.1 \cdot h$ segments for retrieval, the AUC is even lower than the performance without retrieval. Thus, using Eq. (4) for retrieval is necessary. Meanwhile, having a large $K$ greater than $0.15 \cdot h$ will introduce some noise in Eq. (4) and downgrade the AUC slightly. Thus, selecting $0.1 \cdot h$ or $0.15 \cdot h$ for $K$ is generally good choice.

***Table 13.*** *Influence of the number of retrieved segments on AUC. The AUC of not using retrieval (Ratio = 0%) and randomly selecting 10% segments for Eq. (4) is 85.21% and 84.55%, respectively.*

| Ratio (%) | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| AUC (%) | 85.21 | 86.48 | 86.55 | 86.61 | 86.42 | 86.19 |

**Sensitivity Test for $\omega$.** The filter size decides how many local segments are incorporated for the current segment for Gaussian smoothing. From Table 14, we find that AUC converges when the filter size increases to 15. Meanwhile, the VAD performance measured AUC is insensitive to $\omega$ and does not fluctuate much. Thus, we can set the filter size with a medium number like 15.

***Table 14.*** *Influence of filter size $\omega$ in Gaussian Smoothing on AUC.*

| $\omega$ | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| AUC (%) | 86.25 | 86.43 | 86.55 | 86.61 | 86.60 |

**Sensitivity Test for $\sigma_1$.** The AUC performance is also robust on the choice of $\sigma_1$. As shown in Table 15, when we set $\sigma_1$ greater than 1, the AUC generally remains around 86.50%, which again shows the robustness of the design of anomaly scoring in VERA. We can set $\sigma_1$ as 10 for VERA.

***Table 15.*** *Influence of $\sigma_1$ in Gaussian Smoothing on AUC.*

| $\sigma_1$ | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| AUC (%) | 86.17 | 86.49 | 86.55 | 86.49 | 86.54 |

**Sensitivity Test for $\tau$.** The temperature hyperparameter $\tau$ in Eq. (4) controls the entropy of the distribution obtained from the Softmax function while preserving the rank of each element. As demonstrated in Table 16, when $\tau$ is a small number like 10e-8 that is close to 0, the distributions tend to become a trivial distribution with all mass concentrated on the highest-probability class (corresponding to the segment itself), and the result is the same as the one by not using retrieval. As we gradually increase $\tau$ to a reasonably large number (from 0.01 to 1), the AUC value converges around 86.55% with no obvious fluctuation, again proving the robustness of anomaly scoring in VERA regarding hyperparameter selection. Note that when $\tau$ approaches $+\infty$, the distribution tends to become a uniform distribution, which yields an AUC of 86.59%. From the discussion above, we can generally choose $\tau$ to be a number in $[0.01, 1]$ in implementation.

***Table 16.*** *Influence of $\tau$ in Eq. (4) on AUC.*

| $\tau$ | 10e-8 | 0.01 | 0.1 | 1 | $+\infty$ |
|---|---|---|---|---|---|
| AUC (%) | 85.21 | 86.31 | 86.55 | 86.58 | 86.59 |

**Sensitivity Test for $\sigma_2$.** From Table 17, we find that setting $\sigma_2 = 0.5F$ encodes the position information best in the anomaly score. A drop is noticeable if we choose $\sigma_2$ less than $0.5F$ for it will not cover the whole sequence, which is reasonable, while choosing a $\sigma_2$ greater than $0.5F$ does not change much. Thus, based on the physical meaning of $\sigma_2$, which controls the width of the distribution, we should make $\sigma_2$ equal to $0.5F$ in anomaly scoring.

***Table 17.*** *Influence of $\sigma_2$ in Position Weighting on AUC.*

| $\sigma_2$ | w/o Weighting | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|
| AUC (%) | 85.48 | 85.43 | 86.55 | 86.27 |

### B.5. Additional Qualitative Results & Case Studies

W.l.o.g., we take one normal video ("Normal_Videos_018_x264") and another abnormal video ("RoadAccidents127_x264") from the UCF-Crime dataset to demonstrate the explanations provided by a frozen VLM (InternVL2-8B) achieved by using the learned guiding questions $\mathbf{Q}^*$.

First, in Fig. 11 we showcase the explanation of anomaly scoring by VERA regarding a normal video "Normal_Videos_018_x264" in UCF-Crime, which is taken in an airport hallway where no anomaly happens. For this video, VERA assigns a 0 score to each frame. As shown in Fig. 11, for the selected scenes in this video, VERA explains that this is because there are no events that conform to the anomaly descriptions in $\mathbf{Q}^*$. Such explanations are consistent with the recording and again manifest the effectiveness of eliciting the reasoning ability in a frozen VLM for VAD by using learned guiding questions. Note that we do not have an additional figure illustrating the anomaly score dynamic for this video because all scenes are assigned 0 scores by VERA.

![Figure 11](figs/fig11.png)

***Figure 11.*** *Given the normal video "Normal_Videos_018_x264", the frozen VLM (InternVL2-8B) can conclude that no anomaly happens in the video under the guidance of $\mathbf{Q}^*$, which is aligned with the ground truth. Since the anomaly scores for all scenes are zeros by VERA, we do not show the complete anomaly scores with an additional figure.*

Next, we select 6 representative scenes in the abnormal video ("RoadAccidents127_x264") and show the corresponding explanation provided by the frozen VLM in Fig. 12. The main anomaly that happens in this video is a traffic accident where a truck crashes into a train from Frame 2160 to Frame 2299, which corresponds to the 5th scene in Fig. 12. In particular, the figure shows that the learned question "Is there any visible damage or unusual movement in the video that indicates an anomaly?" in $\mathbf{Q}^*$ makes the frozen VLM find a good way to express what it sees in the 5th scene and understand this is an anomaly because the crash is unusual and dangerous. The other scenes are also well explained by the frozen VLM under $\mathbf{Q}^*$. Thus, this again verifies that the learned guiding questions can successfully trigger reasonable explanations in the adopted frozen VLM for VAD.

![Figure 12](figs/fig12.png)

***Figure 12.*** *Given the abnormal video "RoadAccidents127_x264", the frozen VLM (InternVL2-8B) can generate reasonable explanations aligned with the semantic change observed in each scene under the guidance of $\mathbf{Q}^*$. The complete anomaly scores are shown in Fig. 13.*

Meanwhile, we also include the anomaly scores generated by VERA for the abnormal video in Fig. 13. Most frames are assigned to zero except the scenes when someone crosses the road at an unusual speed (the 2nd scene in Fig. 12) and the truck-train crash happens (the 5th scene in Fig. 12). This fluctuation is aligned with the ground truth annotation and common sense about an anomaly, which shows that the anomaly scoring proposed in VERA is reasonable.

![Figure 13](figs/fig13.png)

***Figure 13.*** *Anomaly scores generated by VERA (with InternVL2-8B) in "RoadAccidents127_x264" from UCF-Crime.*

---

## C. Further Discussion on Limitations

Like existing VLM-based VAD methods, VERA's performance relies heavily on the visual perception capabilities of VLMs. Most VLMs employ the CLIP vision encoder [30], which has limitations in capturing fine-grained visual details. This limitation can impair precise anomaly detection. If important visual features are missing during the visual encoding process, then it is unlikely for VERA to perform meaningful VL. Therefore, a fundamental challenge for VLM-based VAD is to ensure sufficient visual and temporal features are encoded. Having verified this capability, VERA can perform VL to extract crucial cues that guide video anomaly reasoning.
