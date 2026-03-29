# VAD Using VLMs — Graduation Project

Visual Anomaly Detection (VAD) using Vision-Language Models (VLMs).
This project leverages multimodal AI APIs (DashScope / Qwen-VL) to detect anomalies in visual inputs.

---

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.10+

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/vad_using_VLMs_graduation_project.git
cd vad_using_VLMs_graduation_project
```

### 2. Create and activate the conda environment

```bash
conda create -n grad2_env python=3.10 -y
conda activate grad2_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Then open `.env` and fill in your API key:

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

> Get your DashScope API key from [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com)

---

## Usage

```bash
conda activate grad2_env
python test_api.py
```

---

## Project Structure

```
vad_using_VLMs_graduation_project/
├── test_api.py          # API connection test with Qwen-VL
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md
```

---

## Notes

- Never commit your `.env` file — it contains private API keys.
- The conda environment is named `grad2_env` throughout this project.
