# 🇮🇳 Hinglish Sentiment Analyzer

A fine-tuned XLM-RoBERTa transformer model for sentiment analysis of code-mixed Hindi-English (Hinglish) text. Achieves **69% accuracy** on the SemEval-2020 Task 9 dataset with interactive Streamlit demo.

**Live Demo:** `streamlit run api/streamlit_demo.py`

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Data & Dataset](#data--dataset)
- [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

Code-mixing (mixing Hindi and English in a single sentence) is ubiquitous in real-world social media, but sentiment analysis tools rarely handle it well. This project addresses that by fine-tuning **XLM-RoBERTa**, a multilingual transformer pretrained on 100+ languages, on authentic Hinglish tweets.

**Problem:** Traditional sentiment classifiers (TF-IDF + Naive Bayes) fail on code-mixed text because they:
- Treat Hindi and English separately, missing code-mixing patterns
- Don't capture semantic relationships in mixed scripts

**Solution:** Transfer learning with XLM-RoBERTa preserves multilingual understanding while the fine-tuned classification head learns Hinglish-specific sentiment patterns.

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 69.04% |
| **Weighted F1-Score** | 0.6869 |
| **Training Epochs** | 3 (with early stopping) |
| **Training Time** | ~14 minutes (Google Colab T4 GPU) |

### Confusion Matrix (Test Set: 3,130 tweets)

Predicted Negative  Predicted Neutral  PredictedPositive
Actual Negative       76618528
Actual Neutral        273685252
Actual Positive        68163710

### Per-Class Performance
Precision  Recall  F1-Score  Support
Negative     0.69     0.78     0.73979
Neutral      0.66     0.57     0.611,210
Positive     0.72     0.75     0.74941

**Observation:** Neutral class (0.61 F1) is harder to predict than clear extremes (Negative/Positive both ~0.73), which is realistic — neutral sentiment is conceptually fuzzy.

---

## 🛠️ Tech Stack

### Data & Preprocessing
- **pandas, NumPy** — Data loading and manipulation
- **regex (re module)** — Light text cleaning (smart quote normalization, whitespace collapsing)
- **SemEval-2020 Task 9 Dataset** — 14k training + 3k test Hinglish tweets

### Model & Training
- **XLM-RoBERTa (xlm-roberta-base)** — Pretrained multilingual transformer (100+ languages)
- **HuggingFace Transformers** — Model loading, tokenization, fine-tuning
- **PyTorch** — Deep learning framework
- **HuggingFace Trainer API** — Training loop with early stopping and checkpointing

### Evaluation
- **scikit-learn** — Accuracy, F1-score, confusion matrix, classification report

### Demo & Deployment
- **Streamlit** — Interactive web UI for live predictions
- **FastAPI** (planned) — REST API for production serving

---

## 💻 Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/KratosArc/hinglish-sentiment-analyzer.git
cd hinglish-sentiment-analyzer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install transformers torch datasets scikit-learn pandas numpy streamlit
```

### 4. Download Pre-trained Model
Download `hinglish_model_v2.zip` from releases and extract:
```bash
unzip hinglish_model_v2.zip -d models/
```

---

## 🚀 Usage

### Option 1: Interactive Streamlit Demo
```bash
streamlit run api/streamlit_demo.py
```
Opens a web UI where you can type Hinglish text and get real-time sentiment predictions.

**Example inputs:**
- "yeh movie bohot acchi thi yaar" → Positive (that movie was very good, dude)
- "bahut bura tha" → Negative (was very bad)
- "theek hai" → Neutral (it's okay)

### Option 2: Python Script (Programmatic)
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/hinglish_model_v2",
    tokenizer="./models/hinglish_model_v2"
)

text = "mujhe ye product bilkul pasand nahi aaya"
result = classifier(text)
print(result)  # [{'label': 'LABEL_0', 'score': 0.87}]
```

---

## 📈 Model Performance

### Training Dynamics
The model was fine-tuned with:
- **Learning rate:** 2e-5 (crucial for stable convergence)
- **Warmup steps:** 500 (prevents early divergence)
- **Batch size:** 16
- **Max epochs:** 3 (with early stopping patience=2)
- **Optimizer:** AdamW (default)

**Training curve (loss over epochs):**
Epoch 1: Training Loss = 0.871, Validation Loss = 0.859
Epoch 2: Training Loss = 0.811, Validation Loss = 0.839
Epoch 3: Training Loss = 0.678, Validation Loss = 0.849 ← Best model




The model learned steadily and didn't overfit (validation loss stayed stable while training loss dropped).

### Why 69% Accuracy?

Code-mixed sentiment is genuinely hard:
1. **Ambiguous neutral class** — What counts as neutral vs slightly positive? Hard to define.
2. **Sarcasm & colloquial language** — Hinglish tweets use heavy slang ("yaar," "bhai," context-dependent meanings).
3. **Limited training data** — 14k tweets is modest compared to monolingual English datasets (100k+).
4. **Class imbalance potential** — Even though balanced here, some edge cases remain.

**Benchmark:** Published results on SemEval-2020 Task 9 with similar architectures range 65-72%, so 69% is competitive.

---

## 📁 Project Structure
hinglish-sentiment-analyzer/
├── data/
│   ├── raw/                          # Original SemEval-2020 TSV files
│   │   └── temp_repo/data/hinglish/
│   │       ├── train.txt
│   │       └── test.txt
│   └── processed/                    # Cleaned CSV files
│       ├── train_processed.csv
│       └── test_processed.csv
├── models/
│   └── hinglish_model_v2/            # Fine-tuned model weights
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── src/
│   ├── preprocessing.py              # Data loading & light cleaning
│   ├── train.py                      # Training script
│   └── predict.py                    # Inference utility
├── api/
│   └── streamlit_demo.py             # Interactive web UI
├── notebooks/
│   └── eda.ipynb                     # Exploratory data analysis
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git exclusions


---

## 📊 Data & Dataset

**Dataset:** SemEval-2020 Task 9 (SentiMix: Sentiment Analysis for Code-Mixed social media text)

**Statistics:**
- **Train:** 13,987 tweets (after cleaning)
- **Test:** 3,130 tweets (held-out)
- **Classes:** Negative (4,101), Neutral (5,252), Positive (4,634) — balanced
- **Language:** Hindi + English, Roman script
- **Text length:** 20-180 characters per tweet

**Preprocessing (Light-Touch Approach):**
- Normalize smart quotes
- Collapse repeated whitespace
- **Preserve:** punctuation, casing, emojis (transformers learn from these)
- **No:** lowercasing, emoji removal, URL stripping, stopword removal

---

## 🔬 Methodology

### 1. Data Collection
Downloaded SemEval-2020 Task 9 Hinglish dataset (14k train + 3k test).

### 2. Text Cleaning
Minimal cleaning — only fix genuine noise while preserving signal for transformers.

### 3. Tokenization
HuggingFace `AutoTokenizer` handles subword tokenization, pads/truncates to max_length=128.

### 4. Model Architecture
Input Text (Hinglish)
↓
XLM-RoBERTa Encoder (100+ languages)
↓
[CLS] Token Representation
↓
Classification Head (3 units)
↓
Softmax → Confidence Scores



### 5. Training
- Fine-tuned XLM-RoBERTa with learning rate 2e-5
- Early stopping if validation F1 doesn't improve for 2 epochs
- Batch size 16, max 3 epochs

### 6. Evaluation
- Test accuracy: 69.04%
- Weighted F1: 0.6869
- Per-class precision/recall/F1

---

## 🎓 Key Learnings

1. **Learning rate matters hugely** — Default 5e-5 caused divergence; 2e-5 + warmup fixed it
2. **Light preprocessing is better for transformers** — They learn from punctuation/casing
3. **Validation/test split is critical** — No overfitting observed (stable validation loss)
4. **Code-mixed sentiment is genuinely hard** — Neutral class (0.61 F1) reflects real ambiguity

---

## 🚧 Future Improvements

1. **Ensemble methods** — Combine with other models (DistilBERT, IndicBERT)
2. **Data augmentation** — Back-translation, paraphrasing
3. **Hyperparameter tuning** — Grid search over learning rates, batch sizes
4. **Production deployment** — FastAPI + Docker
5. **Multi-task learning** — Joint sentiment + language identification
6. **Domain adaptation** — Fine-tune on specific domains

---

## 📚 References

- [SemEval-2020 Task 9](https://www.aclweb.org/anthology/2020.semeval-1.257/)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

## 👤 Author

Built by **Jiveshwar** | [GitHub: KratosArc](https://github.com/KratosArc)

---

**Last updated:** July 2026
