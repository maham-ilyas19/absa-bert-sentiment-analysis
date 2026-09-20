# Aspect-Based Sentiment Analysis (ABSA) using BERT 🎯

An end-to-end Natural Language Processing (NLP) pipeline for Aspect-Based Sentiment Analysis using PyTorch and Hugging Face Transformers (`bert-base-uncased`).

## 📌 Overview

Standard sentiment analysis classifies an entire review as positive or negative. **Aspect-Based Sentiment Analysis (ABSA)** goes a step further by identifying the sentiment toward a **specific target aspect** within the review (e.g., assessing sentiment for "food" vs. "service" separately in a single restaurant review).

### Key Features
- **Model Architecture:** Pre-trained `bert-base-uncased` fine-tuned for sequence classification.
- **Data Formatting:** Custom sequence pairs joining review and aspect text: `[CLS] review_text [SEP] aspect_term [SEP]`.
- **Pipeline:** Automated label encoding, stratified train/val/test splitting, custom PyTorch Dataset handling, and Hugging Face `Trainer` integration.

## 📊 Dataset & Preprocessing

- **Aspect Categories:** `food`, `service`, `ambiance`, `price`
- **Target Sentiment Labels:** `positive`, `negative`
- **Data Split:**
  - 80% Training Set
  - 10% Validation Set
  - 10% Test Set (Stratified)
    
## 🚀 Performance & Results

The fine-tuned BERT model achieved high classification performance across validation and test evaluations:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Macro |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.7427 | 0.6905 | 52.94% | 36.65% |
| 2 | 0.6794 | 0.6073 | 82.35% | 82.35% |
| 3 | 0.4464 | 0.2642 | 94.12% | 94.12% |
| **4** | **0.2204** | **0.1802** | **94.12%** | **94.12%** |
| 5–8 | ~0.03–0.10 | ~0.24–0.27 | 94.12% | 94.12% |

### Final Evaluation Metrics
- **Test Accuracy:** `100%`
- **Test F1 Macro Score:** `1.0`
- **Test Loss:** `0.2488`

## 🛠 Tech Stack

- **Frameworks:** PyTorch, Hugging Face Transformers (`transformers`, `datasets`)
- **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy
- **Base Model:** `bert-base-uncased`

## 💻 How to Use

### 1. Installation
Clone the repository and install the dependencies:
pip install torch transformers scikit-learn pandas numpy
