# 📡 AT&T Spam Detector — Deep Learning (CDSD · Jedha · Bloc 4)

## Overview  
This project tackles the challenge of **spam email detection** using both a **deep learning baseline** and a **transfer learning model (DistilBERT)**. The goal is to accurately classify emails as *spam* or *ham* while balancing **precision and recall**, ensuring spam is caught without misclassifying too many legitimate messages.  

We start with a simple **Deep Averaging Network (DAN)** baseline, where trainable word embeddings are averaged and fed into a linear classifier. Then, we fine-tune **DistilBERT**, a transformer pretrained on large text corpora, to leverage contextual understanding for higher recall and overall performance.  

🔗 **Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1w2vrb2wgOa8pG22f0ErZ2FdnTpW9jFRb?usp=sharing) 

---

## Methodology  

### Preprocessing  
- **Cleaning:** Removal of punctuation, stopwords, and lowercasing.  
- **Train/Test Split:** Stratified split to preserve spam/ham balance.  
- **Tokenization:** Conversion of words into integer IDs.  
- **Padding:** Sequences padded to `max_length = 44` for uniform input size.  

### PyTorch Dataset  
- Wrapped tokenized inputs and labels into a custom `Dataset` class.  
- Provided batches via `DataLoader` for efficient GPU training.  

### Baseline Model — Deep Averaging Network (DAN)  
- **Embedding layer:** Learns word representations during training.  
- **Pooling:** Average pooling to obtain a fixed-size sentence vector.  
- **Classifier:** Linear layer with sigmoid activation for spam probability.  
- **Loss:** Binary Cross-Entropy (BCE).  
- **Optimizer:** Adam with learning rate 1e-3.  

### Training & Evaluation  
- Trained over multiple epochs with gradient descent (forward → loss → backprop → update).  
- Metrics tracked: **Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC, PR-AUC**.  
- Plots: Training loss, Accuracy/F1 across epochs, ROC & PR curves, Confusion Matrix.  

### Transfer Learning — DistilBERT Fine-Tuning  
- **Model:** DistilBERT encoder + custom linear classifier.  
- **Loss:** `BCEWithLogitsLoss` with class weighting (`pos_weight`) to address class imbalance.  
- **Optimizer:** AdamW (lr=5e-5).  
- **Training:** Fine-tuned for 20 epochs on the spam dataset.  
- **Evaluation:** Same metrics/plots as baseline for direct comparison.  

---

## Results & Insights  

| Metric       | Baseline (DAN) | DistilBERT Transfer Learning |
|--------------|----------------|------------------------------|
| Accuracy     | 0.984          | **0.989** |
| Precision    | **0.983**      | 0.969 |
| Recall       | 0.888          | **0.944** |
| Specificity  | **0.998**      | 0.996 |
| F1-score     | 0.933          | **0.956** |
| ROC-AUC      | 0.977          | **0.996** |
| PR-AUC       | 0.956          | **0.986** |

- The **baseline** is simple, fast, and highly precise but misses ~11% of spam.  
- **DistilBERT** achieves much higher recall, reducing missed spam while keeping false positives very low.  
- Transfer learning clearly outperforms the baseline on **F1, ROC-AUC, and PR-AUC**, showing the value of contextual embeddings.  

---

## Tech Stack  
- **Python:** pandas, numpy, matplotlib, scikit-learn  
- **Deep Learning:** PyTorch  
- **Models:**  
  - Baseline: Embedding + Average Pooling + Linear Classifier  
  - Transfer Learning: DistilBERT + Linear Classifier  
- **Metrics:** Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC  
- **Environment:** Google Colab / Jupyter Notebook  
