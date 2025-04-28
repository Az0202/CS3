
# 🎵 Music-Genre Case Study

## 📌 Overview
In this case study, you’ll play the role of a data scientist at a music-streaming company. Your mission is to discover whether song lyrics alone can reveal a track’s genre. You will:

1. **Clean & preprocess** raw lyric data.  
2. Train **traditional ML** models (TF-IDF + Logistic Regression).  
3. Train an **advanced NLP** model (BERT embeddings).  
4. **Evaluate** each approach using confusion matrices, PCA plots, and classification reports.

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/YourUsername/Music-Genre-Case-Study.git
cd Music-Genre-Case-Study
pip install -r requirements.txt
```

### 2. Explore the Case Study Prompt & Rubric
- **Hook Document:** `Hook_Document.pdf`[https://github.com/Az0202/CS3/blob/main/Hook_Document.pdf]
- **Grading Rubric:** `Rubric.pdf`

---

## 📂 Repository Structure
```
.
├─ README.md
├─ LICENSE.md
├─ requirements.txt
│
├─ DATA/
│   ├─ tcc_ceds_music.csv             # Original lyrics dataset
│   ├─ Data Appendix Project 1.pdf    # Schema & source notes
│   ├─ processed_data.csv             # Output of preprocessing
│   ├─ train_data.csv                 # 80% training split
│   └─ test_data.csv                  # 20% testing split
│
├─ SCRIPTS/
│   ├─ preprocess_data.py             # Data cleaning & splitting
│   ├─ train_ml_models.py             # Train TF-IDF + Logistic Regression
│   ├─ train_bert_model.py            # Fine-tune BERT classifier
│   ├─ predict_genre_ml_models.py     # Generate TF-IDF model predictions
│   ├─ predict_genre_bert_model.py    # Generate BERT model predictions
│   ├─ utils.py                       # Shared helper functions
│   └─ visualization_scripts/
│       ├─ visualization_for_ml_models.ipynb
│       └─ visualizations_for_bert_model_performance.ipynb
│
└─ OUTPUT/
    ├─ confusion_matrices/
    │   ├─ Confusion Matrix – Logistic Regression.png
    │   └─ Confusion Matrix – Neural Network.png
    │
    ├─ PCA Visualization of BERT Embeddings.png
    └─ additional_visualizations/      # (Optional deep-dive plots)
        ├─ Distribution of Misclassified Genres.png
        ├─ Model Performance Metrics.png
        ├─ Precision-Recall Curve.png
        ├─ ROC Curve for Genre Classification.png
        └─ True vs Predicted Genre Distribution.png
```

---

## 🔄 Reproducibility

1. **Prepare the data**  
   ```bash
   python SCRIPTS/preprocess_data.py   # cleans and creates processed_data.csv
   # (Optional) re-split:
   python SCRIPTS/train_test_split.py  # regenerates train_data.csv & test_data.csv
   ```

2. **Train your models**  
   ```bash
   python SCRIPTS/train_ml_models.py   # traditional TF-IDF + Logistic Regression
   python SCRIPTS/train_bert_model.py  # BERT fine-tuning
   ```

3. **Make predictions**  
   ```bash
   python SCRIPTS/predict_genre_ml_models.py
   python SCRIPTS/predict_genre_bert_model.py
   ```

4. **View results**  
   - Key confusion matrices: `OUTPUT/confusion_matrices/`  
   - PCA embedding plot: `OUTPUT/PCA Visualization of BERT Embeddings.png`  
   - Full metrics report: `OUTPUT/classification_report.pdf`  
   - For additional analyses (ROC, PR curves, etc.): see `OUTPUT/additional_visualizations/`

---

## 📜 License
This project is licensed under the MIT License. See `LICENSE.md` for details.

---

## 📚 References

1. Tzanetakis, G., & Cook, P. (2002). Musical Genre Classification of Audio Signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293–302. [https://ieeexplore.ieee.org/document/1021073](https://ieeexplore.ieee.org/document/1021073)

2. Pizarro Martinez, S., Zimmermann, M., Offermann, M. S., & Reither, F. (2024). *Exploring Genre and Success Classification through Song Lyrics using DistilBERT: A Fun NLP Venture*. arXiv preprint. [https://arxiv.org/html/2407.21068v1](https://arxiv.org/html/2407.21068v1)

3. GeeksforGeeks. (n.d.). Text Preprocessing for NLP Tasks. [https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/](https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/)

4. Analytics Vidhya. (2021). Text Preprocessing in NLP with Python Codes. [https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/](https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/)

5. Analytics Vidhya. (2021). Metrics to Evaluate Your Classification Model to Take the Right Decisions. [https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/](https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/)

6. Mendeley Data. (2020). Music Lyrics Dataset (1950–2019) for Genre Classification [Dataset]. [https://data.mendeley.com/datasets/3t9vbwxgr5/2](https://data.mendeley.com/datasets/3t9vbwxgr5/2)
```
