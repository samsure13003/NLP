# 📚 Kindle Review Sentiment Analysis: BOW vs TF-IDF vs Word2Vec Benchmark

A benchmark study that asks one question: **which text-representation + classifier combination predicts the sentiment of Amazon Kindle book reviews best?**

I tested **3 feature-extraction techniques × up to 8 machine-learning models = 22 experiments**, all on the same cleaned data and the same train/test split, so the comparison is fair.

> 🏆 **Best result:** TF-IDF + Logistic Regression, **F1 = 0.889**, **Accuracy = 84.4%**, **Recall = 94.4%**

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Limitations & Future Work](#-limitations--future-work)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🎯 Project Overview

Customer reviews are one of the richest sources of feedback for any product, but nobody can read thousands of them by hand. This project builds and compares NLP pipelines that automatically classify a Kindle book review as **positive** or **negative**.

Rather than picking one technique and stopping, I treated this as a **benchmark**: the same data goes through three different ways of turning text into numbers, and each is evaluated with a wide range of classical ML models.

## 📊 Dataset

- **Source:** Amazon Kindle Store reviews, from Julian McAuley's Amazon product data (UCSD).
- **Size used:** 12,000 reviews (a balanced-by-rating subset of the original ~982K-review 5-core dataset).
- **Key columns used:** `reviewText` (input) and `rating` (1–5 stars, used to build the label).

| Rating | Reviews |
|:-----:|:-----:|
| 1 ⭐ | 2,000 |
| 2 ⭐ | 2,000 |
| 3 ⭐ | 2,000 |
| 4 ⭐ | 3,000 |
| 5 ⭐ | 3,000 |

**Label creation:** ratings **< 3 → Negative (0)**, ratings **≥ 3 → Positive (1)**. This gives about 4,000 negative and 8,000 positive reviews (≈ 67% positive).

## 🔄 Workflow

```
Raw Reviews → Cleaning → Lemmatization → Train/Test Split (80/20)
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            ▼                                 ▼                                 ▼
   Bag of Words (BOW)                      TF-IDF                      Word2Vec (averaged)
            │                                 │                                 │
            └──────────── 8 / 8 / 6 classifiers trained on each ────────────────┘
                                              │
                         Accuracy · Precision · Recall · F1 → Benchmark
```

### 1. Preprocessing & Cleaning
- Lower-casing
- Removing special characters
- Stop-word removal (NLTK)
- URL removal (regex)
- HTML tag removal (BeautifulSoup)
- Extra-whitespace removal
- Lemmatization (WordNet)

### 2. Feature Extraction
| Technique | Idea |
|---|---|
| **Bag of Words** | Counts how often each word appears |
| **TF-IDF** | Weights words by how distinctive they are across reviews |
| **Word2Vec** | Dense word embeddings, averaged per review (trained from scratch with Gensim, 200 epochs) |

### 3. Models Compared
`MultinomialNB` · `BernoulliNB` · `ComplementNB` · `Logistic Regression` · `Linear SVM` · `Random Forest` · `Decision Tree` · `KNN`

(Naive Bayes variants that require non-negative features, `MultinomialNB` and `ComplementNB`, were applied to BOW and TF-IDF only, since averaged Word2Vec vectors can be negative.)

### 4. Evaluation
Accuracy, Precision, Recall and F1-score on a held-out 20% test set (2,400 reviews, `random_state=42`). Models were ranked by **F1-score**.

## 🏁 Results

### 🥇 Top 10 Models Overall

| Rank | Features | Model | Accuracy | Precision | Recall | F1 |
|:--:|:--:|:--|:--:|:--:|:--:|:--:|
| 1 | TF-IDF | Logistic Regression | 0.8438 | 0.8410 | 0.9436 | **0.8893** |
| 2 | TF-IDF | SVM | 0.8417 | 0.8594 | 0.9111 | 0.8845 |
| 3 | BOW | MultinomialNB | 0.8388 | 0.8493 | 0.9211 | 0.8837 |
| 4 | BOW | Logistic Regression | 0.8375 | 0.8612 | 0.9011 | 0.8807 |
| 5 | BOW | ComplementNB | 0.8375 | 0.8673 | 0.8923 | 0.8796 |
| 6 | Word2Vec | SVM | 0.8321 | 0.8499 | 0.9080 | 0.8780 |
| 7 | Word2Vec | Logistic Regression | 0.8321 | 0.8512 | 0.9061 | 0.8778 |
| 8 | Word2Vec | Random Forest | 0.8088 | 0.8165 | 0.9192 | 0.8648 |
| 9 | BOW | Random Forest | 0.7967 | 0.7810 | 0.9649 | 0.8633 |
| 10 | TF-IDF | Random Forest | 0.7958 | 0.7805 | 0.9643 | 0.8627 |

### 🔍 Best Model per Feature Type

| Features | Best Model | Accuracy | F1 |
|:--|:--|:--:|:--:|
| BOW | MultinomialNB | 0.8388 | 0.8837 |
| **TF-IDF** | **Logistic Regression** | **0.8438** | **0.8893** |
| Word2Vec | SVM | 0.8321 | 0.8780 |

## 💡 Key Insights

1. **Simple beats complex.** Linear models (Logistic Regression, SVM) and Naive Bayes consistently outperformed Random Forest, Decision Tree and KNN on sparse text features.
2. **TF-IDF edged out BOW and Word2Vec**, but the margin was small (about 1 point of F1). Good preprocessing and model choice matter more than the representation alone.
3. **The same model can behave very differently across features.** MultinomialNB scored F1 = 0.884 on BOW but dropped to 0.813 on TF-IDF (with near-perfect recall of 0.999, meaning it labelled almost everything as positive). Pairing matters.
4. **Word2Vec trained from scratch on only 12K reviews stayed competitive** with the sparse methods while using far fewer dimensions.
5. **Tree-based models and KNN struggled** with high-dimensional sparse text, as expected.

## ⚠️ Limitations & Future Work

Being upfront about these is part of doing the analysis well:

- **Class imbalance:** about 67% of the data is positive, so a model that always predicts "positive" would reach roughly 67% accuracy. This is why F1, Precision and Recall are reported alongside accuracy, and it explains the high recall values.
- **3-star reviews are labelled positive.** A neutral class or a 3-class setup could be more realistic.
- **Word2Vec was trained from scratch** on a small corpus. The pre-trained `word2vec-google-news-300` vectors were loaded for exploration but not used for the final features.
- **Default hyperparameters** were used for all models.

**Next steps**
- [ ] Hyperparameter tuning with `GridSearchCV` / cross-validation
- [ ] Handle imbalance (class weights, SMOTE) and add a confusion matrix / ROC-AUC
- [ ] Use pre-trained embeddings (GloVe / Google News Word2Vec) and N-gram TF-IDF
- [ ] Fine-tune a transformer (BERT / DistilBERT) and compare against this baseline
- [ ] Deploy as a Streamlit / Flask web app for live predictions

## 🛠 Tech Stack

**Language:** Python
**Data:** Pandas, NumPy
**NLP:** NLTK, BeautifulSoup, Regex, Gensim (Word2Vec)
**ML:** Scikit-learn
**Visualization:** Matplotlib, Seaborn
**Environment:** Jupyter Notebook

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn nltk beautifulsoup4 lxml gensim matplotlib seaborn tqdm jupyter

# 3. Launch the notebook
jupyter notebook Project_2_Kindle_Review_Sentiment_Analysis_Benchmark.ipynb
```

NLTK resources (`stopwords`, `wordnet`) are downloaded automatically by the notebook.

## 📁 Project Structure

```
├── all_kindle_review.csv
├── Project_2_Kindle_Review_Sentiment_Analysis_Benchmark.ipynb
└── README.md
```

## 🙏 Acknowledgements

Dataset: Amazon product data by Julian McAuley, UCSD (http://jmcauley.ucsd.edu/data/amazon/). All rights to the data belong to the original owners.

## 👤 Author

**Your Name**
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile) · 💻 [GitHub](https://github.com/your-username) · ✉️ your.email@example.com

⭐ If you found this useful, consider starring the repo!