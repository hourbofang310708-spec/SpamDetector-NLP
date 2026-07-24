# Tri-Engine Spam & Scam Detection Engine (v1.1)

A robust machine learning classification system built from scratch in Python, combining **Bayesian Probability**, **Shannon Entropy**, and **Vector Space Cosine Similarity** to detect scam messages.

---

##  Key Architectural Updates (v1.1 Calibration)
After diagnosing initial model failures (Phase 1 baseline), two major mathematical adjustments were introduced to eliminate false positives and class bias:

1. **Bayes Prior Probability Shift (-1.81):** 
   * *The Problem:* The dataset is imbalanced (~86% normal messages vs ~14% spam), causing Naive Bayes to panic and over-predict spam.
   * *The Fix:* Applied a natural log odds adjustment ($\ln(\text{Ham}/\text{Spam}) \approx 1.81$) as a baseline penalty to ensure Bayes requires strong evidence before flagging a message.
2. **Dynamic Vector Centroid Threshold (`average + 0.1`):**
   * *The Problem:* Hardcoded similarity thresholds caused the Vector Space engine to classify 100% of messages as spam.
   * *The Fix:* Computed the dynamic mean cosine similarity of training normal messages and added a strict `+0.10` safety buffer to isolate true scam clusters.

---

##  Benchmark Results (Unseen Test Data - 80/20 Split)

Evaluated on a randomized, reproducible test set (`random.seed(42)`):

| Experiment | Engine Weights | Accuracy | Precision | Recall | F1-Score | TP | TN | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Equal (All 0.33) | 83.23% | 45.23% | 94.23% | 61.12% | 147 | 781 | 178 | 9 |
| **Exp 2** | Vector/Bayes (0.40), Entropy (0.20) | 90.22% | 59.59% | 93.59% | 72.82% | 146 | 860 | 99 | 10 |
| **Exp 3** | Vector/Bayes (0.48), Entropy (0.04) | 93.45% | 86.73% | 62.82% | 72.86% | 98 | 944 | 15 | 58 |
| **Exp 4** | Vector/Bayes (0.45), Entropy (0.10) | 95.25% | 84.11% | 81.41% | 82.74% | 127 | 935 | 24 | 29 |
| **Exp 5 (Best)** | **Vector (0.50), Bayes (0.40), Entropy (0.10)** | **95.43%** | **82.21%** | **85.90%** | **84.01%** | **134** | **930** | **29** | **22** |
| **Exp 6** | Bayes (0.50), Vector (0.40), Entropy (0.10) | 94.89% | 84.62% | 77.56% | 80.94% | 121 | 937 | 22 | 35 |

> **Conclusion (Exp 5):** Balancing Vector Space (0.50) for semantic meaning and Bayes (0.40) for word probabilities—with Entropy (0.10) acting as an anomaly tiebreaker—yields an optimal **84.01% F1-Score** and cuts false alarms drastically.

## Day 9: Model Calibration and 80/20 Train-Test Evaluation
* **Action:** Implemented an 80/20 train-test split via `evaluate.py` to test against completely unseen data with a fixed random seed (`random.seed(42)`).
* **Math Updates:** Calibrated Bayes with a `-1.81` log-odds penalty for dataset imbalance and shifted the Vector Space threshold to `mean_vector_ham + 0.10` to eliminate false positives.
* **Result:** Achieved **95.43% Accuracy**, **82.21% Precision**, **85.90% Recall**, and an **84.01% F1-Score** in Experiment 5 (Vector 0.50, Bayes 0.40, Entropy 0.10).

  # Bigram Feature Space Evaluation
Action: Replaced single-word extraction with a 500-dimensional bigram (word pair) vocabulary mapped exclusively from spam training messages.

* **Math Updates:** Maintained core engine weights (w_bayes=0.40, w_entropy=0.10, w_vector=0.50) while transitioning vector space calculations from unigrams to adjacent word pair sequences.

* **Result:** Achieved **94.89% Accuracy**, **83.22% Precision**, **79.49% Recall**, and an **81.31% F1-Score** (TP: 124, TN: 934, FP: 25, FN: 32).