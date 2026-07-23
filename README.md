# 🛡️ Tri-Engine Spam & Fraud Detector (v1.0)

An ensemble NLP classification engine built from scratch in pure Python without high-level Machine Learning frameworks.

## 🔬 Core Architecture
This system evaluates incoming text messages across three distinct mathematical dimensions:

1. **Bayesian Log-Probability:** Measures domain word frequencies and likelihood ratios.
2. **Shannon Entropy:** Measures structural character randomness to detect masked spam/obfuscation.
3. **500-D Vector Space (Cosine Similarity):** Computes high-dimensional geometric similarity against known spam centroids.

## 📐 Zero-Magic-Number Calibration
Rather than using arbitrary thresholds, all decision boundaries are dynamically zero-centered by calculating normal baseline means directly from training data ($\mu_{\text{vector}}$, $\mu_{\text{entropy}}$) during initialization.

## 🚀 How to Run
```bash
python evaluate.py

