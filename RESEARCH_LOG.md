# Autonomous Scam Detection Engine: 7-Day Research Log

## Day 1: Problem Definition & Data Sourcing
- Sourced 5,572 labeled SMS messages (4,825 Ham, 747 Scam).
- Identified target domain: Text-based financial risk management.

## Day 2: Baseline Architecture (Bayesian Engine)
- Implemented core Naive Bayes word-frequency scoring.
- Identified problem: Raw frequency subtraction penalized long normal messages.

## Day 3: Information Theory (Shannon Entropy Layer)
- Added Shannon Entropy engine to catch randomized text/URL tokens.
- Observed improvement in catching automated phishing links.

## Day 4: High-Dimensional Vector Math Engine
- Built a 10-dimensional vector space model using Cosine Similarity.
- Encountered Stopword Backfire (common words distorted vector angles).

## Day 5: Dynamic Vocabulary Extraction
- Replaced 10 hardcoded axes with an automated top-500 word vector extraction system.
- Scaled vector dimensionality dynamically during dataset training.

## Day 6: Benchmark & Evaluation System
- Developed `evaluate.py` using an 80/20 train/test split.
- Generated baseline metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Day 7: Component Isolation & Ablation Testing
- Began systematic single-engine testing (Bayes vs. Entropy vs. Vector Space) to quantify individual mathematical impact.

Experiment  	Engine Active	      Accuracy	      Precision	       Recall	      F1-Score	    TP	TN	FP	FN
Exp 1	        Bayes Only	          42.24%	      19.35%	       98.72%	      32.35%        154	317	642	2
Exp 2	        Entropy Only	      83.41%	      45.51%	       94.23%	      61.38%	    147	783	176	9
Exp 3	        Vector Space Only	  13.99%	      13.99%           100.00%	      24.55%	    156 0   959 0
Exp 4	        	       