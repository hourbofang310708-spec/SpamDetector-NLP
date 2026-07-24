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

Experiment  	Engine Active	                                    Accuracy	                   Precision	       Recall	      F1-Score	    TP	TN	FP	FN
Exp 1	        Bayes Only	                                        42.24%	                        19.35%	           98.72%	      32.35%        154	317	642	2
Exp 2	        Entropy Only	                                    83.41%	                        45.51%	           94.23%	      61.38%	    147	783	176	9
Exp 3	        Vector Space Only	                                13.99%	                        13.99%             100.00%	      24.55%	    156 0   959 0
Exp 4	        Equality(bayes = vector space = entropy= 0.33)      65.65%                          28.71%             98.08%         44.41%        153 579 380 3
Exp 5           Bayes heavily (b = 0.70, v = e = 0. 15)             55.87%                          23.83%             98.08%         38.35%        153 470 489 3 
Exp 6           entropy heavily (b = v = 0.15, e = 0. 70 )          82.42%                          44.08%             95.51%         60.32%        149 770 189 7   
Exp 7           vector heavily ( v = 0.70, b = e = 0.15 )           61.52%                          26.42%             98.08%         41.63%        153 533 426 3

## Day 8: 
   # Vector Adjustment: Shifted threshold to average + 0.1 to prevent false positives.
   # Bayes Adjustment: Applied a -1.81 penalty to account for the imbalanced 86/14 ham-to-spam ratio. 
Experiment       Engine Active                                      Accuracy                      Precision             Recall      F1-Score
 TP  TN  FP  FN 

Exp 1            eqaulity( all = 0.33)                              83.23%                        45.23%                94.23%        61.12%       147 781 178 9 

Exp 2            vector and bayes heavily (both = 0.4)              90.22%                        59.59%                93.59%        72.82%       146 860 99  10
Exp 3            vector and bayes heavily (1) ( both = 0.48)        93.45%                        86.73%                62.82%        72.86%       98  944 15  58
Exp 4            vector and bayes heavily (2) ( both 0.45)          95.25%                        84.11%                81.41%        82.74%       127 935 24  29

Exp 5            vector(0.5), bayes ( 0.4)                          95.43%                        82.21%                85.90%        84.01%       134 930 29  22
Exp 6            bayes(0.5), vector(0.4)                            94.89%                        84.62%                77.56%        80.94%       121 937 22  35

Exp 7            Entropy(0.5)                                       83.23%                        45.23%                94.23%        61.12%       147 781 178 9

## Day 9:
    # 500_dimension used from the spam messages only. Since the Exp 5 ( Day 8 ) is the best overall. I am going to use the same for Day 9

Experiment              Engine Acitve                                        Accuracy     Precision    Recall  F1-Score   TP  TN  FP  FN
Exp1 (unigrams)         w_bayes=0.40, w_entropy=0.1, w_vector=0.50           95.16%       82.28%       83.33%  82.80%     130 931 28  26

Exp2 (bigrams)          w_bayes=0.40, w_entropy=0.1, w_vector=0.50           94.89%       83.22%       79.49%  81.31%     124 934 25  32


             


