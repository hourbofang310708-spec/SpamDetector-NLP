import csv
import random
from model import SpamDetector

def run_benchmark(dataset_path, train_ratio=0.8):
    print("==================================================")
    print("   AI MODEL BENCHMARK & EVALUATION SYSTEM         ")
    print("==================================================\n")
    
    # 1. Load all data rows
    rows = []
    with open(dataset_path, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2 and row[0].lower() in ['ham', 'spam']:
                rows.append((row[0].lower(), row[1]))

    print(f"Total dataset size: {len(rows)} messages.")
    
    # Shuffle randomly to ensure an unbiased mix
    random.seed(42)  # Fixed seed so tests are reproducible!
    random.shuffle(rows)

    # 2. Split into 80% Training and 20% Testing
    split_index = int(len(rows) * train_ratio)
    train_data = rows[:split_index]
    test_data = rows[split_index:]

    print(f"Training Set: {len(train_data)} samples | Test Set: {len(test_data)} samples\n")

    # Save training set to a temporary file for model.train()
    temp_train_file = "data/temp_train.txt"
    with open(temp_train_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for label, msg in train_data:
            writer.writerow([label, msg])

    # 3. Train the AI on the 80% set ONLY
    ai = SpamDetector()
    ai.train(temp_train_file)
    print("\nRunning test evaluation on unseen data...")

    # 4. Initialize Confusion Matrix Counters
    tp = 0  # True Positive:  Actual Scam  -> Predicted Scam (Correct)
    tn = 0  # True Negative:  Actual Normal-> Predicted Normal (Correct)
    fp = 0  # False Positive: Actual Normal-> Predicted Scam (False Alarm)
    fn = 0  # False Negative: Actual Scam  -> Predicted Normal (Dangerous Miss)

    # 5. Test the AI on every message in the hidden 20% set
    for actual_label, message in test_data:
        prediction_result = ai.predict(message)
        scam_prob = prediction_result['final_scam_probability']
        
        predicted_label = 'spam' if scam_prob >= 50.0 else 'ham'

        if actual_label == 'spam' and predicted_label == 'spam':
            tp += 1
        elif actual_label == 'ham' and predicted_label == 'ham':
            tn += 1
        elif actual_label == 'ham' and predicted_label == 'spam':
            fp += 1
        elif actual_label == 'spam' and predicted_label == 'ham':
            fn += 1

    total_tests = len(test_data)
    
    # 6. Compute Metrics
    accuracy = ((tp + tn) / total_tests) * 100 if total_tests > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 7. Print the Formal Data Science Report
    print("\n==================================================")
    print("            MODEL PERFORMANCE REPORT              ")
    print("==================================================")
    print(f" Total Exam Questions (Test Set): {total_tests}")
    print("--------------------------------------------------")
    print(f" True Positives  (Scams Caught):    {tp}")
    print(f" True Negatives  (Normals Cleared): {tn}")
    print(f" False Positives (False Alarms):    {fp}")
    print(f" False Negatives (Missed Scams):    {fn}")
    print("--------------------------------------------------")
    print(f" ACCURACY:  {accuracy:.2f}%")
    print(f" PRECISION: {precision:.2f}%")
    print(f" RECALL:    {recall:.2f}%")
    print(f" F1-SCORE:  {f1_score:.2f}%")
    print("==================================================\n")

if __name__ == "__main__":
    run_benchmark("data/spam_data.txt")