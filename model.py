import math
import csv
from collections import Counter

class SpamDetector:
    def __init__(self):
        self.spam_word_counts = Counter()
        self.normal_word_counts = Counter()
        self.total_spams_seen = 0
        self.total_normals_seen = 0
        self.vector_axes = []
        self.average_spam_vector = []
        
        self.mean_vector_ham = 0.0
        self.mean_entropy_ham = 0.0

    def extract_bigrams(self, text):
        """Takes a message and breaks it into adjacent word pairs (bigrams)."""
        words = text.lower().split()
        if len(words) < 2:  # Fixed: must be less than 2 to skip empty/single-word lists
            return []
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        return bigrams

    def train(self, filepath): 
        raw_spam_messages = []
        raw_ham_messages = []  
        
        # Reset counters for a fresh training session
        self.spam_word_counts = Counter()
        self.normal_word_counts = Counter()
        self.total_spams_seen = 0
        self.total_normals_seen = 0

        bigram_counts = {}

        # 1. Open file safely and read rows
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue  # skip blank lines
                
                label = row[0].lower().strip()   # 'ham' or 'spam'
                message = row[1].lower().strip() # the actual text
                words = message.split()          # individual words
                
                if label == 'spam':
                    self.total_spams_seen += 1
                    raw_spam_messages.append(message) # Store full message string for bigrams
                    self.spam_word_counts.update(words)
                else:
                    self.total_normals_seen += 1
                    raw_ham_messages.append(message) 
                    self.normal_word_counts.update(words)

        # 2. Build 500-D Vector Vocabulary exclusively from the "Spam Head" Bigrams
        for msg in raw_spam_messages:
            msg_bigrams = self.extract_bigrams(msg)
            for bg in msg_bigrams:
                bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
                
        # Sort bigrams by frequency and lock in the top 500 most dangerous pairs
        sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
        self.vector_axes = [bg for bg, count in sorted_bigrams[:500]]

        # 3. Build Average Spam Vector using the bigram axes
        raw_spam_vectors = []
        for msg in raw_spam_messages:
            msg_bigrams = self.extract_bigrams(msg)
            message_vector = [1 if axis_bg in msg_bigrams else 0 for axis_bg in self.vector_axes]
            raw_spam_vectors.append(message_vector)

        num_axes = len(self.vector_axes)
        self.average_spam_vector = [0.0] * num_axes
        
        if self.total_spams_seen > 0 and num_axes > 0:
            for i in range(num_axes):
                total_column_sum = sum(vector[i] for vector in raw_spam_vectors)
                self.average_spam_vector[i] = total_column_sum / self.total_spams_seen

        # 4. Calculate baseline averages for normal messages
        if raw_ham_messages:
            vector_scores_ham = [self._calculate_cosine_similarity(msg) for msg in raw_ham_messages]
            entropy_scores_ham = [self._calculate_entropy(msg) for msg in raw_ham_messages]

            self.mean_vector_ham = sum(vector_scores_ham) / len(vector_scores_ham)
            self.mean_entropy_ham = sum(entropy_scores_ham) / len(entropy_scores_ham)

        print(f"[AI Brain Updated]: Learned from {self.total_spams_seen} spams and {self.total_normals_seen} normal messages.")
        print(f"[Vector Space Configured]: Automatically mapped a {len(self.vector_axes)}-dimensional bigram vocabulary!")
        print(f"[Baselines Computed]: Normal Vector Avg = {self.mean_vector_ham:.4f}, Normal Entropy Avg = {self.mean_entropy_ham:.4f}")

    # ==========================================
    # ENGINE 1: BAYESIAN PROBABILITY
    # ==========================================
    def _calculate_bayes(self, text):
        words = text.lower().split()
        spam_evidence = 0.0
        
        for word in words:
            if word in self.spam_word_counts and self.total_spams_seen > 0:
                spam_prob = self.spam_word_counts[word] / self.total_spams_seen
                spam_evidence += spam_prob
            if word in self.normal_word_counts and self.total_normals_seen > 0:
                normal_prob = self.normal_word_counts[word] / self.total_normals_seen
                spam_evidence -= normal_prob

        return spam_evidence

    # ==========================================
    # ENGINE 2: SHANNON ENTROPY
    # ==========================================
    def _calculate_entropy(self, text):
        if not text:
            return 0.0
            
        frequencies = Counter(text)
        length = len(text)
        entropy = 0.0
        
        for count in frequencies.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
            
        if entropy > 4.2:
            return 2.5  
        else:
            return -1.5 

    # ==========================================
    # ENGINE 3: VECTOR MATH (COSINE SIMILARITY - BIGRAMS)
    # ==========================================
    def _calculate_cosine_similarity(self, text):
        # Updated to use bigrams instead of single words for the vector engine!
        message_bigrams = self.extract_bigrams(text)

        user_vector = [1 if axis_bg in message_bigrams else 0 for axis_bg in self.vector_axes]
        
        if not self.average_spam_vector or len(user_vector) != len(self.average_spam_vector):
            return 0.0

        dot_product = sum(u * s for u, s in zip(user_vector, self.average_spam_vector))
        magnitude_user = math.sqrt(sum(u**2 for u in user_vector))
        magnitude_spam = math.sqrt(sum(s**2 for s in self.average_spam_vector))
        
        if magnitude_user == 0 or magnitude_spam == 0:
            return 0.0 
            
        similarity = dot_product / (magnitude_user * magnitude_spam)
        return similarity 

    def _sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1 / (1 + math.exp(-z))

    # ==========================================
    # THE MASTER DECISION MAKER
    # ==========================================
    def predict(self, new_message, w_bayes=0.40, w_entropy=0.1, w_vector=0.50):
        # 1. Gather raw scores from all three engines
        bayes_score = self._calculate_bayes(new_message)
        entropy_score = self._calculate_entropy(new_message)
        vector_score = self._calculate_cosine_similarity(new_message)

        # 2. Apply calibrated penalties and baselines
        c_bayes   = bayes_score - 1.81
        c_entropy = entropy_score - self.mean_entropy_ham 
        c_vector  = vector_score - (self.mean_vector_ham + 0.10)  
        
        # 3. Combine using weights
        z = (w_bayes * c_bayes) + (w_entropy * c_entropy) + (w_vector * c_vector)
        
        final_probability = self._sigmoid(z)
        
        return {
            "bayes": round(bayes_score, 2),
            "entropy": round(entropy_score, 2),
            "vector": round(vector_score, 2),
            "z_raw_score": round(z, 2),
            "final_scam_probability": round(final_probability * 100, 2)
        }