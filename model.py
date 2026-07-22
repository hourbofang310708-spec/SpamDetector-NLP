import math
import csv
from collections import Counter

class SpamDetector:
    def __init__(self):
        self.scam_word_counts = {}
        self.normal_word_counts = {}
        self.total_scams_seen = 0
        self.total_normals_seen = 0
        self.vector_axes = []
        self.average_scam_vector = []
        
        # -----------------------------------------------------------------
        # 🛠️ ADJUSTMENT 1: Data-Driven Normal Baselines (No Magic Numbers!)
        # Stores the mathematical average of regular messages computed in train()
        # -----------------------------------------------------------------
        self.mean_vector_ham = 0.0
        self.mean_entropy_ham = 0.0

    def train(self, filepath):
        raw_scam_messages = []
        raw_ham_messages = []  # 🛠️ Store ham messages to calculate average baseline
        all_words_seen = Counter()

        # Open file safely
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue  # skip blank lines
                
                label = row[0].lower().strip()   # 'ham' or 'spam'
                message = row[1].lower().strip() # the actual text
                words = message.split()          # individual words
                
                all_words_seen.update(words)

                if label == 'spam':
                    self.total_scams_seen += 1
                    raw_scam_messages.append(words) # 🛠️ ADJUSTMENT 2: Populated raw_scam_messages so vector math works!
                    for word in words:
                        self.scam_word_counts[word] = self.scam_word_counts.get(word, 0) + 1
                else:
                    self.total_normals_seen += 1
                    raw_ham_messages.append(message) # Store message text for ham baseline calculations
                    for word in words:
                        # 🛠️ ADJUSTMENT 3: Fixed bug where scam_word_counts was used instead of normal_word_counts
                        self.normal_word_counts[word] = self.normal_word_counts.get(word, 0) + 1

        # Build 500-D Vector Vocabulary
        distinctive_words = []
        for word, total_count in all_words_seen.items():
            if len(word) < 3 or total_count < 5:
                continue
            distinctive_words.append(word)
            
        self.vector_axes = distinctive_words[:500]

        # Build Average Scam Vector
        raw_scam_vectors = []
        for scam_words in raw_scam_messages:
            message_vector = [1 if axis_word in scam_words else 0 for axis_word in self.vector_axes]
            raw_scam_vectors.append(message_vector)

        num_axes = len(self.vector_axes)
        self.average_scam_vector = [0.0] * num_axes
        
        if self.total_scams_seen > 0 and num_axes > 0:
            for i in range(num_axes):
                total_column_sum = sum(vector[i] for vector in raw_scam_vectors)
                self.average_scam_vector[i] = total_column_sum / self.total_scams_seen

        # -----------------------------------------------------------------
        # 🛠️ ADJUSTMENT 4: Compute data-driven baselines for normal ("ham") messages!
        # -----------------------------------------------------------------
        if raw_ham_messages:
            vector_scores_ham = [self._calculate_cosine_similarity(msg) for msg in raw_ham_messages]
            entropy_scores_ham = [self._calculate_entropy(msg) for msg in raw_ham_messages]

            self.mean_vector_ham = sum(vector_scores_ham) / len(vector_scores_ham)
            self.mean_entropy_ham = sum(entropy_scores_ham) / len(entropy_scores_ham)

        print(f"[AI Brain Updated]: Learned from {self.total_scams_seen} scams and {self.total_normals_seen} normal messages.")
        print(f"[Vector Space Configured]: Automatically mapped a {len(self.vector_axes)}-dimensional vocabulary!")
        print(f"[Baselines Computed]: Normal Vector Avg = {self.mean_vector_ham:.4f}, Normal Entropy Avg = {self.mean_entropy_ham:.4f}")

    # ==========================================
    # ENGINE 1: BAYESIAN PROBABILITY
    # ==========================================
    def _calculate_bayes(self, text):
        words = text.lower().split()
        scam_evidence = 0.0
        
        for word in words:
            if word in self.scam_word_counts and self.total_scams_seen > 0:
                scam_prob = self.scam_word_counts[word] / self.total_scams_seen
                scam_evidence += scam_prob
            if word in self.normal_word_counts and self.total_normals_seen > 0:
                normal_prob = self.normal_word_counts[word] / self.total_normals_seen
                scam_evidence -= normal_prob

        # 🛠️ ADJUSTMENT 5: Moved return OUTSIDE the for-loop! (It was returning after inspecting only 1 word!)
        return scam_evidence

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
            return 2.5  # High entropy signal
        else:
            return -1.5 # Normal language signal

    # ==========================================
    # ENGINE 3: VECTOR MATH (COSINE SIMILARITY)
    # ==========================================
    def _calculate_cosine_similarity(self, text):
        words = text.lower().split()

        user_vector = [1 if axis_word in words else 0 for axis_word in self.vector_axes]
        dot_product = sum(u * s for u, s in zip(user_vector, self.average_scam_vector))
        magnitude_user = math.sqrt(sum(u**2 for u in user_vector))
        magnitude_scam = math.sqrt(sum(s**2 for s in self.average_scam_vector))
        
        if magnitude_user == 0 or magnitude_scam == 0:
            return 0.0 
            
        similarity = dot_product / (magnitude_user * magnitude_scam)

        # 🛠️ ADJUSTMENT 6: Removed artificial `* 5.0` multiplier! Return pure 0.0 to 1.0 score.
        return similarity 

    def _sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1 / (1 + math.exp(-z))

    # ==========================================
    # THE MASTER DECISION MAKER
    # ==========================================
    def predict(self, new_message, w_bayes=0.20, w_entropy=0.60, w_vector=0.20):
        # 1. Gather raw scores
        bayes_score = self._calculate_bayes(new_message)
        entropy_score = self._calculate_entropy(new_message)
        vector_score = self._calculate_cosine_similarity(new_message)
        
        # -------------------------------------------------------------
        # 🛠️ ADJUSTMENT 7: Zero-Center around normal baselines (NO MAGIC NUMBERS!)
        # -------------------------------------------------------------
        c_bayes   = bayes_score                           # Bayes log-odds is naturally centered
        c_entropy = entropy_score - self.mean_entropy_ham # Subtract training average
        c_vector  = vector_score - self.mean_vector_ham   # Subtract training average

        # -------------------------------------------------------------
        # 🛠️ ADJUSTMENT 8: Apply single-pass ensemble weights
        # -------------------------------------------------------------
        z = (w_bayes * c_bayes) + (w_entropy * c_entropy) + (w_vector * c_vector)
        
        final_probability = self._sigmoid(z)
        
        return {
            "bayes": round(bayes_score, 2),
            "entropy": round(entropy_score, 2),
            "vector": round(vector_score, 2),
            "z_raw_score": round(z, 2),
            "final_scam_probability": round(final_probability * 100, 2)
        }