from sklearn.svm import SVC
import numpy as np
import torch
from scipy.sparse import csr_matrix

class SupportVectorClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear", probability=True, C=1.0)
        self.word_weights = None
        self.word_index = None

    def fit(self, training_data, validation_data, epochs=5):
        """
        training_data: List of (torch.Tensor, label)
        validation_data: List of (torch.Tensor, label)
        """
        x_train = csr_matrix([vec.numpy() for vec, _ in training_data])
        y_train = np.array([label for _, label in training_data])
        x_val = csr_matrix([vec.numpy() for vec, _ in validation_data])
        y_val = np.array([label for _, label in validation_data])

        self.model.fit(x_train, y_train)

        train_acc = np.mean(self.model.predict(x_train) == y_train)
        val_acc = np.mean(self.model.predict(x_val) == y_val)
        self.word_weights = self.model.coef_[0].toarray().flatten()

        print(f"[SVM] Training Accuracy: {train_acc:.3f}")
        print(f"[SVM] Validation Accuracy: {val_acc:.3f}")
        return train_acc, val_acc

    def set_word_index(self, word2idx):
        """
        Save word2index for test-time word lookup
        """
        self.word_index = {idx: word for word, idx in word2idx.items()}

    def predict(self, vector):
        """
        tfidf_vector: torch.Tensor of shape (vocab_size,)
        Returns: (predicted label, confidence, top words)
        """
        np_vec = vector.numpy().reshape(1, -1)
        prob_dist = self.model.predict_proba(np_vec)[0]
        prediction = self.model.predict(np_vec)[0]
        confidence = round(float(np.max(prob_dist)), 3)

        reshaped_weights = self.word_weights.squeeze()
        influences = vector.numpy() * reshaped_weights
        top_indices = influences.argsort()[-5:][::-1]
        keywords = [self.word_index[i] for i in top_indices if vector[i] > 0]

        return prediction, confidence, keywords
