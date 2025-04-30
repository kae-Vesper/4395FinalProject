from sklearn.svm import SVC
import numpy as np
import torch

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
        X_train = torch.stack([vec for vec, _ in training_data]).numpy()
        y_train = np.array([label for _, label in training_data])
        X_val = torch.stack([vec for vec, _ in validation_data]).numpy()
        y_val = np.array([label for _, label in validation_data])

        self.model.fit(X_train, y_train)

        train_acc = np.mean(self.model.predict(X_train) == y_train)
        val_acc = np.mean(self.model.predict(X_val) == y_val)
        self.word_weights = self.model.coef_[0]

        print(f"[SVM] Training Accuracy: {train_acc:.3f}")
        print(f"[SVM] Validation Accuracy: {val_acc:.3f}")
        return train_acc, val_acc

    def set_word_index(self, word2idx):
        """
        Save word2index for test-time word lookup
        """
        self.word_index = {idx: word for word, idx in word2idx.items()}

    def predict(self, tfidf_vector):
        """
        tfidf_vector: torch.Tensor of shape (vocab_size,)
        Returns: (predicted label, confidence, top words)
        """
        np_vec = tfidf_vector.numpy().reshape(1, -1)
        prob_dist = self.model.predict_proba(np_vec)[0]
        prediction = self.model.predict(np_vec)[0]
        confidence = round(float(np.max(prob_dist)), 3)

        influences = tfidf_vector.numpy() * self.word_weights
        top_indices = influences.argsort()[-5:][::-1]
        keywords = [self.word_index[i] for i in top_indices if tfidf_vector[i] > 0]

        return prediction, confidence, keywords
