#Rohan - Implementing TD-IDF with Linear regression
from time import perf_counter
import torch
import torch.nn as nn
import json
import csv
import math
import string
from collections import defaultdict

unk = '<UNK>'

# Here we are preprocessing: Lowercase, remove punctuation, tokenize
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

# After that we will load the data
def get_data():
    label_map = {'ham': 0, 'spam': 1}
    datasets = {}

    for name in ["training.json", "validation.json", "test.json"]:
        try:
            with open(f'./{name}', 'r') as file:
                datasets[name] = json.load(file)
        except FileNotFoundError:
            print(f"Error: {name} not found.")
            exit(1)

    tra, val, test = [], [], []

    for dataset_name, dataset in datasets.items():
        for elt in dataset:
            label = elt.get("label", "").strip().lower()
            if label in label_map:
                processed_message = preprocess(elt.get("message", ""))
                if processed_message:
                    entry = (processed_message, label_map[label])
                    if dataset_name == "training.json":
                        tra.append(entry)
                    elif dataset_name == "validation.json":
                        val.append(entry)
                    else:
                        test.append(entry)

    return tra, val, test

# Computing the TD-IDF
def compute_tf_idf(data):
    doc_count = len(data)
    word_doc_freq = defaultdict(int)

    for document, _ in data:
        unique_words = set(document)
        for word in unique_words:
            word_doc_freq[word] += 1

    vectors = []
    for document, y in data:
        vector = {}
        total_words = len(document)
        for word in document:
            tf = document.count(word) / total_words
            idf = math.log((1 + doc_count) / (1 + word_doc_freq[word])) + 1
            vector[word] = tf * idf
        vectors.append((vector, y))

    return vectors

# Convert to tensor
def convert_to_vector(data, vocab):
    vectorized_data = []
    vocab_list = list(vocab)
    word2index = {word: i for i, word in enumerate(vocab_list)}

    for document, y in data:
        vector = torch.zeros(len(vocab))
        for word, score in document.items():
            if word in word2index:
                vector[word2index[word]] = score
        vectorized_data.append((vector, y))

    return vectorized_data, word2index

class LogisticRegressionModel(nn.Module):
    def __init__(self, vocab_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(vocab_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train_model(self, train_data, valid_data, epochs=10, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            epoch_loss = 0
            print(f"Starting epoch {epoch+1}")
            for x, y in train_data:
                y = torch.tensor([y], dtype=torch.float32)
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            valid_acc = self.evaluate(valid_data)
            print(f"Epoch {epoch+1} finished: Loss = {epoch_loss:.4f}, Validation Accuracy = {valid_acc:.2f}")

    def evaluate(self, data):
        correct = 0
        for x, y in data:
            pred_label = 1 if self.forward(x).item() > 0.5 else 0
            if pred_label == y:
                correct += 1
        return correct / len(data)

    def test(self, message_tensor, word2index):
        # Create reverse index mapping
        index2word = {i: word for word, i in word2index.items()}

        # Get non-zero indices and scores
        non_zero_indices = torch.nonzero(message_tensor).squeeze()
        impactful_words = []

        # Convert tensor to CPU for numpy conversion if needed
        message_tensor = message_tensor.cpu()

        # Extract top impactful words using weight matrix
        weights = self.linear.weight[0].detach()
        top_indices = torch.argsort(torch.abs(weights), descending=True)[:5]

        impactful_words = [index2word[idx.item()] for idx in top_indices
                           if idx.item() in index2word]

        # Get prediction
        pred_confidence = self.forward(message_tensor).item()
        pred_label = 1 if pred_confidence > 0.5 else 0

        return pred_label, pred_confidence, impactful_words

def test_model(name, model, test_data, test_data_vector, word2index):
    correct = 0
    with open(f"{name}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence", "impactful words"])

        i=0
        for message, label in test_data_vector:
            pred_label, confidence, words = model.test(message, word2index)
            if pred_label == label:
                correct += 1
            writer.writerow([test_data[i], pred_label, label, f"{confidence:.2f}", ", ".join(words)])
            i = i+1

    return correct / len(test_data)

if __name__ == '__main__':
    epochs = 10
    print("========== Loading data ==========\n")
    train_data, valid_data, test_data = get_data()

    print("========== Computing TF-IDF ==========\n")
    start_time = perf_counter()
    train_data_vector = compute_tf_idf(train_data)
    valid_data_vector = compute_tf_idf(valid_data)
    test_data_vector = compute_tf_idf(test_data)
    vocab = set(word for doc, _ in train_data_vector for word in doc.keys())
    train_data_vector, word2index = convert_to_vector(train_data_vector, vocab)
    valid_data_vector, _ = convert_to_vector(valid_data_vector, vocab)
    test_data_vector, _ = convert_to_vector(test_data_vector, vocab)
    end_time = perf_counter()
    print(f"TF-IDF computation time: {end_time - start_time:.4f} seconds")

    print("========== Training Logistic Regression Model ==========\n")
    lr_model = LogisticRegressionModel(len(vocab))
    start_time = perf_counter()
    lr_model.train_model(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time: {end_time - start_time:.4f} seconds")

    print("\n========== Testing Model ==========\n")
    start_time = perf_counter()
    test_acc = test_model("Logical_Regression_Model", lr_model, test_data, test_data_vector, word2index)
    end_time = perf_counter()
    print(f"Test accuracy: {test_acc:.2f}")
    print(f"Testing time: {end_time - start_time:.4f} seconds")
