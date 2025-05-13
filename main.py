from time import perf_counter
import torch
import torch.nn as nn
import json
import csv
import math
import string
from collections import defaultdict
from SVM import SupportVectorClassifier
from tdifd import LogisticRegressionModel
from BERT import BERTModel
from tqdm import tqdm

unk = '<UNK>'

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


# Updated make_vocab with frequency-based trimming
def make_vocab(data, max_words=10000, unk_threshold=5):
    word_counts = defaultdict(int)
    # Count all words in training data
    for doc, _ in data:
        for word in doc:
            word_counts[word] += 1

    # Filter rare words below threshold and keep top N
    filtered_words = {word: count for word, count in word_counts.items()
                      if count >= unk_threshold}
    sorted_words = sorted(filtered_words.items(),
                          key=lambda x: -x[1])[:max_words]

    return {word for word, _ in sorted_words}


# Updated make_indices with explicit UNK handling
def make_indices(vocab):
    vocab_list = sorted(vocab)
    # Add UNK token at index 0 for consistent handling
    vocab_list = [unk] + vocab_list
    return {
        "word2index": {word: idx for idx, word in enumerate(vocab_list)},
        "index2word": vocab_list,
        "vocab_size": len(vocab_list)
    }

def convert_to_vector(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def convert_to_tf_vector(data, word2index, vocab):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(vocab)+1)
        for word, score in document.items():
            if word in word2index:
                vector[word2index[word]] = score
        vectorized_data.append((vector, y))
    return vectorized_data

def test_model(name, model, test_data, test_data_vector, word2index):
    count = 0
    accurate = 0
    with open(f"./{name}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence", "impactful words"])

        i=0
        for message, label in tqdm(test_data_vector, desc="Testing", leave=True):
            pred_label, confidence, words = model.test(message,word2index)
            if pred_label == label:
                accurate += 1
            writer.writerow([test_data[count][0], pred_label, label, confidence, words])
            count += 1
    return accurate/count

if __name__ == '__main__':
    #hyperparameters
    hidden_dim = 256
    epochs = 10

    #load data
    print("========== Loading data ==========\n")
    train_data, valid_data, test_data = get_data()
    vocab = make_vocab(train_data)
    indices = make_indices(vocab)
    word2index = indices["word2index"]
    index2word = indices["index2word"]

    #Compute TF-IDF
    print("========== Computing TF-TDF ==========")
    start_time = perf_counter()
    train_data_vector = compute_tf_idf(train_data)
    valid_data_vector = compute_tf_idf(valid_data)
    test_data_vector = compute_tf_idf(test_data)
    end_time = perf_counter()
    print(f"TF-IDF computation time: {end_time - start_time:.4f} seconds\n")

    #Vectorize data
    print("========== Vectorizing data for TF-IDF ==========")
    start_time = perf_counter()
    train_data_vector = convert_to_tf_vector(train_data_vector, word2index, vocab)
    valid_data_vector = convert_to_tf_vector(valid_data_vector, word2index, vocab)
    test_data_vector = convert_to_tf_vector(test_data_vector, word2index, vocab)
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time: .4f} seconds \n")

    #logical regression model
    print("========== Building logical regression model ==========\n")
    #initialize model
    lr_model = LogisticRegressionModel(len(vocab)+1)

    #train
    print("---------- Training model ----------")
    start_time = perf_counter()
    print(f"Training data size: {len(train_data_vector)}")
    lr_model.train_model(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time-start_time:.4f} seconds\n")

    #test
    print("---------- Testing model -----------")
    start_time = perf_counter()
    test_acc = test_model("Logical_Regression_Model", lr_model, test_data, test_data_vector, word2index)
    end_time = perf_counter()
    print(f"Test accuracy: {test_acc:.2f}")
    print(f"Testing time:  {end_time-start_time:.4f} seconds\n")

    #SVM model
    print("========== Building SVM model ==========\n")
    # initialize model
    svm_model = SupportVectorClassifier()
    svm_model.set_word_index(word2index)

    print("---------- Vectorizing data for SVM ----------")
    start_time = perf_counter()
    train_data_vector = convert_to_vector(train_data, word2index)
    valid_data_vector = convert_to_vector(valid_data, word2index)
    test_data_vector = convert_to_vector(test_data, word2index)
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time: .4f} seconds \n")

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    train_acc, valid_acc = svm_model.fit(train_data_vector, valid_data_vector, epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time - start_time:.4f} seconds\n")

    # test
    print("---------- Testing model -----------")
    start_time = perf_counter()

    accurate = 0
    with open("./svm.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence", "keywords"])

        for i, (vector, true_label) in enumerate(test_data_vector):
            # Extract message text from original test data
            message = test_data[i][0]

            # Get prediction for single vector
            pred_label, confidence, keywords = svm_model.predict(vector)

            # Write results
            writer.writerow([message, pred_label, true_label, confidence, keywords])

            if pred_label == true_label:
                accurate += 1

    print("Test accuracy: ", accurate / len(test_data_vector))

    print(f"Testing time:  {end_time - start_time:.4f} seconds\n")

    #BERT model
    print("========== Building BERT model ==========\n")
    # initialize model
    bert_model = BERTModel()

    #vectorizing data for bert model
    print("---------- Vectorizing data for BERT ----------")
    start_time = perf_counter()
    train_texts = [' '.join(doc) for doc, label in train_data]
    train_labels = [label for doc, label in train_data]
    valid_texts = [' '.join(doc) for doc, label in valid_data]
    valid_labels = [label for doc, label in valid_data]
    test_texts = [' '.join(doc) for doc, label in test_data]
    test_labels = [label for doc, label in test_data]
    end_time = perf_counter()
    print(f"Vectorization time: {end_time - start_time:.4f} seconds\n")

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    bert_model.train((train_texts, train_labels), (valid_texts, valid_labels), epochs)
    end_time = perf_counter()
    print(f"Training time:  {end_time - start_time:.4f} seconds\n")

    # test
    print("---------- Testing model -----------")
    start_time = perf_counter()

    # Get detailed predictions
    predictions = bert_model.test((test_texts, test_labels), return_details=True)

    # Write results to CSV
    with open("./bert.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "pred_label", "label", "confidence", "impactful_words"])

        for text, pred_label, confidence, true_label in tqdm(predictions, desc="building csv", leave=True):
            impactful_words = bert_model.get_word_impacts(text, true_label)
            impactful_words = [item[0] for item in impactful_words]

            writer.writerow([
                text,
                pred_label,
                true_label,
                f"{confidence:.4f}",
                impactful_words
            ])

    end_time = perf_counter()
    print(f"Testing time: {end_time - start_time:.4f} seconds")
