from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import time

unk = '<UNK>'

def get_data():
    with open('./training.json', 'r') as training_f:
        training = json.load(training_f)
    with open('./validation.json', 'r') as validation_f:
        validation = json.load(validation_f)
    with open('./test.json', 'r') as test_f:
        testing = json.load(test_f)

    tra = []
    val = []
    test = []

    for elt in training:
        tra.append((elt["message"].split(), int(elt["label"])))
    for elt in validation:
        val.append((elt["message"].split(), int(elt["label"])))
    for elt in testing:
        test.append((elt["message"].split(), int(elt["label"])))

    return tra, val, test

if __name__ == '__main__':
    #hyperparameters
    hidden_dim = 256
    epochs = 10

    #load data
    print("========== Loading data ==========\n")
    train_data, valid_data, test_data = get_data()
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    #Vectorize data
    print("========== Vectorizing data ==========\n")
    train_data = convert_to_vector(train_data, word2index)
    valid_data = convert_to_vector(valid_data, word2index)
    test_data = convert_to_vector(test_data, word2index)

    #logical regression model
    print("========== Building logical regression model ==========\n")
    #initialize model
    lr_model = LogisticRegressionModel()

    #train
    print("---------- Training model ----------")
    start_time = perf_counter()
    train_acc, valid_acc = lr_model.train(train_data, valid_data, epochs, hidden_dim)
    end_time = perf_counter()
    print("Train accuracy: ", train_acc)
    print("Validation accuracy: ", valid_acc)
    print(f"Training time:  {end_time-start_time:.4f} seconds")

    #test
    print("\n---------- Testing model -----------\n")
    start_time = perf_counter()
    test_acc = lr_model.test(test_data)
    end_time = perf_counter()
    print("Test accuracy: ", test_acc)
    print(f"Testing time:  {end_time-start_time:.4f} seconds")

    #SVM model
    print("========== Building SVM model ==========\n")
    # initialize model
    svm_model = SVMModel()

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    train_acc, valid_acc = svm_model.train(train_data, valid_data, epochs, hidden_dim)
    end_time = perf_counter()
    print("Train accuracy: ", train_acc)
    print("Validation accuracy: ", valid_acc)
    print(f"Training time:  {end_time - start_time:.4f} seconds")

    # test
    print("\n---------- Testing model -----------\n")
    start_time = perf_counter()
    test_acc = svm_model.test(test_data)
    end_time = perf_counter()
    print("Test accuracy: ", test_acc)
    print(f"Testing time:  {end_time - start_time:.4f} seconds")

    #BERT model
    print("========== Building BERT model ==========\n")
    # initialize model
    bert_model = BERTModel()

    # train
    print("---------- Training model ----------")
    start_time = perf_counter()
    train_acc, valid_acc = bert_model.train(train_data, valid_data, epochs, hidden_dim)
    end_time = perf_counter()
    print("Train accuracy: ", train_acc)
    print("Validation accuracy: ", valid_acc)
    print(f"Training time:  {end_time - start_time:.4f} seconds")

    # test
    print("\n---------- Testing model -----------\n")
    start_time = perf_counter()
    test_acc = bert_model.test(test_data)
    end_time = perf_counter()
    print("Test accuracy: ", test_acc)
    print(f"Testing time:  {end_time - start_time:.4f} seconds")