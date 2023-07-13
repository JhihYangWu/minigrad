# Showcase how minigrad can be used to recognize which language a last name is in.
# Based off of https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Download training data from https://download.pytorch.org/tutorial/data.zip
# and put names folder from data.zip next to this script.

import unicodedata
import string
import os
from minigrad.tensor import Tensor
import minigrad.optim as optim
import random
import numpy as np
from tqdm import trange

N_ITERS = 500
ALL_LETTERS = string.ascii_letters + ".,;'"
BETA = 0.99  # For moving loss and acc.
BATCH_SIZE = 256
HIDDEN_SIZE = 256

def main():
    names = load_data()
    langs = list(names.keys())
    num_lang = len(langs)
    model = RNN(len(ALL_LETTERS), num_lang)
    training_data = create_training_data(names, langs)
    n_one = Tensor([-1])
    optimizer = optim.Adam(model.params(), lr=1e-3)
    moving_loss, moving_acc = 0, 0
    for iter in (t := trange(N_ITERS)):
        g_w1s = np.zeros((BATCH_SIZE,) + model.w1.data.shape, np.float32)
        g_w2s = np.zeros((BATCH_SIZE,) + model.w2.data.shape, np.float32)
        for b in range(BATCH_SIZE):
            i = np.random.randint(0, len(training_data))
            training_example = training_data[i]
            hidden = Tensor(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))
            for j in range(len(training_example[0])):
                pred, hidden = model.forward(training_example[0][j], hidden)
            true_y = training_example[1]
            loss = pred.log2().mul(true_y).sum().mul(n_one)
            loss.backward() 
            g_w1s[b] = model.w1.grad.data
            g_w2s[b] = model.w2.grad.data
        g_w1 = g_w1s.sum(axis=0)
        g_w2 = g_w2s.sum(axis=0)
        model.w1.grad.data = g_w1
        model.w2.grad.data = g_w2
        optimizer.step()
        # Print stats.
        pred_label = np.argmax(pred.data)
        actual_label = np.argmax(true_y.data)
        acc = 1 if pred_label == actual_label else 0
        moving_loss = BETA * moving_loss + (1 - BETA) * loss.data[0]
        moving_acc = BETA * moving_acc + (1 - BETA) * acc
        moving_loss_corr = moving_loss / (1 - BETA ** (1+iter))
        moving_acc_corr = moving_acc / (1 - BETA ** (1+iter))
        t.set_description("Loss: %.5f | Accuracy: %.2f" % (moving_loss_corr, moving_acc_corr))
    # Play with model.
    while True:
        user_input = ""
        while user_input == "":
            user_input = input("Enter a last name: ")
            user_input = unicode_to_ascii(user_input.strip().lower().capitalize())
        tensors = []
        for c in user_input:
            t = np.zeros((1, len(ALL_LETTERS)), dtype=np.float32)
            t[0, ALL_LETTERS.index(c)] = 1
            t = Tensor(t)
            tensors.append(t)
        hidden = Tensor(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))
        for t in tensors:
            pred, hidden = model.forward(t, hidden)
        pred = pred.data[0]
        print(f"Prediction for {user_input}: ", end="")
        for _ in range(3):
            i = np.argmax(pred)
            conf = round(pred[i] * 100, 2)
            print(f"{langs[i]} {conf}% | ", end="")
            pred[i] = -1
        print()
        print()

class RNN:
    def __init__(self, input_size, output_size):
        self.w1 = Tensor.rand_init(input_size + HIDDEN_SIZE, HIDDEN_SIZE)
        self.w2 = Tensor.rand_init(HIDDEN_SIZE, output_size)

    def params(self):
        return [self.w1, self.w2]

    def forward(self, input, hidden):
        combined = input.cat(hidden)
        hidden = combined.matmul(self.w1)
        output = hidden.matmul(self.w2).softmax(dim=1)
        return output, hidden

def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )

def load_data():
    names = {}
    for file in os.listdir("names/"):
        if file.endswith(".txt"):
            language = file[:-4]
            lines = open("names/" + file, encoding="utf-8").read().strip().split("\n")
            names[language] = [unicode_to_ascii(n) for n in lines]
    return names

def create_training_data(names, langs):
    # Find minimum number of last names for balancing.
    min_num = None
    for lang in names:
        if min_num is None:
            min_num = len(names[lang])
        else:
            min_num = min(min_num, len(names[lang]))
    retval = []
    for lang in names:
        random.shuffle(names[lang])
        for name in names[lang][:min_num]:
            char_tensors = []
            for c in name:
                t = np.zeros((1, len(ALL_LETTERS)), dtype=np.float32)
                t[0, ALL_LETTERS.index(c)] = 1
                t = Tensor(t)
                char_tensors.append(t)
            output_tensor = np.zeros((1, len(langs)), dtype=np.float32)
            output_tensor[0, langs.index(lang)] = 1
            output_tensor = Tensor(output_tensor)
            retval.append((char_tensors, output_tensor))
    random.shuffle(retval)
    return retval

if __name__ == "__main__":
    main()

