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

N_ITERS = 40000
ALL_LETTERS = string.ascii_letters + ".,;'"
BETA = 0.99  # For moving loss and acc.

def main():
    names = load_data()
    langs = list(names.keys())
    num_lang = len(langs)
    model = RNN(len(ALL_LETTERS), num_lang)
    training_data = create_training_data(names, langs)
    n_one = Tensor([-1])
    optimizer = optim.Adam(model.params(), lr=1e-7)
    moving_loss, moving_acc = 0, 0
    for iter in (t := trange(N_ITERS)):
        i = np.random.randint(0, len(training_data))
        training_example = training_data[i]
        hidden = Tensor(np.zeros((1, 256), dtype=np.float32))
        for j in range(len(training_example[0])):
            pred, hidden = model.forward(training_example[0][j], hidden)
        true_y = training_example[1]
        loss = pred.log2().mul(true_y).sum().mul(n_one)
        loss.backward() 
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

class RNN:
    def __init__(self, input_size, output_size, hidden_size=256):
        self.w1 = Tensor.rand_init(input_size + hidden_size, hidden_size)
        self.w2 = Tensor.rand_init(hidden_size, output_size)

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
    retval = []
    for lang in names:
        for name in names[lang]:
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

