# Showcase how minigrad can be used to recognize which language a last name is in.
# Based off of https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Download training data from https://download.pytorch.org/tutorial/data.zip
# and put names folder from data.zip next to this script.

import unicodedata
import string
import os
from minigrad.tensor import Tensor
import random
import numpy as np

ALL_LETTERS = string.ascii_letters + ".,;'"

def main():
    names = load_data()
    langs = list(names.keys())
    num_lang = len(langs)
    rnn = RNN(len(ALL_LETTERS), num_lang)
    training_data = create_training_data(names, langs)

class RNN:
    def __init__(self, input_size, output_size):
        self.w1 = Tensor.rand_init(input_size + 128, 128)
        self.w2 = Tensor.rand_init(128, output_size)

    def params(self):
        return [self.w1, self.w2]

    def forward(self, input, hidden):
        combined = input.cat(hidden)
        hidden = combined.matmul(self.w1)
        output = hidden.matmul(self.w2).softmax()
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

