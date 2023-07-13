# Showcase how minigrad can be used to recognize which language a last name is in.
# Based off of https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Download training data from https://download.pytorch.org/tutorial/data.zip
# and put names folder from data.zip next to this script.

import unicodedata
import string
import os
from minigrad.tensor import Tensor

ALL_LETTERS = string.ascii_letters + ".,;'"

def main():
    names = load_data()
    num_lang = len(names)
    rnn = RNN(len(ALL_LETTERS), num_lang)

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

if __name__ == "__main__":
    main()

