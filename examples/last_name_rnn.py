# Showcase how minigrad can be used to recognize which language a last name is in.
# Based off of https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Download training data from https://download.pytorch.org/tutorial/data.zip
# and put names folder from data.zip next to this script.

import unicodedata
import string
import os

ALL_LETTERS = string.ascii_letters + ".,;'"

def main():
    names = load_data()

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

