# Showcase how minigrad can be used to recognize which language a last name is in.
# Based off of https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Download training data from https://download.pytorch.org/tutorial/data.zip
# and put names folder from data.zip next to this script.

import unicodedata
import string

ALL_LETTERS = string.ascii_letters + ".,;'"

def main():
    pass

def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )

if __name__ == "__main__":
    main()

