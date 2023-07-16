# Test script for dates_attention_train.py

import pickle
from dates_attention_train import ATT_NN, TOP_LSTM, BOT_LSTM

def main():
    rightward = pickle.load(open("dates_attention_trained/rightward.pickle", "rb"))
    leftward = pickle.load(open("dates_attention_trained/leftward.pickle", "rb"))
    small_nn = pickle.load(open("dates_attention_trained/small_nn.pickle", "rb"))
    main_lstm = pickle.load(open("dates_attention_trained/main_lstm.pickle", "rb"))

if __name__ == "__main__":
    main()

