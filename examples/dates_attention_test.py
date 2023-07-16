# Test script for dates_attention_train.py

import pickle
from dates_attention_train import ATT_NN, TOP_LSTM, BOT_LSTM, random_date, HUMAN_VOCAB, MACHINE_VOCAB, cvt_to_tensors, BOT_H_SIZE, TOP_H_SIZE
from minigrad.tensor import Tensor
from faker import Faker
import numpy as np
import time

def main():
    rightward = pickle.load(open("dates_attention_trained/rightward.pickle", "rb"))
    leftward = pickle.load(open("dates_attention_trained/leftward.pickle", "rb"))
    small_nn = pickle.load(open("dates_attention_trained/small_nn.pickle", "rb"))
    main_lstm = pickle.load(open("dates_attention_trained/main_lstm.pickle", "rb"))
    faker = Faker()
    while True:
        human_str, expected_str = random_date(faker)
        print("Input           :", human_str)
        print("Expected  Output:", expected_str)
        print("Predicted Output: ", end="", flush=True)
        
        # Run attention model.
        x = cvt_to_tensors(human_str, HUMAN_VOCAB)
        right_mem = Tensor.zeros(1, BOT_H_SIZE)
        right_act = Tensor.zeros(1, BOT_H_SIZE)
        left_mem = Tensor.zeros(1, BOT_H_SIZE)
        left_act = Tensor.zeros(1, BOT_H_SIZE)
        # Compute bottom activations.
        bot_activations = [[None, None] for _ in range(len(x))]
        for i in range(len(x)):
            right_mem, right_act = rightward.forward(x[i], right_mem, right_act)
            left_mem, left_act = leftward.forward(x[-1-i], left_mem, left_act)
            bot_activations[i][0] = right_act
            bot_activations[-1-i][1] = left_act
        # Combine bot_activations.
        bot_activations = [pair[0].cat(pair[1]) for pair in bot_activations]
        # Start running top LSTM.
        top_mem = Tensor.zeros(1, TOP_H_SIZE)
        top_act = Tensor.zeros(1, TOP_H_SIZE)
        top_prev_pred = Tensor.zeros(1, len(MACHINE_VOCAB))
        for i in range(10):  # Top LSTM will always generate 10 outputs because isoformat.
            time.sleep(0.1)
            # Calc attention for each of bottom activations.
            attentions = []
            sum_attentions = Tensor.zeros(1, 1)
            for j in range(len(bot_activations)):
                attentions.append(small_nn.forward(top_act, bot_activations[j]).exp2())
                sum_attentions = sum_attentions.add(attentions[-1])
            # Use attentions and bot_activations to create single bot_activation.
            bot_activation = Tensor.zeros(1, 2*BOT_H_SIZE)
            for j in range(len(bot_activations)):
                bot_activation = bot_activation.add(attentions[j].div(sum_attentions).broadcast_to(shape=bot_activations[j].shape).mul(bot_activations[j]))
            # Output a tensor.
            pred, top_mem, top_act = main_lstm.forward(top_prev_pred, bot_activation, top_mem, top_act)
            print(MACHINE_VOCAB[np.argmax(pred.data)], end="", flush=True)
            top_prev_pred = pred
        # Done running attention model.

        print()
        print()
        time.sleep(1)

if __name__ == "__main__":
    main()

