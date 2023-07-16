# Showcase how minigrad can be used to create attention models.

# Task: train an attention model to translate dates into standard format.
# Example input: saturday december 10 1994
# Expected output: 1994-12-10
# Example input: feb 5 2002
# Expected output: 2002-02-05

# pip3 install Faker
# pip3 install Babel

from faker import Faker
from babel.dates import format_date
import random
import numpy as np
from minigrad.tensor import Tensor
import minigrad.optim as optim
from tqdm import trange

FK_FORMATS = ["short",
              "medium", "medium", "medium",
              "long", "long", "long", "long", "long",
              "full", "full", "full",
              "d MMM YYY",
              "d MMMM YYY", "d MMMM YYY", "d MMMM YYY", "d MMMM YYY", "d MMMM YYY",
              "dd/MM/YYY",
              "EE d, MMM YYY",
              "EEEE d, MMMM YYY"]
TOP_H_SIZE = 128  # Hidden size for top LSTM.
BOT_H_SIZE = 64  # Hidden size for bottom LSTMs.
ATT_H_SIZE = 32  # Hidden size for small attention NN.
HUMAN_VOCAB = [" ", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
MACHINE_VOCAB = ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
N_ITERS = 5000
BETA = 0.99  # For moving loss and acc.

def main():
    faker = Faker()
    x_train, y_train = gen_dataset(100000, faker)
    # Create two bottom LSTMs for bidirectional.
    rightward = BOT_LSTM(len(HUMAN_VOCAB))
    leftward = BOT_LSTM(len(HUMAN_VOCAB))
    small_nn = ATT_NN()
    main_lstm = TOP_LSTM(len(MACHINE_VOCAB))
    rightward_optim = optim.Adam(rightward.params(), lr=LEARNING_RATE)
    leftward_optim = optim.Adam(leftward.params(), lr=LEARNING_RATE)
    small_nn_optim = optim.Adam(small_nn.params(), lr=LEARNING_RATE)
    main_lstim_optim = optim.Adam(main_lstm.params(), lr=LEARNING_RATE)
    n_one = Tensor([-1])
    moving_loss, moving_acc = 0, 0
    for iter in (t := trange(N_ITERS)):
        for b in range(BATCH_SIZE):
            rand_i = np.random.randint(0, len(x_train))
            x = cvt_to_tensors(x_train[rand_i], HUMAN_VOCAB)
            y = cvt_to_tensors(y_train[rand_i], MACHINE_VOCAB)
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
            loss = Tensor.zeros(1)
            acc = 0
            for i in range(10):  # Top LSTM will always generate 10 outputs because isoformat.
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
                # Accumulate accuracy.
                if np.argmax(pred.data) == np.argmax(y[i].data):
                    acc += 0.1
                # Compare pred to true value to compute loss.
                this_loss = pred.log2().mul(y[i]).sum().mul(n_one)
                loss = loss.add(this_loss)
                top_prev_pred = pred
            # Backprop.
            loss.backward(want_zero_grads=b == 0)
            # Record loss and acc.
            moving_loss = BETA * moving_loss + (1 - BETA) * loss.data[0]
            moving_acc = BETA * moving_acc + (1 - BETA) * acc
        rightward_optim.step()
        leftward_optim.step()
        small_nn_optim.step()
        main_lstim_optim.step()
        # Print stats.
        t.set_description("Loss: %.5f | Accuracy: %.2f" % (moving_loss, moving_acc))
    # Save model.
    print("Saving model.")
    import pickle, os
    if not os.path.isdir("dates_attention_trained"):
        os.mkdir("dates_attention_trained")
    pickle.dump(rightward, open("dates_attention_trained/rightward.pickle", "wb"))
    pickle.dump(leftward, open("dates_attention_trained/leftward.pickle", "wb"))
    pickle.dump(small_nn, open("dates_attention_trained/small_nn.pickle", "wb"))
    pickle.dump(main_lstm, open("dates_attention_trained/main_lstm.pickle", "wb"))
    print("Saved model.")

def cvt_to_tensors(string, vocab):
    retval = []
    for char in string:
        retval.append(cvt_to_tensor(char, vocab))
    return retval

def cvt_to_tensor(char, vocab): 
    t = np.zeros((1, len(vocab)), dtype=np.float32)
    t[0, vocab.index(char)] = 1
    t = Tensor(t)
    return t
    
class ATT_NN:
    def __init__(self):
        self.w1 = Tensor.rand_init(TOP_H_SIZE + 2*BOT_H_SIZE, ATT_H_SIZE)
        self.b1 = Tensor.rand_init(1, ATT_H_SIZE)
        self.w2 = Tensor.rand_init(ATT_H_SIZE, 1)
        self.b2 = Tensor.rand_init(1, 1)

    def params(self):
        return [self.w1, self.b1,
                self.w2, self.b2]

    def forward(self, top_hidden, bot_activation):
        combined = top_hidden.cat(bot_activation)
        e = combined.matmul(self.w1).add(self.b1).relu().matmul(self.w2).add(self.b2)
        return e

class TOP_LSTM:
    def __init__(self, output_size):
        self.w_c = Tensor.rand_init(output_size + 2*BOT_H_SIZE + TOP_H_SIZE, TOP_H_SIZE)
        self.w_u = Tensor.rand_init(output_size + 2*BOT_H_SIZE + TOP_H_SIZE, TOP_H_SIZE)
        self.w_f = Tensor.rand_init(output_size + 2*BOT_H_SIZE + TOP_H_SIZE, TOP_H_SIZE)
        self.w_o = Tensor.rand_init(output_size + 2*BOT_H_SIZE + TOP_H_SIZE, TOP_H_SIZE)
        self.b_c = Tensor.rand_init(1, TOP_H_SIZE)
        self.b_u = Tensor.rand_init(1, TOP_H_SIZE)
        self.b_f = Tensor.rand_init(1, TOP_H_SIZE)
        self.b_o = Tensor.rand_init(1, TOP_H_SIZE)
        self.w_pred = Tensor.rand_init(TOP_H_SIZE, output_size)
        self.b_pred = Tensor.rand_init(1, output_size)
        
    def params(self):
        return [self.w_c, self.w_u, self.w_f, self.w_o,
                self.b_c, self.b_u, self.b_f, self.b_o,
                self.w_pred, self.b_pred]

    def forward(self, prev_output, bot_activation, prev_memory, prev_activation):
        combined = prev_output.cat(bot_activation).cat(prev_activation)
        c_tilde = combined.matmul(self.w_c).add(self.b_c).tanh()
        update_gate = combined.matmul(self.w_u).add(self.b_u).sigmoid()
        forget_gate = combined.matmul(self.w_f).add(self.b_f).sigmoid()
        output_gate = combined.matmul(self.w_o).add(self.b_o).sigmoid()
        new_memory = update_gate.mul(c_tilde).add(forget_gate.mul(prev_memory))
        new_activation = new_memory.tanh().mul(output_gate)
        pred = new_activation.matmul(self.w_pred).add(self.b_pred).softmax(dim=1)
        return pred, new_memory, new_activation

class BOT_LSTM:
    def __init__(self, input_size):
        self.w_c = Tensor.rand_init(input_size + BOT_H_SIZE, BOT_H_SIZE)
        self.w_u = Tensor.rand_init(input_size + BOT_H_SIZE, BOT_H_SIZE)
        self.w_f = Tensor.rand_init(input_size + BOT_H_SIZE, BOT_H_SIZE)
        self.w_o = Tensor.rand_init(input_size + BOT_H_SIZE, BOT_H_SIZE)
        self.b_c = Tensor.rand_init(1, BOT_H_SIZE)
        self.b_u = Tensor.rand_init(1, BOT_H_SIZE)
        self.b_f = Tensor.rand_init(1, BOT_H_SIZE)
        self.b_o = Tensor.rand_init(1, BOT_H_SIZE)
        
    def params(self):
        return [self.w_c, self.w_u, self.w_f, self.w_o,
                self.b_c, self.b_u, self.b_f, self.b_o]

    def forward(self, input, prev_memory, prev_activation):
        combined = prev_activation.cat(input)
        c_tilde = combined.matmul(self.w_c).add(self.b_c).tanh()
        update_gate = combined.matmul(self.w_u).add(self.b_u).sigmoid()
        forget_gate = combined.matmul(self.w_f).add(self.b_f).sigmoid()
        output_gate = combined.matmul(self.w_o).add(self.b_o).sigmoid()
        new_memory = update_gate.mul(c_tilde).add(forget_gate.mul(prev_memory))
        new_activation = new_memory.tanh().mul(output_gate)
        return new_memory, new_activation

def gen_dataset(m, faker):
    xs, ys = [], []
    for i in range(m):
        x, y = random_date(faker)
        xs.append(x)
        ys.append(y)
    return xs, ys

def random_date(faker):
    dt = faker.date_object()
    date = format_date(dt, format=random.choice(FK_FORMATS), locale="en")
    human_format = date.lower().replace(",", "")
    machine_format = dt.isoformat()
    return human_format, machine_format

if __name__ == "__main__":
    main()

