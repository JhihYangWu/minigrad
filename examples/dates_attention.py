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

FK_FORMATS = ["short",
              "medium", "medium", "medium",
              "long", "long", "long", "long", "long",
              "full", "full", "full",
              "d MMM YYY",
              "d MMMM YYY", "d MMMM YYY", "d MMMM YYY", "d MMMM YYY", "d MMMM YYY",
              "dd/MM/YYY",
              "EE d, MMM YYY",
              "EEEE d, MMMM YYY"]
BOT_H_SIZE = 128  # Hidden size for bottom LSTMs.

def main():
    faker = Faker()
    x_train, y_train = gen_dataset(50000, faker)
    x_test, y_test = gen_dataset(1000, faker)

class BOTTOM_LSTM:
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

