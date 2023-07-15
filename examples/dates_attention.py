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

def main():
    faker = Faker()
    x_train, y_train = gen_dataset(50000, faker)
    x_test, y_test = gen_dataset(1000, faker)

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

