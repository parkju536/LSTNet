import random


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:min(length, idx + batch_size)]