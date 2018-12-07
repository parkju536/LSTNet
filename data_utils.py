import random
import logging

COL_LIST = None


logger = logging.getLogger()


def load_data_from_csv(
        data_path = "data.csv",
        x_col_list = COL_LIST,
        y_col_list = COL_LIST,
        x_len = 5,
        y_len = 1,
        foresight = 1,
        dev_ratio = 0.1,
        test_ratio = 0.1,
        seed = None,
):
    train_x = 0
    dev_x = 0
    test_x = 0
    train_y = 0
    dev_y = 0
    test_y = 0

    return train_x, dev_x, test_x, train_y, dev_y, test_y


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:min(length, idx + batch_size)]



