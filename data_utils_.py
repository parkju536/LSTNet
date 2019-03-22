import random
import pandas as pd
import numpy as np
import logging
import pickle


logger = logging.getLogger()

def load_data_from_csv(data_path, x_col_list, y_col_list, x_len, y_len, foresight=0, dev_ratio=.1, test_len=7, seed=None):

    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    ndim_x = len(x_col_list)
    ndim_y = len(y_col_list)

    full_x = np.empty((0, x_len, ndim_x), dtype=np.float32)
    full_y = np.empty((0, y_len, ndim_y), dtype=np.float32)
    full_dt = np.empty((0, y_len), dtype=str)

    grouped = data.groupby(pd.Grouper(freq='Y'))
    for year, group in grouped:

        if not group.shape[0]:
            coutinue
        else:
            group_index = group.index.astype('str')

            source_x = group[x_col_list].sort_index().values.reshape(-1, ndim_x).astype('float32')
            source_y = group[y_col_list].sort_index().values.reshape(-1, ndim_y).astype('float32')

            slided_x = np.array([source_x[i:i + x_len] for i in range(0, len(source_x) - x_len - foresight - y_len)])
            y_start_idx = x_len + foresight
            slided_y = np.array([source_y[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len)])
            slided_dt = np.array([group_index[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len)])

            full_x = np.concatenate([full_x, slided_x], axis=0)
            full_y = np.concatenate([full_y, slided_y], axis=0)
            full_dt = np.concatenate([full_dt, slided_dt], axis=0)

    assert len(full_x) == len(full_y)

    # squeeze second dim if y_len = 1

    if y_len == 1:
        full_y = np.squeeze(full_y, axis=1)
        full_dt = np.squeeze(full_dt, axis=1)

    
    tr_x = full_x[:-test_len]
    tr_y = full_y[:-test_len]

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x, test_x = tr_x[tr_ind], tr_x[dev_ind], full_x[-test_len:]
    train_y, dev_y, test_y = tr_y[tr_ind], tr_y[dev_ind], full_y[-test_len:]
    k = 7
    train_x = train_x[:, :, :k]
    dev_x = dev_x[:, :, :k]
    test_x = test_x[:, :, :k]
    
    train_y = train_y[:,6]
    dev_y = dev_y[:,6]
    # 'close' values for 1:end
    test_y_before = test_y[:,7]
    test_y = test_y[:,6]
    print("train_x shape:", train_x.shape)
    print("test_y shape:",test_y.shape)
    print("test_y_before shape:",test_y_before.shape)
        
    test_dt = full_dt[-test_len:]
    
    return train_x, dev_x, test_x, train_y, dev_y, test_y, test_dt, test_y_before
    
    
def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:min(length, idx + batch_size)]


if __name__=="__main__":
    COL_LIST = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'diff', 'before_y']
    data_path = './data/MS_scaled_diff_data_.csv'
    x_col_list = COL_LIST
    y_col_list = COL_LIST
    x_len = 3
    y_len = 1
    
    train_x, dev_x, test_x, train_y, dev_y, test_y_, test_dt, test_y_before = load_data_from_csv(data_path, x_col_list, y_col_list, x_len, y_len, foresight=0, dev_ratio=.1, test_len=7, seed=None)


