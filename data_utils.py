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

    end_dt = pd.to_datetime(full_dt[-1]).date()
    start_dt = end_dt - pd.Timedelta(days=test_len)

    if y_len > 1:
        start_dt = start_dt.astype('str')[0]

    # str(start_dt)

    test_ind = -1
    for i in range(len(full_dt)):
        if y_len == 1:
            d = full_dt[i]
        else:
            d = full_dt[i, 0]

        if d == str(start_dt):
            test_ind = i + 1
            break

    assert test_ind != -1

    tr_x = full_x[:test_ind]
    tr_y = full_y[:test_ind]

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x, test_x = tr_x[tr_ind], tr_x[dev_ind], full_x[test_ind:]
    train_y, dev_y, test_y = tr_y[tr_ind], tr_y[dev_ind], full_y[test_ind:]
    # print(train_y.shape)
    # print(train_y[:,6].shape)
    print(train_x.shape)
    train_x = train_x[:, :, :6]
    dev_x = dev_x[:, :, :6]
    test_x = test_x[:, :, :6]
    # train_x = train_x[:, :, 7]
    # dev_x = dev_x[:, :, 7]
    # test_x = test_x[:, :, 7]
    # train_x = np.expand_dims(train_x, axis=-1)
    # dev_x = np.expand_dims(dev_x, axis=-1)
    # test_x = np.expand_dims(test_x, axis=-1)
    print(train_x.shape)
    
    train_y = train_y[:,6] * -1.0
    dev_y = dev_y[:,6] * -1.0
    test_y_ = test_y[:,6] * -1.0
    print(test_y_.shape)
    
    #exit()
    test_y_before = test_y[:,7]


    test_dt = full_dt[test_ind:]
    
    return train_x, dev_x, test_x, train_y, dev_y, test_y_, test_dt, test_y_before


def load_energy_data_from_csv(data_path, x_col_list, y_col_list, x_len, y_len, foresight=0, dev_ratio=.1, test_len=7, seed=None):

    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    ndim_x = len(x_col_list)
    ndim_y = len(y_col_list)

    full_x = np.empty((0, x_len, ndim_x), dtype=np.float32)
    full_y = np.empty((0, y_len, ndim_y), dtype=np.float32)
    full_dt = np.empty((0, y_len), dtype=str)

    grouped = data.groupby(pd.Grouper(freq='D'))
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

    end_dt = pd.to_datetime(full_dt[-1])
    start_dt = end_dt - pd.Timedelta(days=test_len)

    if y_len > 1:
        start_dt = start_dt.astype('str')[0]

    # str(start_dt)

    test_ind = -1
    for i in range(len(full_dt)):
        if y_len == 1:
            d = full_dt[i]
        else:
            d = full_dt[i, 0]

        if d == str(start_dt):
            test_ind = i + 1
            break

    assert test_ind != -1

    tr_x = full_x[:test_ind]
    tr_y = full_y[:test_ind]

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x, test_x = tr_x[tr_ind], tr_x[dev_ind], full_x[test_ind:]
    train_y, dev_y, test_y = tr_y[tr_ind], tr_y[dev_ind], full_y[test_ind:]
    
    # train_x = train_x[:, :, 1:]
    # dev_x = dev_x[:, :, 1:]
    # test_x = test_x[:, :, 1:]

    train_cur_x = train_y[:,1:]
    dev_cur_x = dev_y[:,1:]
    test_cur_x = test_y[:,1:]

    train_y = train_y[:,0]
    dev_y = dev_y[:,0]
    test_y_ = test_y[:,0]
    
    print("train_x.shape", train_x.shape)
    print("test_y_.shape", test_y_.shape)
    print("train_cur_x.shape", train_cur_x.shape)

    test_dt = full_dt[test_ind:]
    
    return train_x, dev_x, test_x, train_y, dev_y, test_y_, train_cur_x, dev_cur_x, test_cur_x, test_dt

def load_data_mem(data_path, x_col_list, y_col_list, x_len, y_len, mem_len, foresight=0, dev_ratio=.1, test_len=7, seed=None):
    
    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    ndim_x = len(x_col_list)
    ndim_y = len(y_col_list)

    m_lst = []
    y_lst = []
    dt_lst = []

    source_x = data[x_col_list].values.reshape(-1, ndim_x).astype('float32')
    source_y = data[y_col_list].values.reshape(-1, ndim_y).astype('float32')
    data_index = data.index.astype('str')

    # sliding total data
    # slided_x : [time, x_len, dim]
    mem_len  = mem_len * x_len
    input_len = x_len + mem_len
    total_x = np.array([source_x[i:i + x_len] for i in range(mem_len, len(source_x) - x_len - foresight - y_len+1)])
    total_m = np.array([source_x[i:i + mem_len] for i in range(0, len(source_x) - input_len - foresight - y_len+1)])

    y_start_idx = input_len + foresight
    total_y = np.array([source_y[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len+1)])

    # datetime for y
    total_dt = np.array([data_index[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len+1)])

    # 
    print(total_x.shape)
    print(total_y.shape)
    print(total_m.shape)

    if y_len == 1:
        total_y = np.squeeze(total_y, axis=1)
        total_dt = np.squeeze(total_dt, axis=1)
    
    end_dt = pd.to_datetime(total_dt[-1]).date()
    start_dt = end_dt - pd.Timedelta(days=test_len)

    if y_len > 1:
        start_dt = start_dt.astype('str')[0]

    #print(str(start_dt))
    
    test_ind = -1
    for i in range(len(total_dt)):
        if y_len == 1:
            d = total_dt[i]
        else:
            d = total_dt[i, 0]

        if d == str(start_dt):
            test_ind = i + 1
            break

    assert test_ind != -1

    tr_x = total_x[:test_ind]
    tr_m = total_m[:test_ind]
    tr_y = total_y[:test_ind]

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x, test_x = tr_x[tr_ind], tr_x[dev_ind], total_x[test_ind:]
    train_y, dev_y, test_y = tr_y[tr_ind], tr_y[dev_ind], total_y[test_ind:]
    train_m, dev_m, test_m = tr_m[tr_ind], tr_m[dev_ind], total_m[test_ind:]

    print(train_y.shape)
    print(train_y[:,6].shape)

    train_y = train_y[:,6]
    dev_y = dev_y[:,6]
    test_y = test_y[:,6]
    test_dt = total_dt[test_ind:]
    
    return train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt
    
    
def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:min(length, idx + batch_size)]


if __name__=="__main__":
    COL_LIST = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data_path = './data/MS_scaled_data.csv'
    x_col_list = COL_LIST
    y_col_list = COL_LIST
    x_len = 3
    y_len = 1
    mem_len = 10
    train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = load_data_mem(data_path, x_col_list, y_col_list, x_len, y_len, mem_len, foresight=0, dev_ratio=.1, test_len=50, seed=None)


