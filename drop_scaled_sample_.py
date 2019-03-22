import os
import shutil 
from glob import glob
import numpy as np
import pandas as pd
import pickle

def data_loader(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data


def scaler(dataframe, col_range_dict=None, feature_range=(-1., 1.)):

    if set(dataframe.columns) - set(col_range_dict):
        raise AttributeError("'col_real_range_dict' should be contains all columns in 'dataframe'")
    else:
        result_frame = dataframe.copy()
        feature_max = feature_range[1]
        feature_min = feature_range[0]

        for col in dataframe.columns:
            real_min, real_max = col_range_dict[col]
            scale = (feature_max-feature_min)/(real_max-real_min)
            scaled_col = (result_frame[col]-real_min) * scale + feature_min
            result_frame[col]= scaled_col
        
        return result_frame


def unscaler(scaled, col_to_unscale, col_range_dict='./data/energy_range_dict.p', feature_range=(-1., 1.)):

    unscaled = scaled
    with open(col_range_dict, 'rb') as f:
        range_dict = pickle.load(f)

    feature_max = feature_range[1]
    feature_min = feature_range[0]

    for i, col in enumerate(col_to_unscale):
        real_min, real_max = range_dict[col]
        scale = (feature_max-feature_min)/(real_max-real_min)
        unscaled_col = (unscaled[:,i] - feature_min) / scale + real_min
        unscaled[:,i] = unscaled_col
    
    return unscaled


if __name__ == "__main__":

    file_path = './data/energydata_complete.csv'
    # col_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'diff', 'before_y']
    col_list = ['Appliances', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 
                 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']

    # start_dt = '1986-03-14'
    # end_dt = '2019-03-05'
    # base_freq = '1D'
    data = data_loader(file_path)
    data_np = data.values
    
    # (8312, 7)
    # (19735, 25)
    print(data_np.shape)

    # mins = np.min(data_np, axis=0)
    # maxs = np.max(data_np, axis=0)
    # col_range_dict = {}
    # for i, col in enumerate(col_list):
    #     col_range_dict[col] = [mins[i], maxs[i]]\
    col_range_dict = {
    "Appliances" : [0, 1200], "T1" : [-5, 30], "RH_1" : [1, 100], "T2" : [-5, 30], "RH_2" : [1, 100], "T3" : [-5, 30], "RH_3" : [1, 100], 
    "T4" : [-5, 30], "RH_4" : [1, 100], "T5" : [-5, 30], "RH_5" : [1, 100], "T6" : [-5, 30], "RH_6" : [1, 100], "T7" : [-5, 30], 
    "RH_7" : [1, 100], "T8" : [-5, 30], "RH_8" : [1, 100], "T9" : [-5, 30], "RH_9" : [1, 100], "T_out" : [-5, 30], 
    "Press_mm_hg" : [729, 775], "RH_out" : [1, 100], "Windspeed" : [0, 14], "Visibility" : [1, 66], "Tdewpoint" : [-6, 16]
    }
    
    scaling_range = (-1., 1.)

    full_scaled = scaler(data, col_range_dict=col_range_dict, feature_range=scaling_range)
    
    full_scaled.to_csv('./data/energy_scaled_data_.csv', encoding='utf-8')
    with open('./data/energy_range_dict.p', 'wb') as f:
        pickle.dump(col_range_dict, f)
    
    scaled = data_loader('./data/energy_scaled_data_.csv')
    col_range_dict = './data/energy_range_dict.p'
    print(len(scaled.values))

    unscaled = unscaler(scaled.values, col_to_unscale=col_list)
    print(unscaled[:5])