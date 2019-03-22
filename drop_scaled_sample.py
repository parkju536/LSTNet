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

# def datetime_setter(data, start_dt=None, end_dt=None, freq='1D'):
    
#     full_datetime = pd.date_range(start=str(start_dt), end=str(end_dt), freq=base_freq)
#     selected_datetime = full_datetime
#     # selected_datetime = full_datetime[(int(start_hour) <= full_datetime.hour) & (full_datetime.hour < int(end_hour))]
    
#     tmp_table = pd.DataFrame([], index=selected_datetime)
    
#     joined = tmp_table.join(data, how='left', sort=True)
    
#     return joined

class RangeScaler(object):
    def __init__(self, real_range=(-10., 10.), feature_range=(0., 1.), copy=True):
       
        self.base_range = np.array((0., 1.))
        self.real_range = np.array(real_range, dtype=np.float32)
        self.feature_range = np.array(feature_range, dtype=np.float32)
        self.copy = copy
        
        self.real_min, self_real_max = real_range
        self.real_length = real_range[1] - real_range[0]
        
        self.feature_min, self.feature_max = feature_range
        self.feature_length = feature_range[1] - feature_range[0]
        self.feature_mean = np.mean(feature_range)
        
    def _check_array(self, input_x):
       
        if isinstance(input_x, np.ndarray):
            return input_x
        elif isinstance(input_x, (pd.Series, pd.DataFrame)):
            return input_x.values
        else:
            raise TypeError("Input must be one of this")
    
    def _partial_scale(self, input_x):
        
        input_x = self._check_array(input_x)
        base_ranged = ((input_x - self.real_min) / self.real_length)
        
        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.feature_length) + self.feature_min
        
    def transform(self, input_x):
        
        return self._partial_scale(input_x)
    
    def invers_transform(self, scaled_x):
        
        base_ranged = (scaled_x, self.feature_min) / self.feature_length
        
        if np.array_equal(self.feature_range, self.base_range):
            return base_ranged
        else:
            return (base_ranged * self.real_lenegth) + self.real_min
        
def column_range_scaler(dataframe, col_real_range_dict=None, feature_range=(-1., 1.), use_clip=False):
    
    if set(dataframe.columns) - set(col_range_dict):
        raise AttributeError("'col_real_range_dict' should be contains all columns in 'dataframe'")
    else:
        result_frame = dataframe.copy()
        scaler_dict = dict()
        for col in dataframe.columns:
            col_real_range = col_real_range_dict[col]
            
            scaler = RangeScaler(real_range=col_real_range, feature_range=feature_range)
            scaled = scaler.transform(result_frame[[col]])

            scaler_dict[col] = scaler

            if result_frame[col].isnull().all():
                result_frame[col] = np.mean(feature_range)
            else:
                result_frame[col]= scaled
        
        return result_frame, scaler_dict

    
file_path = './data/MSFT_2014-2019.csv'
col_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

start_dt = '2014-03-06'
end_dt = '2019-03-06'
# start_hour 
# end_hour
# resample X
base_freq = '1D'
'''
2014-2019
col_range_dict = {
    'Open' : [37, 116],
    'High' : [38, 117],
    'Low' : [37, 115],
    'Close' : [37, 116],
    'Adj Close' : [33, 114],
    'Volume' : [7425600,29083900],
}
'''
col_range_dict = {
    'Open' : [15, 115],
    'High' : [16, 117],
    'Low' : [14, 113],
    'Close' : [15, 115],
    'Adj Close' : [11, 114],
    'Volume' : [17686996,879723200],
}
scaling_range = (-1., 1.)

data = data_loader(file_path)
data
# selected = datetime_setter(data, start_dt=start_dt, end_dt=end_dt, freq=base_freq)
# selected
# # Column Scaling : Range
full_scaled, full_scaler_dict = column_range_scaler(data, col_real_range_dict=col_range_dict, feature_range=scaling_range)

full_scaled.to_csv('./data/MS_1999-2019_weekly_scaled_data.csv', encoding='utf-8')