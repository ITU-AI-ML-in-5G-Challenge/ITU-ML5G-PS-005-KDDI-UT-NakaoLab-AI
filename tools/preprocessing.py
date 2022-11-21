import numpy as np
import pandas as pd

class Processor:
  def __init__(self, training_data, test_data):
    self.train_data = training_data.iloc[:,2:]
    self.test_data = test_data.iloc[:,2:]
    
    self.train_label = training_data.label
    self.test_label = test_data.label
    
    self.train_data.drop(self.train_data.columns[self.train_data.columns.str.contains('name')], axis=1, inplace=True)
    self.test_data.drop(self.test_data.columns[self.test_data.columns.str.contains('name')], axis=1, inplace=True)
    
    self.max = np.max(self.train_data)
    self.min = np.min(self.train_data)
    
    self.scaled_train = (self.train_data - self.min) / (self.max - self.min)
    self.scaled_test = (self.test_data - self.min) / (self.max - self.min)
    
    self.scaled_train[self.scaled_train.isnull()] = 0
    self.scaled_test[self.scaled_test.isnull()] = 0
    
    self.train_Y = self.scaled_train["amf.amf.app.five-g.RM.RegInitFail"]
    self.test_Y = self.scaled_test["amf.amf.app.five-g.RM.RegInitFail"]
    
    self.train_X = self.scaled_train.drop("amf.amf.app.five-g.RM.RegInitFail", axis=1)
    self.test_X = self.scaled_test.drop("amf.amf.app.five-g.RM.RegInitFail", axis=1)
    
  def get_scaled_data(self):
    return self.train_X, self.train_Y, self.test_X, self.test_Y 
  
  def _calc_diff(self):
    normal_mean = np.mean(self.train_X[self.train_label=="normal"])
    loss_mean = np.mean(self.train_X[self.train_label=="br-cp_bridge-loss-congestion-with-time-start"])
    return list(loss_mean[loss_mean>normal_mean*2].index)
  
  def get_diff_data(self):
    diff_columns = self._calc_diff()
    return self.train_X[diff_columns], self.train_Y, self.test_X[diff_columns], self.test_Y
    
    