import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

class Processor:
  def __init__(self, training_data, test_data):
    self.train_data = training_data.iloc[:,2:]
    self.test_data = test_data.iloc[:,2:]
    
    self.train_label = training_data.label
    self.test_label = test_data.label
    
    self.train_data.drop(self.train_data.columns[self.train_data.columns.str.contains('name')], axis=1, inplace=True)
    self.test_data.drop(self.test_data.columns[self.test_data.columns.str.contains('name')], axis=1, inplace=True)
    
    scaler = MinMaxScaler()

    scaler.fit(self.train_data)
    self.scaled_train = scaler.transform(self.train_data)
    self.scaled_test = scaler.transform(self.test_data)
    
    self.scaled_train[np.isnan(self.scaled_train)] = 0
    self.scaled_test[np.isnan(self.scaled_test)] = 0
    
    self.scaled_train = pd.DataFrame(self.scaled_train, columns=self.train_data.columns)
    self.scaled_test = pd.DataFrame(self.scaled_test, columns=self.test_data.columns)
    
    self.train_Y = self.scaled_train["amf.amf.app.five-g.RM.RegInitFail"]
    self.test_Y = self.scaled_test["amf.amf.app.five-g.RM.RegInitFail"]
    
    self.train_X = self.scaled_train.drop("amf.amf.app.five-g.RM.RegInitFail", axis=1)
    self.test_X = self.scaled_test.drop("amf.amf.app.five-g.RM.RegInitFail", axis=1)
    
  def get_scaled_data(self):
    return self.train_X, self.train_Y, self.test_X, self.test_Y 
  
  def get_removed_data(self):
    train_sum = self.train_X.sum()
    train_zero = set(train_sum[train_sum==0].index)
    removed_train_X = self.train_X.drop(train_zero, axis=1, inplace=False)
    removed_test_X = self.test_X.drop(train_zero, axis=1, inplace=False)
    return removed_train_X, self.train_Y, removed_test_X, self.test_Y
  
  def _calc_diff(self):
    normal_mean = np.mean(self.train_X[self.train_label=="normal"])
    loss_mean = np.mean(self.train_X[self.train_label=="br-cp_bridge-loss-congestion-with-time-start"])
    return list(loss_mean[loss_mean>normal_mean*2].index)
  
  def get_diff_data(self):
    diff_columns = self._calc_diff()
    return self.train_X[diff_columns], self.train_Y, self.test_X[diff_columns], self.test_Y

  def get_RF_data(self, metrics_num=500):
    train_X_smf = self.train_X.drop('smf.smf.app.five-g.SM.PduSessionCreationFailNSI', axis=1)
    test_X_smf = self.test_X.drop('smf.smf.app.five-g.SM.PduSessionCreationFailNSI', axis=1)
    with open('data/feature_importance.pkl', 'rb') as f:
      features = pickle.load(f)
    indices = np.argsort(features)[::-1]
    indices = indices[indices<metrics_num-1]
    train_RF = train_X_smf.iloc[:,indices].copy()
    test_RF = test_X_smf.iloc[:,indices].copy()
    train_RF.loc[:,'smf.smf.app.five-g.SM.PduSessionCreationFailNSI'] = self.train_X['smf.smf.app.five-g.SM.PduSessionCreationFailNSI'].values
    test_RF.loc[:,'smf.smf.app.five-g.SM.PduSessionCreationFailNSI'] = self.test_X['smf.smf.app.five-g.SM.PduSessionCreationFailNSI'].values
    return train_RF, self.train_Y, test_RF, self.test_Y    
    
  def get_cadvisor_data(self):
    cad_train_X = self.train_X.loc[:,~self.train_X.columns.str.contains("infra")]
    cad_test_X = self.test_X.loc[:,~self.test_X.columns.str.contains("infra")]
    return cad_train_X, self.train_Y, cad_test_X, self.test_Y