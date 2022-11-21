import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn import metrics

rcParams['figure.figsize'] = 15, 6

class evaluation:
  def __init_(self, pred, test, label, threshold=0.33333):
    self.pred = pred
    self.test = test
    self.label = label
    self.threshold = threshold
    
  def score(self):
    
    
  def visualization(self):
    
  
  def MSE(self):
    