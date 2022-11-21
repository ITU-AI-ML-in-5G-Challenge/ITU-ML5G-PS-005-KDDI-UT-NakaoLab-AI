import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn import metrics

rcParams['figure.figsize'] = 15, 6

class Evaluation:
  def __init__(self, pred, test, label, timesteps, delay, threshold=0.33333):
    self.pred = pred
    self.test = test
    self.label = label
    self.timesteps = timesteps
    self.delay = delay
    self.threshold = threshold
    
  def score(self):
    pred_600 = self.pred[:,-10]
    label_600 = self.label.reshape(300,-1)[:,-10]
    test_ = [0 if i=='normal' else 1 for i in label_600]
    pred_ = [0 if i<self.threshold else 1 for i in pred_600]
    print(metrics.confusion_matrix(test_, pred_))
    print(metrics.classification_report(test_, pred_))
    
  def visualization(self):
    test_vis = self.test[-20:-10]
    pred_vis = self.pred[-20:-10]
    x = np.arange(len(test_vis.flatten()))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, test_vis.flatten(), label='y_test')
    for i in range(len(pred_vis)):
      if i==0:
        ax.plot(x[i*70+self.timesteps+self.delay:(i+1)*70], pred_vis[i], label='y_pred', c='r')
      else:
        ax.plot(x[i*70+self.timesteps+self.delay:(i+1)*70], pred_vis[i], c='r')
    ax.legend()
    plt.show()
  
  def MSE(self):
    pred_fla = self.pred.flatten()
    test_fla = self.test[:,self.timesteps+self.delay:].flatten()
    return sum((pred_fla - test_fla)**2)/(len(pred_fla))