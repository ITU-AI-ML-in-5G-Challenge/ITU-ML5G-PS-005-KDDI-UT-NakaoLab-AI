import numpy as np


class ReccurentTrainingGenerator(Sequence):
    def _resetindices(self):
        self.num_called = 0
        shaffled_idx = np.random.permutation(self.all_idx)
        temp_idx = (len(shaffled_idx)//self.batch_size)*self.batch_size
        indices = shaffled_idx[:temp_idx].reshape(self.steps_per_epoch-1, self.batch_size)
        indices = list(indices)
        indices.append(shaffled_idx[-(len(shaffled_idx)%self.batch_size):])
        self.indices = indices
        
    def __init__(self, x_set, y_set, batch_size, timesteps, delay):
        """
        x_set      : input numpy array
        y_set      : output numpy array
        batch_size : batch size
        timesteps  : input size
        delay      : delay size of output
        """
        self.x = np.array(x_set)
        self.y = np.array(y_set)
        self.batch_size = batch_size
        self.steps = timesteps
        self.delay = delay
        
        self.all_idx = np.array([[70*j+i for i in range(70-self.steps-self.delay)] for j in range(int(len(self.y)/70))]).flatten()
        self.num_samples = (70-(self.steps+self.delay))*(len(self.y)/70)
        self.steps_per_epoch = int(np.ceil( self.num_samples / float(batch_size)))
        
        self._resetindices()
        
    def __len__(self):
        """ return steps per epoch """
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ return batch data """
        indices_temp = self.indices[idx]
        
        batch_x = np.array([self.x[i:i+self.steps] for i in indices_temp])
        batch_y = self.y[indices_temp+self.steps+self.delay-1]
        
        if self.num_called==(self.steps_per_epoch-1):
            self._resetindices()
        else:
            self.num_called += 1
        
        return batch_x, batch_y
      
      
class ReccurentTestGenerator(Sequence):
    def _resetindices(self):
        temp_idx = (len(self.all_idx)//self.batch_size)*self.batch_size
        indices = self.all_idx[:temp_idx].reshape(self.steps_per_epoch-1, self.batch_size)
        indices = list(indices)
        indices.append(self.all_idx[-(len(self.all_idx)%self.batch_size):])
        self.indices = indices
        
    def __init__(self, x_set, batch_size, timesteps, delay):
        """
        x_set      : input numpy array
        y_set      : output numpy array
        batch_size : batch size
        timesteps  : input size
        delay      : delay size of output
        """
        self.x = np.array(x_set)
        self.batch_size = batch_size
        self.steps = timesteps
        self.delay = delay
        
        self.all_idx = np.array([[70*j+i for i in range(70-self.steps-self.delay)] for j in range(int(len(self.x)/70))]).flatten()
        self.num_samples = (70-(self.steps+self.delay))*(len(self.x)/70)
        self.steps_per_epoch = int(np.ceil( self.num_samples / float(batch_size)))
        
        self._resetindices()
        
    def __len__(self):
        """ return steps per epoch """
        return self.steps_per_epoch
        
    def __getitem__(self, idx):
        """ return batch data """
        indices_temp = self.indices[idx]
        
        batch_x = np.array([self.x[i:i+self.steps] for i in indices_temp])
        
        return batch_x