# Difference: handles the same graph

import numpy as np
import pickle

class Importer(object):
  def __init__(self, dat_file, frac):
    with open(dat_file, 'rb') as f:
      data = pickle.load(f)
    
    self.graph = data['graph'].astype(np.float32)
      
    self.signals, self.labels = \
      self.get_batch_perm_(data['labels'].shape[0],
                           data['signals'], 
                           data['labels'])

    assert ((self.signals.shape[0] == self.labels.shape[0]))
    assert (np.sum(frac) == 1.), 'Sum of fractions should be 1'

    self.train_frac = frac[0]
    self.valid_frac = frac[1]
    self.test_frac = frac[2]

    self.data_length = int(self.labels.shape[0])
    self.train_length = int(np.floor(self.data_length * self.train_frac))
    self.valid_length = int(np.floor(self.data_length * self.valid_frac))
    self.test_length = self.data_length - self.train_length - self.valid_length

    self.epoch = int(0)
    self.epoch_valid = int(0)
    self.epoch_test = int(0)

    self.cursor = int(0)
    self.cursor_valid = int(self.train_length)
    self.cursor_test = int(self.train_length + self.valid_length)

  def get_train_dimension(self):
    train_dimension = [self.train_length]
    train_dimension.extend(self.signals.shape[1:])
    return train_dimension

  def get_valid_dimension(self):
    valid_dimension = [self.valid_length]
    valid_dimension.extend(self.signals.shape[1:])
    return valid_dimension

  def get_test_dimension(self):
    test_dimension = [self.test_length]
    test_dimension.extend(self.signals.shape[1:])
    return test_dimension

  def next_batch_(self, batch_size, data_type='train'):
    # Extract data parameters
    if data_type == 'train':
      p_start = 0
      p_end = self.train_length
      d_length = self.train_length
      cursor = self.cursor
      epoch = self.epoch
    elif data_type == 'valid':
      p_start = self.train_length
      p_end = self.train_length + self.valid_length
      d_length = self.valid_length
      cursor = self.cursor_valid
      epoch = self.epoch_valid
    elif data_type == 'test':
      p_start = self.train_length + self.valid_length
      p_end = self.data_length
      d_length = self.test_length
      cursor = self.cursor_test
      epoch = self.epoch_test
    else:
      print('Not valid data type (train, valid, test)')
      return None

    # Extract data
    if cursor + batch_size >= p_end:
      signals_a = self.signals[cursor:p_end]
      labels_a = self.labels[cursor:p_end]

      cursor = (cursor + batch_size) - d_length
      epoch += 1

      signals_b = self.signals[p_start:cursor]
      labels_b = self.labels[p_start:cursor]

      signals = np.concatenate((signals_a, signals_b), axis=0)
      labels = np.concatenate((labels_a, labels_b), axis=0)
    else:
      signals = self.signals[cursor:cursor + batch_size]
      labels = self.labels[cursor:cursor + batch_size]
      cursor += batch_size

    # data params update
    if data_type == 'train':
      self.cursor = cursor
      self.epoch = epoch
    elif data_type == 'valid':
      self.cursor_valid = cursor
      self.epoch_valid = epoch
    elif data_type == 'test':
      self.cursor_test = cursor
      self.epoch_test = epoch
      
    return self.graph, signals, labels

  def get_batch_perm_(self, batch_size, signals, labels):
    # signals.shape = (n, 32)
    # signals_placeholder.shape = (n, 1, 32)

    # labels.shape = (n, 1)
    # labels_placeholder.shape = (n, )

    signals = signals.reshape(signals.shape[0], 1, signals.shape[1])
    labels = labels.squeeze().astype('int32')

    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    signals = signals[indices]
    labels = labels[indices]

    return signals.astype(np.float32), labels.astype(np.float32)

  def next_batch_train_(self, batch_size):
    return self.next_batch_(batch_size, data_type='train')

  def next_batch_valid_(self, batch_size):
    return self.next_batch_(batch_size, data_type='valid')

  def next_batch_test_(self, batch_size):
    return self.next_batch_(batch_size, data_type='test')

  def reset(self):
    self.epoch = int(0)
    self.epoch_valid = int(0)
    self.epoch_test = int(0)

    self.cursor = int(0)
    self.cursor_valid = int(self.train_length)
    self.cursor_test = int(self.train_length + self.valid_length)
    

