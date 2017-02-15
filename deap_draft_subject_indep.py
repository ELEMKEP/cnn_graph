# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:07:03 2017

To process DEAP data with GCNN algorithm.

Data(DEAP) is in Geneva order
Coordinate file is in Twente order

@author: JANG-LAB
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
import scipy.sparse
from lib import models, utils, graph, coarsening
from graph_signal_importer2 import ImporterIndep
import time
import pickle

# Training parameters
COARSEN_LEVEL = 6
#DATA_FILE = 'out_thresh_20161221134736.dat'  # mat-file storing data
#DATA_FILE = 'raw_20170126221414.dat' # power signal, distance 5-band graph (thresholded 0-1)
# DATA_FILE = 'raw_20170213165455_subject_dep.dat' # pow, dist, 5-band, 0~ thresh, subj-dep
DATA_FILE = 'raw_20170213173430_per_subject.dat' # pow, dist, 5-band, 0~ thresh, subj-wise

def gc_dist(u, v):
  assert u[2]==v[2], 'Should have same radius in spherical coordinate!'
  
  th = np.deg2rad(np.abs(u[0]-v[0]))
  ph1 = np.deg2rad(u[1])
  ph2 = np.deg2rad(v[1])
  
  s = np.arccos(np.sin(ph1)*np.sin(ph2) + np.cos(ph1)*np.cos(ph2)*np.cos(th))
  d = u[2]*s
  return d

def draw_graph(coords, graph, labels=None, rad=0.05, axlim=0.6):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim([-axlim, axlim])
    ax.set_ylim([-axlim, axlim])
    
    angles = np.deg2rad(coords[:, 0])
    radius = coords[:, 1]
    X = radius * np.cos(angles)
    Y = radius * np.sin(angles)
    
    circles = list()
    texts = list()
    for idx in range(coords.shape[0]):
      circles.append(Circle((X[idx], Y[idx]), 
                             radius=rad, 
                             color=(0.5, 0.5, 1),
                             clip_on=False,
                             figure=fig))
      texts.append(Text(X[idx],
                        Y[idx], 
                        multialignment='right',
                        text=labels[idx], 
                        color='k',
                        size=20,
                        figure=fig))
    
    rows, cols, vals = scipy.sparse.find(graph)
    vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals)) * 0.6 + 0.2
    lines = list()
    for idx in range(len(vals)):
      p1 = rows[idx]
      p2 = cols[idx]
      lines.append(Line2D([X[p1], X[p2]],
                          [Y[p1], Y[p2]],
                          color=[vals[idx]]*3,
                          figure=fig))
    
    for line in lines:
      ax.add_artist(line)
    
    for circle in circles:
      ax.add_artist(circle)
    
    for text in texts:
      ax.add_artist(text)
    fig.canvas.draw()
      
  
def adjacency_dist_single_layer():
  twente_to_geneva = [0 , 1 , 3 , 2 , 5 , 4 , 7 , 6 , \
                      9 , 8 , 11, 10, 13, 14, 15, 12, \
                      29, 28, 30, 26, 27, 24, 25, 31, \
                      22, 23, 20, 21, 18, 19, 17, 16]
  
  tw_labels = list()
  tw_2d_sph_coords = list()
  tw_3d_car_coords = list()
  tw_3d_sph_coords = list()
  with open('10-20_32ch.ced', 'rb') as f:
    first_line = f.readline()
    while True:
      a = f.readline()
      
      asp = str.split(str(a, 'utf-8'))
      if len(asp) < 10:
        break    
      tw_labels.append(asp[1])
      tw_2d_sph_coords.append(np.asarray(asp[2:4]).astype(np.float32))
      tw_3d_car_coords.append(np.asarray(asp[4:7]).astype(np.float32))
      tw_3d_sph_coords.append(np.asarray(asp[7:10]).astype(np.float32))
  
  ge_labels = list()
  ge_2d_sph_coords = list()
  ge_3d_car_coords = list()
  ge_3d_sph_coords = list()
  for idx in twente_to_geneva:
    ge_labels.append(tw_labels[idx])
    ge_2d_sph_coords.append(tw_2d_sph_coords[idx])
    ge_3d_car_coords.append(tw_3d_car_coords[idx])
    ge_3d_sph_coords.append(tw_3d_sph_coords[idx])
    
  ge_2d_sph_coords = np.asarray(ge_2d_sph_coords)
  ge_3d_car_coords = np.asarray(ge_3d_car_coords)
  ge_3d_sph_coords = np.asarray(ge_3d_sph_coords)
  
  dist_metric = 'euclidean'
  dist, idx = graph.distance_scipy_spatial(ge_3d_sph_coords, k=4, 
                                           metric=gc_dist)
  
  for i in range(idx.shape[0]):
    print([ge_labels[i]],[ge_labels[j] for j in idx[i]])
  
  # Starting graph process
  A = graph.adjacency(dist, idx).astype(np.float32)
  return A

# Start point

# Neural network


importer = ImporterIndep(DATA_FILE)
model_perf = utils.model_perf()

for subject_idx in range(32):
  A = importer.graph

  graphs, perm = coarsening.coarsen(A, levels=COARSEN_LEVEL,
                                    self_connections=False)
  L = [graph.laplacian(gA, normalized=True) for gA in graphs]

  subject_train = list(range(32))
  subject_valid = [subject_train.pop(subject_idx)]

  importer.set_train_val_subjects(subject_train, subject_valid)

  n_train = importer.train_length
  n_valid = importer.valid_length

  _, signal_train, labels_train = importer.next_batch_train_(n_train)
  _, signal_valid, labels_valid = importer.next_batch_valid_(n_valid)

  print((signal_train.shape,
         labels_train.shape,
         signal_valid.shape,
         labels_valid.shape))

  signal_dim = signal_train.shape[-1]
  signal_train = np.reshape(signal_train, (-1, signal_dim))
  signal_valid = np.reshape(signal_valid, (-1, signal_dim))

  t_start = time.process_time()
  signal_train = coarsening.perm_data(signal_train, perm)
  signal_valid = coarsening.perm_data(signal_valid, perm)
  print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
  del perm

  common = dict()
  common['dir_name'] = 'deap/'
  common['num_epochs'] = 50
  common['batch_size'] = 580
  common['decay_steps'] = n_train / common['batch_size']
  common['eval_frequency'] = 124 #124 iter for 1 epoch in 31/1 subject(s)
  common['brelu'] = 'b1relu'
  common['pool'] = 'mpool1'
  C = 2  # number of classes

  name = 'test'+str(subject_idx)
  params = common.copy()
  params['dir_name'] += name
  params['regularization'] = 5e-4
  params['dropout']        = 0
  params['learning_rate']  = 0.00001
  params['decay_rate']     = 0.95
  params['momentum']       = 0.9
  params['F']              = [40, 80, 80, 160, 160, 240]
  params['K']              = [4, 2, 2, 2, 2, 1]
  params['p']              = [2, 2, 2, 2, 2, 2]
  params['M']              = [C]


  # Go to real training
  t_start = time.process_time()
  model_perf.test(models.cgcnn(L, **params), name, params,
                  signal_train, labels_train,
                  signal_valid, labels_valid,
                  signal_valid, labels_valid)
  print('Execution time: {:.2f}s'.format(time.process_time() - t_start))

print('Total result')
print('Model accuracy in TEST')
print(model_perf.test_accuracy)
print('Model F1-value in TEST')
print(model_perf.test_f1)
print('Model loss in TEST')
print(model_perf.test_loss)

with open('test_result.dat', 'wb') as f:
  pickle.dump(model_perf, f)








