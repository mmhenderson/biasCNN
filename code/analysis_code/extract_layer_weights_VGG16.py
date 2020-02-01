#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:53:36 2019

@author: mmhender

Some code to extract list of tensors, their sizes, and their weights from network after some amount of training. 

"""

import tensorflow as tf
import os 

import numpy as np
import matplotlib.pyplot as plt
#import scipy

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as prt
from tensorflow.contrib.framework import arg_scope as arg_scope
from tensorflow.python import pywrap_tensorflow

import slim.nets.vgg as vgg
from slim.nets.vgg import vgg_16 as vgg_16

slim = tf.contrib.slim


root = '/usr/local/serenceslab/maggie/biasCNN/';


#ckptfile = os.path.join(root, 'logs', 'vgg16','ImageNet','scratch_imagenet_rot_0_square','params1','model.ckpt-2194')
#metafile = os.path.join(root, 'logs', 'vgg16','ImageNet','scratch_imagenet_rot_0_square','params1','model.ckpt-2194.meta')

# this is for a model that i trained
ckptfile = os.path.join(root, 'logs', 'vgg16','ImageNet','scratch_imagenet_rot_0_square','params1','model.ckpt-400476')
metafile = os.path.join(root, 'logs', 'vgg16','ImageNet','scratch_imagenet_rot_0_square','params1','model.ckpt-400476.meta')

# this is for a pre-trained model
ckptfile_pretrained = os.path.join(root, 'checkpoints', 'vgg16_ckpt','vgg_16.ckpt')

#%%Get a list of the names of all tensors of interest, and their sizes
    
prt(ckptfile,[],all_tensors=False)

# list of all the layers i'm interested in
layers2load= ['conv1_1','conv1_2',
   'conv2_1','conv2_2',
   'conv3_1','conv3_2','conv3_3',
   'conv4_1','conv4_2','conv4_3',
   'conv5_1','conv5_2','conv5_3',
   'fc6',
   'fc7',
   'fc8']
reader = pywrap_tensorflow.NewCheckpointReader(ckptfile)
var_to_shape_map = reader.get_variable_to_shape_map()

# long list of every single tensor (includes extra variables)
tensor_list = list(var_to_shape_map.keys())
size_list = list(var_to_shape_map.values())

# make a shorter list of just the layers i am interested in here
tensor_sizes = dict()
tensors2load = []
for kk in range(np.size(layers2load)):
  weights_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'weights' in ii and 'RMSProp' not in ii]
  weights_tensor = weights_tensor[0]
  biases_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'biases' in ii and 'RMSProp' not in ii]
  biases_tensor = biases_tensor[0]
  
  tensor_sizes[weights_tensor] = var_to_shape_map[weights_tensor]
  tensor_sizes[biases_tensor] = var_to_shape_map[biases_tensor]
  
  tensors2load.append('%s:0'%weights_tensor)
  
tensor_list = list(tensor_sizes.keys())
size_list = list(tensor_sizes.values())

#%% get the weights for each of these tensors, at the specified training step
w_all = []

tf.reset_default_graph()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(metafile)
    saver.restore(sess, ckptfile)
    # get the graph
    g = tf.get_default_graph()
  
    w_all = [sess.run(g.get_tensor_by_name(tt)) for tt in tensors2load]
  
    
#%% make plots of some the filters
plt.close('all')
plt.rcParams.update({'font.size': 8})

layers_plot = [1,4,7] # which layers?
input_chans_plot = [0]  # which input channels?
output_chans_plot = np.arange(0,64,4) # which output channels?

for ll in range(np.size(layers_plot)):
  
  w = w_all[layers_plot[ll]]

  for cc in input_chans_plot:
  
    fig = plt.figure();
    px = 0
    for ff in output_chans_plot:
      px=px+1
      plt.subplot(np.ceil(np.sqrt(np.size(output_chans_plot))),np.ceil(np.sqrt(np.size(output_chans_plot))),px)
      plt.pcolormesh(np.squeeze(w[:,:,cc,ff]))
      plt.axis('square')
      plt.axis('off')
      plt.title('map %d'%(ff))
      
    plt.suptitle('%s\ninput channel %d\n'%(tensors2load[layers_plot[ll]],cc))
     
    fig.set_size_inches(8,8)

#%% get activations sizes from random images
# using the pre-trained model
# don't have a meta file here so we have to use the .py file to define the graph
# note this block of code will NOT run for the model once we've re-trained it, 
# because the names of the layers are slightly different 
tf.reset_default_graph()

with tf.Session() as sess:
    
    graph = tf.get_default_graph()
    with graph.as_default():
     
        # input these key parameters, they match how the network was trained
        batch_size = 10
        image_size = 224
        # passing a bunch of blank images in, just as a placeholder
        images = np.ones([batch_size, image_size, image_size, 3])
        
        with arg_scope(vgg.vgg_arg_scope()):
           
            inputs = tf.placeholder(tf.float32, shape=(batch_size,
                                                         image_size, image_size, 3))
            
            [logits, end_pts] = vgg_16(inputs, num_classes=1000,
                       is_training=False,spatial_squeeze=False)

        # restore the model from saved checkpt
        saver = tf.train.Saver()
        saver.restore(sess, ckptfile_pretrained)
        
        #% first run one set of images through the network...get activations
        mydict = {'Placeholder:0': images}
    
        out = sess.run([logits,end_pts], feed_dict = mydict)
    
        logit_activations = out[0]
        layer_activations = list(out[1].values())
        layer_names = list(out[1].keys())
        activ_sizes = { layer_names[ii] : np.shape(layer_activations[ii]) for ii in range(np.size(layer_names)) }
