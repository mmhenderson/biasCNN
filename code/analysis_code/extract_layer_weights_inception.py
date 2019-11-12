#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:53:36 2019

@author: mmhender

Some code to extract list of tensors, their sizes, and their weights from network after some amount of training. 
This works for nasnet now, but can easily be changed for another network
Also this verifies that the early layers of network are frozen when we re-train the last few layers
Exception is any tensors with moving_mean or moving_variance...these differ on each training iteration because batch normalization happens on each batch of ims.

"""
import tensorflow as tf
import os 

import numpy as np
import matplotlib.pyplot as plt
import scipy

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as prt
from tensorflow.contrib.framework import arg_scope as arg_scope
from tensorflow.python import pywrap_tensorflow

from slim.nets import inception_v3 as inception_v3

root = '/usr/local/serenceslab/maggie/biasCNN/';


ckptfile1 = os.path.join(root, 'logs', 'inception_oriTrn1_short','model.ckpt-1')
metafile1 = os.path.join(root, 'logs', 'inception_oriTrn1_short','model.ckpt-1.meta')


ckptfile2 = os.path.join(root, 'logs', 'inception_oriTrn1_long','model.ckpt-10000')
metafile2 = os.path.join(root, 'logs', 'inception_oriTrn1_long','model.ckpt-10000.meta')

ckptfile3 = os.path.join(root, 'checkpoints', 'inception_ckpt', 'inception_v3.ckpt')

#%% print list of all tensors
# Load a .ckpt file for the model after some amount of training, inspect the checkpoint file using a tensorflow built-in function.
    
prt(ckptfile1,[],all_tensors=False)
prt(ckptfile3,[],all_tensors=False)

#%%Get a list of the names of all tensors

reader = pywrap_tensorflow.NewCheckpointReader(ckptfile1)
var_to_shape_map = reader.get_variable_to_shape_map()
ind = 0 # can change this to other numbers
tensor_list = list(var_to_shape_map.keys())
tensors2check = []
for kk in tensor_list:
    # i'm choosing a random cell here to look at, there are a ton of tensors in this model so it takes forever to check all of them. but you could.
    if ('Mixed_7b' in kk): 
        tensors2check.append('%s:%d' %(kk,ind))

# here we'll store the weights for each of these tensors
w_check = []
 
#%% get the weights of each layer, before any re-training

tf.reset_default_graph()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(metafile1)
    saver.restore(sess, ckptfile1)
    # get the graph
    g = tf.get_default_graph()

    w_1 = []
    for tt in range(np.size(tensors2check)):
        
        w_1.append(sess.run(g.get_tensor_by_name(tensors2check[tt])))
        
    w_check.append(w_1)

#%% same thing after re-tuning the last few layers
tf.reset_default_graph()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(metafile2)
    saver.restore(sess, ckptfile2)
    # get the graph
    g = tf.get_default_graph()
     
    w_2 = []
    for tt in range(np.size(tensors2check)):
        
        w_2.append(sess.run(g.get_tensor_by_name(tensors2check[tt])))
        
    w_check.append(w_2)
    
#%% get out the activations for same image set, before and after training
# these activation measurements were saved out by "eval_image_classifier_MMH_biasCNN.py"
# they have already been subjected to PCA within layer
model_str = 'inception_oriTst1';

import load_activations
#allw, all_labs, info = load_activations.load_activ_nasnet_oriTst0()
allw, all_labs, info = load_activations.load_activ(model_str)
layer_labels = info['layer_labels']
nLayers = info['nLayers']

plt.close('all')
#corrs = np.zeros(nLayers, 1)
#for l1  in range(nLayers):
for l1 in [2,5,8]:
    act1 = (allw[l1][0])
    act2 = (allw[l1][1])
    
    ii=150; # choose a random image in the set
    
    plt.figure();plt.plot(act1[ii,:]);plt.plot(act2[ii,:]);
    plt.legend(info['timepoint_labels'])
    plt.title('%s\nactivations for a single image, after PCA' %(layer_labels[l1]))
    corrs = []
    for ii in np.arange(0,np.shape(act1)[0],100):
        r,p = scipy.stats.pearsonr(act1[ii,:],act2[ii,:])
        corrs.append(r)
    
#%% evaluate the original model before we did anything to it
# don't have a meta file here so we have to use the .py file to define the graph 

# note this block of code will NOT run for the model once we've re-trained it, because the sizes of the logits layer are different 
# since we specified 180 possible labels in orientation space and the original model has 1001. 
# can overcome this by specifying which layers to restore and which not to restore. 
# this is a bit complicated to do here but is accomplished in eval_image_classifier_MMH_biasCNN

tf.reset_default_graph()

with tf.Session() as sess:
    
    graph = tf.get_default_graph()
    with graph.as_default():
     
        # input these key parameters, they match how the network was trained
        batch_size = 10
        image_size = 299
        # passing a bunch of blank images in, just as a placeholder
        images = np.ones([batch_size, image_size, image_size, 3])
        # testing for effect of batch normalization - if we change one of the images in the batch, does this change activations measured for individual ims?
        images2 = np.concatenate((np.ones([batch_size-1, image_size, image_size, 3]), 2*np.ones([1, image_size, image_size, 3])), axis=0)
         
        # this argscope line seems to be very important, otherwise the checkpoint doesn't get loaded correctly.
       
        with arg_scope(inception_v3.inception_v3_arg_scope()):
#            
            inputs = tf.placeholder(tf.float32, shape=(batch_size,
                                                         image_size, image_size, 3))
            
            logits =  inception_v3.inception_v3(inputs, num_classes=1001, is_training=False,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV3',
                 create_aux_logits=True)
            
            test_vars = []
            for tt in range(np.size(tensors2check)):
                
                test_vars.append(graph.get_tensor_by_name(tensors2check[tt]))
           
        # restore the model from saved checkpt
        saver = tf.train.Saver()
        saver.restore(sess, ckptfile3)
      
        #% first run one set of images through the network...get activations
        mydict = {'Placeholder:0': images}
    
        out = sess.run([logits, test_vars], feed_dict = mydict)
    
        logit_activations = out[0][0]
        other_activations = out[0][1]
        
        # look at the weights for each layer
        w_3 = []
        for tt in range(np.size(tensors2check)):
            w_3.append(out[1][tt])
        
        w_check.append(w_3)
       
        # look at the activations, in resp to these images
        activ_list= list(other_activations.keys())
        activ2check = activ_list
        
        a_check = []
        a_1 = []
        for aa in range(np.size(activ2check)):
            a_1.append(other_activations[activ2check[aa]])
            
        a_check.append(a_1)     
            
        #% now run a slightly different set of images through - just changed the last one
        
        mydict = {'Placeholder:0': images2}
    
        out = sess.run([logits, test_vars], feed_dict = mydict)
    
        logit_activations = out[0][0]
        other_activations = out[0][1]
        
        w_4 = []
        for tt in range(np.size(tensors2check)):
            w_4.append(out[1][tt])
        
        w_check.append(w_4)
       
        a_2 = []
        for aa in range(np.size(activ2check)):
            a_2.append(other_activations[activ2check[aa]])
            
        a_check.append(a_2)   
        
#%% does batch normalization impact the activations for identical image? NO

# these should be the same
for aa in range(np.size(activ2check)):
    print(np.array_equal(a_check[0][aa][0:batch_size-1,:], a_check[1][aa][0:batch_size-1,:]))
  
# these should be different (last image was actually different)
for aa in range(np.size(activ2check)):
    print(np.array_equal(a_check[0][aa][batch_size-1,:], a_check[1][aa][batch_size-1,:]))
  
#%% print out the comparisons between all these
# all are the same except for the moving mean and moving variance
 
for tt in range(np.size(tensors2check)):
    print('%s:%s' %(np.array_equal(w_check[0][tt],w_check[1][tt]), tensors2check[tt]))
    
for tt in range(np.size(tensors2check)):
    print('%s:%s' %(np.array_equal(w_check[0][tt],w_check[1][tt]), tensors2check[tt]))
    
