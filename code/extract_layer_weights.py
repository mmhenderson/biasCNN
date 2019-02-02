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

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as prt


#from slim.nets import inception_v4 as inception_v4

from slim.nets.nasnet import nasnet as nasnet

from tensorflow.contrib.framework import arg_scope as arg_scope


root = '/usr/local/serenceslab/maggie/biasCNN/';


ckptfile1 = os.path.join(root, 'logs', 'nasnet_retrained_grating_orient_sf_short','model.ckpt-1')
metafile1 = os.path.join(root, 'logs', 'nasnet_retrained_grating_orient_sf_short','model.ckpt-1.meta')


ckptfile2 = os.path.join(root, 'logs', 'nasnet_retrained_grating_orient_sf_long','model.ckpt-10000')
metafile2 = os.path.join(root, 'logs', 'nasnet_retrained_grating_orient_sf_long','model.ckpt-10000.meta')


ckptfile3 = os.path.join(root, 'checkpoints', 'pnas_ckpt', 'nasnet-a_large_04_10_2017','model.ckpt')
#%% print list of all tensors
# Load a .ckpt file for the model after some amount of training, inspect the checkpoint file using a tensorflow built-in function.

#prt(ckptfile1,'reduction_cell_1/prev_1x1/weights',all_tensors=False)
prt(ckptfile1,[],all_tensors=False)

#%% Print all tensors in the pre-trained network before we did anything to it
# note that this file with .ckpt extension does not strictly exist, but somehow this works anyway. 

prt(ckptfile3,[],all_tensors=False)
 
#%% comparing the weights of different layers, after different amounts of re-training (they should all be the same since they're frozen)
#sess.close()
tf.reset_default_graph()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(metafile1)
    saver.restore(sess, ckptfile1)
    # get the graph
    g = tf.get_default_graph()
    
    list_of_ops = [op.name for op in g.get_operations()]
    
    sublist = [];
    
    for op in list_of_ops:
        if 'logits' in op:
            sublist.append(op)
            
    # can change these tensor names to anything...see the full list from prt function for options
#    w1_1 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/beta:0'))
#    w2_1 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/gamma:0'))
#    w3_1 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/moving_mean:0'))
#    w4_1 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/moving_variance:0'))

#    w1_1 = sess.run(g.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/depthwise_weights:0'))
#    w2_1 = sess.run(g.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/pointwise_weights:0'))
#    w3_1 = sess.run(g.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/depthwise_weights:0'))
#    w4_1 = sess.run(g.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/pointwise_weights:0'))

#%%
tf.reset_default_graph()

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph(metafile2)
    saver.restore(sess, ckptfile2)
    # get the graph
    g = tf.get_default_graph()
    
#    w1_2 = sess.run(g.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/depthwise_weights:0'))
#    w2_2 = sess.run(g.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/pointwise_weights:0'))
#    w3_2 = sess.run(g.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/depthwise_weights:0'))
#    w4_2 = sess.run(g.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/pointwise_weights:0'))

#    w1_2 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/beta:0'))
#    w2_2 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/gamma:0'))
#    w3_2 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/moving_mean:0'))
#    w4_2 = sess.run(g.get_tensor_by_name('reduction_cell_1/prev_bn/moving_variance:0'))


    batch_size = 128
    image_size = 299
    # passing a bunch of blank images in, just as a placeholder
    images = np.ones([batch_size, image_size, image_size, 3])
    
    with arg_scope(nasnet.nasnet_large_arg_scope()):
#            
        inputs = tf.placeholder(tf.float32, shape=(batch_size,
                                                     image_size, image_size, 3))
        
        logits = nasnet.build_nasnet_large(inputs, num_classes=1001,
                   is_training=False,
                   final_endpoint=None)
    

   
  
    
    mydict = {'Placeholder:0': images}
    
    out = sess.run([logits], feed_dict = mydict)
    
#%% evaluate the original model before we did anything to it
# don't have a meta file here so we have to use the .py file to define the graph for NASNet

# note this block of code will NOT run for the model once we've re-trained it, because the sizes of the logits layer are different 
# since we specified 180 possible labels in orientation space and the original model has 1001. 
# can overcome this by specifying which layers to restore and which not to restore. 
# this is a bit complicated to do here but is accomplished in eval_image_classifier_MMH_biasCNN

tf.reset_default_graph()

with tf.Session() as sess:
    
    graph = tf.get_default_graph()
    with graph.as_default():
     
        # input these key parameters, they match how the network was trained
        batch_size = 128
        image_size = 299
        # passing a bunch of blank images in, just as a placeholder
        images = np.ones([batch_size, image_size, image_size, 3])
        # testing for effect of batch normalization - if we change one of the images in the batch, does this change activations measured for individual ims?
        images2 = np.concatenate((np.ones([batch_size-1, image_size, image_size, 3]), 2*np.ones([1, image_size, image_size, 3])), axis=0)
         
        # this argscope line seems to be very important, otherwise the checkpoint doesn't get loaded correctly.
       
        with arg_scope(nasnet.nasnet_large_arg_scope()):
#            
            inputs = tf.placeholder(tf.float32, shape=(batch_size,
                                                         image_size, image_size, 3))
            
            logits = nasnet.build_nasnet_large(inputs, num_classes=1001,
                       is_training=False,
                       final_endpoint=None)
            
            # this is a tensor that we think should have the same value here as after training...extracting its weights here.
#            test_var_1 = graph.get_tensor_by_name('reduction_cell_1/prev_bn/beta:0')
#            test_var_2 = graph.get_tensor_by_name('reduction_cell_1/prev_bn/gamma:0')
#            test_var_3 = graph.get_tensor_by_name('reduction_cell_1/prev_bn/moving_mean:0')
#            test_var_4 = graph.get_tensor_by_name('reduction_cell_1/prev_bn/moving_variance:0')
#            
            test_var_1 = graph.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/depthwise_weights:0')
            test_var_2 = graph.get_tensor_by_name('cell_7/comb_iter_0/left/separable_5x5_1/pointwise_weights:0')
            test_var_3 = graph.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/depthwise_weights:0')
            test_var_4 = graph.get_tensor_by_name('cell_9/comb_iter_0/left/separable_5x5_2/pointwise_weights:0')
           
        # restore the model from saved checkpt
        saver = tf.train.Saver()
        saver.restore(sess, ckptfile3)
        
        mydict = {'Placeholder:0': images}
    
        out = sess.run([logits, test_var_1, test_var_2, test_var_3, test_var_4], feed_dict = mydict)
    
        logit_activations = out[0][0]
        other_activations = out[0][1]
        w1_3 = out[1]
        w2_3 = out[2]
        w3_3 = out[3]
        w4_3 = out[4]
        
        activ_list= list(other_activations.keys())
#        np.array_equal(logit_activations, other_activations['Logits'])
        
        # can extract activation pattern from any layer of interest...activ_list is their names
        a1_1 = other_activations['Cell_3'];
        a2_1 = other_activations['Reduction_Cell_1']
        a3_1 = other_activations['Predictions']
        
        
        mydict = {'Placeholder:0': images2}
    
        out2 = sess.run([logits, test_var_1, test_var_2, test_var_3, test_var_4], feed_dict = mydict)
    
        logit_activations = out2[0][0]
        other_activations = out2[0][1]
#        w1_3 = out[1]
#        w2_3 = out[2]
#        w3_3 = out[3]
#        w4_3 = out[4]
        
        activ_list= list(other_activations.keys())
#        np.array_equal(logit_activations, other_activations['Logits'])
        
        a1_2 = other_activations['Cell_3'];
        a2_2 = other_activations['Reduction_Cell_1']
        a3_2 = other_activations['Predictions']
        
#%% does batch normalization impact the activations for identical image? NO

# these should be the same
print(np.array_equal(a1_1[0], a1_2[0]))
print(np.array_equal(a2_1[0], a2_2[0]))
print(np.array_equal(a3_1[0], a3_2[0]))

# these should be different
print(np.array_equal(a1_1[127], a1_2[127]))
print(np.array_equal(a2_1[127], a2_2[127]))
print(np.array_equal(a3_1[127], a3_2[127]))
    
#%% these should all be TRUE

#beta and gamma are same...moving mean and variance are different.

print(np.array_equal(w1_1,w1_2))
print(np.array_equal(w2_1,w2_2))
print(np.array_equal(w3_1,w3_2))
print(np.array_equal(w4_1,w4_2))

print(np.array_equal(w1_1,w1_3))
print(np.array_equal(w2_1,w2_3))
print(np.array_equal(w3_1,w3_3))
print(np.array_equal(w4_1,w4_3))
