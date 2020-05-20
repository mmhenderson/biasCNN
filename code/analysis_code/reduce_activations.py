#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform PCA on activation patterns from each layer of a network corresponding to
a dataset of images. Reduce the activations to a smaller form that we can save 
and look at in python later.
"""
import os
import numpy as np
from sklearn import decomposition
import shutil
import tensorflow as tf # importing tensorflow only for argument parsing here

#%% set up inputs based on how the function is called

tf.app.flags.DEFINE_string(
    'activ_path','', 'The path to load the big activation files from.')

tf.app.flags.DEFINE_string(
    'reduced_path','', 'The path to save the reduced activation files to.')

tf.app.flags.DEFINE_string(
    'model_name','', 'The name of the current model.')

tf.app.flags.DEFINE_integer(
    'min_components_keep',10, 'The minimum number of components to save.')

tf.app.flags.DEFINE_integer(
        'pctVar', 95, 'The percent of variance to explain')

tf.app.flags.DEFINE_integer(
        'num_batches', 96, 'How many batches the dataset is divided into.')

FLAGS = tf.app.flags.FLAGS

min_components_keep = FLAGS.min_components_keep
pctVar = FLAGS.pctVar
num_batches = FLAGS.num_batches
   
#%% get layer names for this network

model_name = FLAGS.model_name

if model_name=='vgg_16':
        
    # list of all the endpoints in this network.
    layers2load = ['conv1_conv1_1','conv1_conv1_2',
     'conv2_conv2_1','conv2_conv2_2',
     'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3',
     'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3',
     'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3',
     'fc6',
     'fc7',
     'fc8',
     'pool1','pool2','pool3','pool4','pool5',
     'logits']
    
    nLayers = len(layers2load)
    for nn in range(nLayers-1):
        layers2load[nn] = 'vgg_16_' + layers2load[nn]
        
elif model_name=='vgg16_simul':
        
    # list of all the endpoints in this network.
    layers2load = ['conv1_1','conv1_2',
     'conv2_1','conv2_2',
     'conv3_1','conv3_2','conv3_3',
     'conv4_1','conv4_2','conv4_3',
     'conv5_1','conv5_2','conv5_3',
     'fc6',
     'pool1','pool2','pool3','pool4','pool5']
    
    nLayers = len(layers2load)
    print('loading %d layers for %s\n'%(nLayers,model_name))
elif model_name=='inception_v3':
    
    
    #   list of all the endpoints in this network.
    layers2load = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3','Conv2d_2b_3x3','MaxPool_3a_3x3',
     'Conv2d_3b_1x1','Conv2d_4a_3x3','MaxPool_5a_3x3',
     'Mixed_5b','Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
     'Mixed_6d','Mixed_6e','Mixed_7a', 'Mixed_7b','Mixed_7c','AuxLogits',
     'AvgPool_1a', 'PreLogits','Logits', 'Predictions']
    nLayers = len(layers2load)

elif model_name=='nasnet_large':

    layers2load = []
    
    for cc in range(17):
        layers2load.append('Cell_%d' % (cc+1))
    layers2load.append('global_pool')
    layers2load.append('logits')
    
    nLayers = len(layers2load)

 
#%% load in the weights from the network - BEFORE and AFTER retraining

def main(_):

    path2load = FLAGS.activ_path
    path2save = FLAGS.reduced_path
    
    n_components_keep = 600
    print('N_COMPONENTS_KEEP = %d\n'%n_components_keep)
    if tf.gfile.Exists(path2save):
        print('deleting contents of %s' % path2save)
        shutil.rmtree(path2save, ignore_errors = True)
        tf.gfile.MakeDirs(path2save)    
    else:
        tf.gfile.MakeDirs(path2save)   
        
    for ll in range(nLayers):
        allw = None
        
        for bb in np.arange(0,num_batches,1):
    
            file = os.path.join(path2load, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
            print('loading from %s\n' % file)
            w = np.squeeze(np.load(file))
            # w will be nIms x nFeatures
            w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
            
            if bb==0:
                allw = w
            else:
                allw = np.concatenate((allw, w), axis=0)
    
            
#            if bb==0:
#              # preallocate the array now (it's big esp for first layers)
#              batch_size=np.shape(w)[0]
#              allw = np.zeros([num_batches*batch_size,np.shape(w)[1]])
#  
#            batch_inds = np.arange(bb*batch_size, (bb+1)*batch_size,1)
#            allw[batch_inds,:] = w
    
         #%% Run  PCA on this weight matrix to reduce its size
        
       
        pca = decomposition.PCA(n_components = np.min((n_components_keep, np.shape(allw)[1])))
        print('\n STARTING PCA WITH %d COMPONENTS MAX\n'%(n_components_keep))
        print('size of allw before reducing is %d by %d'%(np.shape(allw)[0],np.shape(allw)[1]))
        weights_reduced = pca.fit_transform(allw)   
        
        var_expl = pca.explained_variance_ratio_
        
        n_comp_needed = np.where(np.cumsum(var_expl)>pctVar/100)
        if np.size(n_comp_needed)==0:
          n_comp_needed = n_components_keep
          print('need >%d components to capture %d percent of variance' % (n_comp_needed, pctVar))
        else:
          n_comp_needed = n_comp_needed[0][0]
          print('need %d components to capture %d percent of variance' % (n_comp_needed, pctVar))
          
        if n_comp_needed<min_components_keep:
            n_comp_needed = min_components_keep
      
        weights_reduced = weights_reduced[:,0:n_comp_needed]
      
        print('saving %d components\n'%np.shape(weights_reduced)[1])
        #%% Save the result as a single file
        
        fn2save = os.path.join(path2save, 'allStimsReducedWts_' + layers2load[ll] +'.npy')
        
        np.save(fn2save, weights_reduced)
        print('saving to %s\n' % (fn2save))
        
        fn2save = os.path.join(path2save, 'allStimsVarExpl_' + layers2load[ll] +'.npy')
            
        np.save(fn2save, var_expl)
        print('saving to %s\n' % (fn2save))
        
if __name__ == '__main__':
    tf.app.run()
