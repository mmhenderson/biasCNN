#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

#from IPython import get_ipython
#def __reset__(): get_ipython().magic('reset -sf')
#__reset__()

import os
import numpy as np
#import matplotlib.pyplot as plt
#
#import scipy

from sklearn import decomposition

import shutil

import tensorflow as tf


#%% set up inputs based on how the function is called

tf.app.flags.DEFINE_string(
    'activ_path','', 'The path to load the big activation files from.')

tf.app.flags.DEFINE_string(
    'reduced_path','', 'The path to save the reduced activation files to.')

tf.app.flags.DEFINE_integer(
    'n_components_keep',500, 'The number of components to save.')

tf.app.flags.DEFINE_integer(
        'pctVar', 80, 'The percent of variance to explain (only affects print statements)')

tf.app.flags.DEFINE_integer(
        'num_batches', 80, 'How many batches the dataset is divided into.')

FLAGS = tf.app.flags.FLAGS

#%% more parameters that are always the same

# list of all the endpoints in this network.
layers2load = ['Conv2d_1a_3x3',
 'Conv2d_2a_3x3',
 'Conv2d_2b_3x3',
 'MaxPool_3a_3x3',
 'Conv2d_3b_1x1',
 'Conv2d_4a_3x3',
 'MaxPool_5a_3x3',
 'Mixed_5b',
 'Mixed_5c',
 'Mixed_5d',
 'Mixed_6a',
 'Mixed_6b',
 'Mixed_6c',
 'Mixed_6d',
 'Mixed_6e',
 'Mixed_7a',
 'Mixed_7b',
 'Mixed_7c',
 'AuxLogits',
 'AvgPool_1a',
 'PreLogits',
 'Logits',
 'Predictions']

nLayers = len(layers2load)

n_components_keep = FLAGS.n_components_keep
pctVar = FLAGS.pctVar
num_batches = FLAGS.num_batches

#%% information about the stimuli. 
sf_vals = np.logspace(np.log10(1.25), np.log10(3.88),3)
stim_types = ['Gaussian']
nOri=180
nSF=3
nPhase=16
nType=1
nNoiseLevels = 1
noise_levels = [0.20]

# list all the image features in a big matrix, where every row is unique.
noiselist = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
typelist = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
sflist=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))

featureMat = np.concatenate((noiselist,typelist,orilist,sflist,phaselist),axis=1)

assert np.array_equal(featureMat, np.unique(featureMat, axis=0))

actual_labels = orilist



#%% load in the weights from the network - BEFORE and AFTER retraining

#for ll in range(np.size(layers2load)):
   
def main(_):

#    for nn in range(nNoiseLevels):
        
        #set up my folders for loading, and saving reduced weights
#        path2load = os.path.join(activ_path, 'noise%.2f' % noise_levels[nn])        
#        path2save = os.path.join(reduced_path, 'noise%.2f' % noise_levels[nn])
    path2load = FLAGS.activ_path
    path2save = FLAGS.reduced_path
    
    if tf.gfile.Exists(path2save):
        print('deleting contents of %s' % path2save)
#        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        shutil.rmtree(path2save, ignore_errors = True)
        tf.gfile.MakeDirs(path2save)    
    else:
        tf.gfile.MakeDirs(path2save)   
        
    for ll in range(nLayers):
#    for ll in range(17,19):
        allw = None
        
        for bb in np.arange(0,num_batches,1):
    
            file = os.path.join(path2load, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
            print('loading from %s\n' % file)
            w = np.squeeze(np.load(file))
            w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
            if bb==0:
                allw = w
            else:
                allw = np.concatenate((allw, w), axis=0)
    
            file = os.path.join(path2load, 'batch' + str(bb) + '_labels_orig.npy')    
            feat = np.expand_dims(np.load(file),1)
         
            if bb==0:
                allfeat = feat
            else:
                allfeat = np.concatenate((allfeat,feat),axis=0) 
    
            file = os.path.join(path2load, 'batch' + str(bb) + '_labels_predicted.npy')    
            pred = np.expand_dims(np.load(file),1)
         
            if bb==0:
                allpred = pred
            else:
                allpred = np.concatenate((allpred,pred),axis=0) 
    
        # double check and make sure we loaded all the right trials
        assert np.array_equal(allfeat, actual_labels)
         
         #%% Run  PCA on this weight matrix to reduce its size
        
        pca = decomposition.PCA(n_components = np.min((n_components_keep, np.shape(allw)[1])))
        weights_reduced = pca.fit_transform(allw)   
        
        var_expl = pca.explained_variance_ratio_
        if np.max(np.cumsum(var_expl))>pctVar/100:
            nCompNeeded = np.where(np.cumsum(var_expl)>pctVar/100)[0][0]
            print('need %d components to capture %d percent of variance, saving first %d\n' % (nCompNeeded, pctVar, np.min((n_components_keep, np.shape(allw)[1]))))
        else:
            print('need > %d components to capture %d percent of variance, saving first %d\n' % (np.min((n_components_keep, np.shape(allw)[1])), pctVar, np.min((n_components_keep, np.shape(allw)[1]))))
    
       
#        plt.figure()
#        plt.scatter(np.arange(0,np.shape(var_expl)[0],1), np.cumsum(var_expl))
#        plt.title('%s layer: Cumulative pct var explained by each component' % layers2load[ll])
    
        
        #%% Save the result as a single file
        
        fn2save = os.path.join(path2save, 'allStimsReducedWts_' + layers2load[ll] +'.npy')
        
        np.save(fn2save, weights_reduced)
        print('saving to %s\n' % (fn2save))
        
        fn2save = os.path.join(path2save, 'allStimsVarExpl_' + layers2load[ll] +'.npy')
            
        np.save(fn2save, var_expl)
        print('saving to %s\n' % (fn2save))
        
        fn2save = os.path.join(path2save, 'allStimsLabsPredicted_' + layers2load[ll] +'.npy')
            
        np.save(fn2save, allpred)
        print('saving to %s\n' % (fn2save))
        
        fn2save = os.path.join(path2save, 'allStimsLabsOrig_' + layers2load[ll] +'.npy')
            
        np.save(fn2save, allfeat)
        print('saving to %s\n' % (fn2save))
        
if __name__ == '__main__':
    tf.app.run()
