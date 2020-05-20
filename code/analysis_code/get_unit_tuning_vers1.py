#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import os
import numpy as np
import load_activations

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

nSamples =4
model='vgg16'
param_str='params1'

training_str_list = ['scratch_imagenet_rot_0_cos'];
dataset_all = 'FiltImsAllSFCos'
 
# collect all tuning curves, [nTrainingSchemes x nLayers]
all_units = []

def analyze_orient_tuning(dataset_all, model,param_str,training_str,ckpt_num):
  
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  #%% load full activation patterns (all units, not reduced)
  
  # different versions of the evaluation image set (samples)
  tc = [] 
  for kk in range(nSamples):
  
    if kk==0 and 'FiltIms' not in dataset_all:
      dataset = dataset_all
    elif 'FiltIms' in dataset_all:
      dataset = '%s_rand%d'%(dataset_all,kk+1)
    else:
      dataset = '%s%d'%(dataset_all,kk)
        
    if kk==0:
      info = load_activations.get_info(model,dataset)
      nLayers = info['nLayers']
      layers2load = info['layer_labels_full']     
      layer_labels = info['layer_labels']

    # find the exact name of the checkpoint file of interest
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters    
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'orient_tuning' in ckpt_dirs[jj] and ckpt_num[0:2] in nums[jj][0:2]]
    assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
    ckpt_num_actual= ckpt_dir.split('_')[2][5:]

      
    file_path = os.path.join(root,'activations',model,training_str,param_str,dataset,
                               'eval_at_ckpt-%s_orient_tuning'%(ckpt_num_actual))
    for ll in range(nLayers):

      file_name = os.path.join(file_path,'AllUnitsOrientTuning_%s.npy'%(layers2load[ll]))
      print('loading from %s\n'%file_name)
    
      # [nUnits x nSf x nOri]
    
      t = np.load(file_name)
    
      if kk==0:
        nUnits = np.shape(t)[0]
        nSF = np.shape(t)[1]
        nOri = np.shape(t)[2]
        ori_axis = np.arange(0.5, nOri,1) 
        tc.append(np.zeros([nSamples, nUnits, nSF, nOri]))
        
      tc[ll][kk,:,:,:] = t
    
  #%% count units with zero response
  # going to get rid of the totally non responsive ones to reduce the size of this big matrix.  
  # also going to get rid of any units with no response variance (exactly same response for all stims)
   
  nTotalUnits = np.zeros([nLayers,1])
  propZeroUnits = np.zeros([nLayers,1])
  propConstUnits = np.zeros([nLayers,1])
  
  r = []
  for ll in range(nLayers):
    nUnits = np.shape(tc[ll])[1]
    is_zero = np.zeros([nUnits, nSamples,nSF])
    is_constant_nonzero = np.zeros([nUnits, nSamples, nSF])
    for kk in range(nSamples):
      print('identifying nonresponsive units in %s, sample %d'%(layer_labels[ll],kk))
      for sf in range(nSF):
        # take out data, [nUnits x nOri]
        vals = tc[ll][kk,:,sf,:]
        
        # find units where signal is zero for all ori
        # add these to a running list of which units were zero for any sample and spatial frequency.
        is_zero[:,kk,sf] = np.all(vals==0,axis=1)
        
        # find units where signal is constant for all images (no variance)
        constval = tc[ll][0,:,0,0]
        const = np.all(np.equal(vals, np.tile(np.expand_dims(constval, axis=1), [1,nOri])),axis=1)
        # don't count the zero units here so we can see how many of each...
        is_constant_nonzero[:,kk,sf] = np.logical_and(const, ~np.all(vals==0,axis=1))
        
    is_zero_any = np.any(np.any(is_zero,axis=2),axis=1)  
    propZeroUnits[ll] = np.sum(is_zero_any==True)/nUnits
    
    is_constant_any = np.any(np.any(is_constant_nonzero,axis=2),axis=1)
    propConstUnits[ll] = np.sum(is_constant_any==True)/nUnits

    nTotalUnits[ll] = nUnits
    
    # now put the good units only into a new matrix...
    units2use = np.logical_and(~is_zero_any, ~is_constant_any)
    r.append(tc[ll][:,units2use,:,:])
    
  #%% save the proportion of non-responsive units in each layer
  save_name =os.path.join(save_path,'PropZeroUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,propZeroUnits)
  save_name =os.path.join(save_path,'PropConstUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,propConstUnits)
  save_name =os.path.join(save_path,'TotalUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,nTotalUnits)
  
  #%% Save the units at each layer, after removing non-responsive ones
  for ll in range(nLayers): 
    save_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,r[ll])
  
  #%% count units with tuned responses
  tolerance_deg = 5;
  
  propPosTunedUnits = np.zeros((nSF, nLayers))
  propNegTunedUnits = np.zeros((nSF, nLayers))
  propNotTunedUnits = np.zeros((nSF, nLayers))
  
  # these are all big lists that go [nSF x nLayers]
  # each element is [nUnits long]
  pos_tuned_units = []
  neg_tuned_units = []
  not_tuned_units = []
  pos_peaks_all = []
  neg_peaks_all = []
  

  for sf in range(nSF):
    post = []; negt = [];  nott = []; posp = []; negp = []
    for ll in range(nLayers):
      print('identifying tuned units in %s, sf %d'%(layer_labels[ll],sf))
      # take out data, [nSamples x nUnits x nOri]
      vals = r[ll][:,:,sf,:]
      nUnits = np.shape(vals)[1]
      # save a list of which units are tuned
      has_pos_peak = np.zeros((nUnits,1))
      has_neg_peak = np.zeros((nUnits,1))
      # save the actual peak values (if it's a negative 1, that means it doesn't get filled in)
      pos_peaks = np.ones((nUnits,1))*(-1)
      neg_peaks = np.ones((nUnits,1))*(-1)
#      for uu in range(50):
      for uu in range(nUnits):
#      for uu in np.arange(51,500,50):
        print('uu=%d \ %d'%(uu,nUnits))
        # looking for units whose orientation peak (max) is about the same for all samples. 
        # can say these units have "consistent" tuning peak.
        unitvals = vals[:,uu,:]
        mxinds = np.argmax(unitvals,axis=1) 
        # if the max identified is at index zero, this might be because there's no real max...make sure that's not the case here
        if np.any(mxinds==0) and np.any(unitvals[:,1]==unitvals[:,0]):
          has_pos_peak[uu] = 0
        else:
          # all pairwise differences have to be less than the tolerance value
          mxrep = np.tile(np.expand_dims(mxinds,axis=1), (1, nSamples))
          diffs = np.triu(mxrep-np.transpose(mxrep))
          diffs = np.abs(diffs[diffs!=0])          
          has_pos_peak[uu] = np.all(diffs<tolerance_deg)
          if has_pos_peak[uu]==1:
            pos_peaks[uu] = np.mean(ori_axis[mxinds])
          
        # also looking for units with a consistent negative dip 
        # (probably fewer of these, and they might overlap with above group)    
        mninds = np.argmin(unitvals,axis=1)  
        # if the in identified is at index zero, this might be because there's no real min...make sure that's not the case here
        if np.any(mninds==0) and np.any(unitvals[:,1]==unitvals[:,0]):
          has_neg_peak[uu] = 0
        else:
          # all pairwise differences have to be less than the tolerance value
          mnrep = np.tile(np.expand_dims(mninds,axis=1), (1, nSamples))
          diffs = np.triu(mnrep-np.transpose(mnrep))
          diffs = np.abs(diffs[diffs!=0])          
          has_neg_peak[uu] = np.all(diffs<tolerance_deg)
          if has_neg_peak[uu]==1:
            neg_peaks[uu] = np.mean(ori_axis[mninds])
          
#        plt.figure();plt.plot(ori_axis,np.transpose(unitvals))
#        plt.title('Unit %d, pos tuning=%d, pos peak=%d, neg tuning=%d, neg peak=%d'%
#              (uu,has_pos_peak[uu],pos_peaks[uu],has_neg_peak[uu],neg_peaks[uu]))
        
      pos_inds = np.squeeze(has_pos_peak==1)
      neg_inds = np.squeeze(has_neg_peak==1)
      not_inds = np.squeeze(np.logical_and(has_neg_peak==0, has_pos_peak==0))
      
      # saving a list of which units had positive tuning, negative tuning, or neither.
      post.append(pos_inds)    
      negt.append(neg_inds)    
      nott.append(not_inds)    
       
      # saving a list of the actual peaks for those that have them.
      posp.append(pos_peaks)    
      negp.append(neg_peaks)    

      propPosTunedUnits[sf,ll] = np.sum(has_pos_peak==1)/nUnits
      propNegTunedUnits[sf,ll] = np.sum(has_neg_peak==1)/nUnits
      propNotTunedUnits[sf,ll] = np.sum(np.logical_and(has_neg_peak==0, has_pos_peak==0))/nUnits
      
     
    pos_tuned_units.append(post)
    neg_tuned_units.append(negt)
    not_tuned_units.append(nott)
   
    pos_peaks_all.append(posp)
    neg_peaks_all.append(negp)
    
    #%% save the result of all this analysis
    
    save_name =os.path.join(save_path,'Pos_tuned_units_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,pos_tuned_units)
    save_name =os.path.join(save_path,'Neg_tuned_units_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,neg_tuned_units)
    save_name =os.path.join(save_path,'Un-tuned_units_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,not_tuned_units)
    
    save_name =os.path.join(save_path,'Pos_peaks_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,pos_peaks_all)
    save_name =os.path.join(save_path,'Neg_peaks_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,neg_peaks_all)
    
    save_name =os.path.join(save_path,'PropPosTunedUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,propPosTunedUnits)
    save_name =os.path.join(save_path,'PropNegTunedUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,propNegTunedUnits)

 #%% main function to decide between subfunctions
  
if __name__=='__main__':
  
  for tr in range(np.size(training_str_list)):
    
    training_str = training_str_list[tr]
  
    if 'pretrained' in training_str or 'stop_early' in training_str:
      ckpt_str = '0'
    else:
      ckpt_str = '400000'
    
    analyze_orient_tuning(dataset_all, model, param_str, training_str, ckpt_str)

   