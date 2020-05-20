# -*- coding: utf-8 -*-
"""
Spyder Editor

Make grating images with a specified orientation, spatial frequency, contrast 
and amount of noise (the CircGratings dataset)
These ones do have the circular smoothed window, and they DO have phase jitter 
(each set has two phases, 180 deg apart, but each set's phase pair is jittered)

MMH 1/27/20

"""

import numpy as np
import os
#import glob
#import matplotlib.pyplot as plt
from PIL import Image
#import shutil
# find my root directory and define some paths here
root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

#%% set parameters for what is in this image set
nSetsEach = 1
seeds=[895748]
nSF = 6;
nPhasePairs = 24
scale_by = 1
image_set = 'PhaseVaryingCosGratings'
#%% Define spatial frequencies: each one is a different image set, so we can have a bunch of phases in each
 #what spatial frequencies do you want? these will each be in a separate
#  folder. Units are cycles per pixel.
freq_levels_cpp_orig = np.logspace(np.log10(0.02),np.log10(0.4),nSF);
# adjusting these so that they'll be directly comparable with an older
# version of the experiment (in which we had smaller 140x140 images)
freq_levels_cycles_per_image = freq_levels_cpp_orig*140;

# final size the images get saved at
final_size = [224*scale_by, 224*scale_by]
image_size = final_size[0]
# these are the actual cycles-per-pixel that we want, so that we end up
# with the same number of cycles per image as we had in the older version.
freq_levels_cpp = freq_levels_cycles_per_image/image_size;
# these are relative to the "standard" image size of 224, so that folder names match up even when image sizes are different 
freq_levels_cpp_print = freq_levels_cycles_per_image/224
#%% more parameters
# specify different amounts of noise
noise_level = 0.01;
# specify different contrast levels
contrast_level = 0.8;

# this is the number of phase "pairs" - each of these pairs will be  a complementary pair, 
# such that together it makes a complete 360 degree rotation about the center.
phase_values = np.linspace(0,90,nPhasePairs)

# list out all the orientations
orient_vals = np.linspace(0,179,180);  

# create a meshgrid for making all stimuli
X=np.arange(-0.5*image_size+.5, .5*image_size-.5+1, 1)
Y=np.arange(-0.5*image_size+.5, .5*image_size-.5+1, 1)
[x,y] = np.meshgrid(X,Y);

#%% parameters of the image set
# making a circular mask with cosine fading to background
cos_mask = np.zeros(final_size)
values = final_size[0]/2*np.linspace(-1,1,final_size[0])
gridx,gridy = np.meshgrid(values,values)
r = np.sqrt(gridx**2+gridy**2)

# creating three ring sections based on distance from center
outer_range = 100*scale_by
inner_range = 50*scale_by

# inner values: set to 1
cos_mask[r<inner_range] = 1
faded_inds = np.logical_and(r>=inner_range, r<outer_range)

# middle values: create a smooth fade
cos_mask[faded_inds] = 0.5*np.cos(np.pi/(outer_range-inner_range)*(r[faded_inds]-inner_range)) + 0.5

# outer values: set to 0
cos_mask[r>=outer_range] = 0

# make it three color channels
mask_image = np.tile(np.expand_dims(cos_mask,2),[1,1,3])

# also want to change the background color from 0 (black) to a mid gray color 
# (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
# will be subtracted when the images are centered during preproc.
_R_MEAN = 124
_G_MEAN = 117
_B_MEAN = 104

mask_to_add = np.concatenate((_R_MEAN*np.ones([image_size,image_size,1]), _G_MEAN*np.ones([image_size,image_size,1]),_B_MEAN*np.ones([image_size,image_size,1])), axis=2)
mask_to_add = mask_to_add*(1-mask_image)

#%% LOOP OVER SETS, MAKE nSetsEach FOLDERS

for ss in range(nSetsEach):
  
  np.random.seed(seeds[ss])
  
  # loop over SF
  for sf in range(nSF):

      # here, making a distinct image set for each spatial frequency.
      image_folder = os.path.join(root, 'biasCNN','images','gratings','%s_SF_%0.2f'%(image_set,freq_levels_cpp_print[sf]))
  
      if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
          
      # loop over phases
      for pp in range(nPhasePairs):
               
        thisdir = os.path.join(image_folder, 'Phase_%.2f/'%(phase_values[pp]));
        if not os.path.isdir(thisdir):
            os.mkdir(thisdir)   
            
            this_phase_pair_deg = np.array([phase_values[pp], 180-phase_values[pp]])    
            
            for oo in range(np.size(orient_vals)):

                # now looping over the two complementary phases in this pair - 
                # together, these two phases will make up a full 360 degree rotation.
                for xx in range(np.size(this_phase_pair_deg)):
                    
                    this_phase_radians = this_phase_pair_deg[xx]*np.pi/180
                    
                  
                    #%% make the full field grating
                    # range is [-1,1] to start
                    sine = (np.sin(freq_levels_cpp[sf]*2*np.pi*(y*np.sin(orient_vals[oo]*np.pi/180)+x*np.cos(orient_vals[oo]*np.pi/180))-this_phase_radians));

                    # make the values range from 1 +/-noise to
                    # -1 +/-noise
                    sine = sine + np.random.normal(0,1,np.shape(sine))*noise_level;

                    # now scale it down (note the noise also gets scaled)
                    sine = sine*contrast_level;

                    # shouldnt ever go outside the range [-1,1] so values won't
                    # get cut off (requires that noise is low if contrast is
                    # high)
                    assert np.max(sine)<=np.float64(1) 
                    assert np.min(sine)>=np.float64(-1)

                    # change the scale from [-1, 1] to [0,1]
                    # the center is exactly 0.5 - note the values may not
                    # span the entire range [0,1] but will be centered at
                    # 0.5.
                    stim_scaled = (sine+1)/2;

                    # convert from [0,1] to [0,255]
                    stim_scaled = stim_scaled*255;
            
                    # double check my size
                    assert np.shape(stim_scaled)[0]==image_size
                    assert np.shape(stim_scaled)[1]==image_size
                        
                    stim_masked = np.tile(np.expand_dims(stim_scaled,axis=2),3)*mask_image
                               
                    stim_masked_adj = stim_masked+mask_to_add
                    
                    assert(np.all(np.squeeze(stim_masked_adj[0,0])==[_R_MEAN,_G_MEAN,_B_MEAN]))
                    
                    # put this new array in an image and save it
                    image_final = Image.fromarray(stim_masked_adj.astype('uint8'))     
                    
                    fn2save = os.path.join(thisdir, 'CosineWindow_PhaseComplement%d_%ddeg.png'%(xx,orient_vals[oo]))
                    print('saving to %s... \n'%fn2save)
                    image_final.save(fn2save,format='PNG')
                   