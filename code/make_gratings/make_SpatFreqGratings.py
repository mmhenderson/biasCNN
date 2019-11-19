# -*- coding: utf-8 -*-
"""
Spyder Editor

Make grating images with a specified orientation, spatial frequency, contrast 
and amount of noise (the SpatFreqGratings dataset)

MMH 11/14/19

"""

import numpy as np
import os
#import glob
#import matplotlib.pyplot as plt
from PIL import Image
#import shutil

# find my root directory and define some paths here
root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

# mask file for my smoothed circular window
mask_file = os.path.join(root,'biasCNN','code','image_proc_code','Smoothed_mask.png')

# path to where all the images will get saved
image_folder = os.path.join(root, 'biasCNN','images','gratings','SpatFreqGratings')

# final size the images get saved at
final_size = [224, 224]
image_size = final_size[0]

# this is a mask of range 0-255 - use this to window the image
mask_image = Image.open(mask_file)            
mask_image = np.tile(np.expand_dims(np.array(mask_image),2),[1,1,3])
mask_image = mask_image/255 # change to 0-1 range

# also want to change the background color from 0 (black) to a mid gray color 
# (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
# will be subtracted when the images are centered during preproc.
_R_MEAN = 124
_G_MEAN = 117
_B_MEAN = 104

mask_to_add = np.concatenate((_R_MEAN*np.ones([224,224,1]), _G_MEAN*np.ones([224,224,1]),_B_MEAN*np.ones([224,224,1])), axis=2)
mask_to_add = mask_to_add*(1-mask_image)

#%% enter parameters here

# what spatial frequencies do you want? these will each be in a separate
#  folder. Units are cycles per pixel.
freq_levels_cpp_orig = np.logspace(np.log10(0.02),np.log10(0.4),6);
# adjusting these so that they'll be directly comparable with an older
# version of the experiment (in which we had smaller 140x140 images)
freq_levels_cycles_per_image = freq_levels_cpp_orig*140;

# these are the actual cycles-per-pixel that we want, so that we end up
# with the same number of cycles per image as we had in the older version.
freq_levels_cpp = freq_levels_cycles_per_image/image_size;

# specify different amounts of noise
noise_levels = [0.01];
nn=0

# specify different contrast levels
contrast_levels = [0.8];

# how many random instances do you want to make?
numInstances = 4;

# two opposite phases
phase_levels = [0,180];

# start with a meshgrid
X=np.arange(-0.5*image_size+.5, .5*image_size-.5+1, 1)
Y=np.arange(-0.5*image_size+.5, .5*image_size-.5+1, 1)
[x,y] = np.meshgrid(X,Y);

#%% loop and make images


for cc in range(np.size(contrast_levels)):
        
    for ff in range(np.size(freq_levels_cpp)):
        
        
        thisdir = os.path.join(image_folder, 'SF_%.2f_Contrast_%.2f/'%(freq_levels_cpp[ff], contrast_levels[cc]));
        if not os.path.isdir(thisdir):
            os.mkdir(thisdir)        

        this_freq_cpp = freq_levels_cpp[ff];

        orient_vals = np.linspace(0,179,180);
     
        for oo in range(np.size(orient_vals)):

            for pp in range(np.size(phase_levels)):
                
                phase_vals = np.ones([numInstances,1])*phase_levels[pp]*np.pi*180
                
                for ii in range(numInstances):
                    
                    #%% make the full field grating
                    # range is [-1,1] to start
                    sine = (np.sin(this_freq_cpp*2*np.pi*(y*np.sin(orient_vals[oo]*np.pi/180)+x*np.cos(orient_vals[oo]*np.pi/180))-phase_vals[ii]));

                    # make the values range from 1 +/-noise to
                    # -1 +/-noise
                    sine = sine + np.random.normal(0,1,np.shape(sine))*noise_levels[nn];

                    # now scale it down (note the noise also gets scaled)
                    sine = sine*contrast_levels[cc];

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
                    
                    fn2save = os.path.join(thisdir, 'Gaussian_phase%d_ex%d_%ddeg.jpeg'%(phase_levels[pp],ii+1,orient_vals[oo]))
                    print('saving to %s... \n'%fn2save)
                    image_final.save(fn2save,format='JPEG')
               