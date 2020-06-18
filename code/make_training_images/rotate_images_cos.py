# -*- coding: utf-8 -*-
"""
Spyder Editor

Rotate images from the ILSVRC2012-CLS dataset by specified increments,
to use as a training set for CNN.

MMH 10/29/2019

"""

import numpy as np
import os
import glob
#import matplotlib.pyplot as plt
from PIL import Image
import shutil

#%% Path information
# find my root directory and define some paths here
root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

# path to where all the original and new images will be
image_folder = os.path.join(root, 'biasCNN','images','ImageNet','ILSVRC2012')

# find all the names of all my synsets, from folder names
synsets = glob.glob(os.path.join(image_folder, 'validation','n*'))
synsets = [synsets[ss].split('/')[-1] for ss in range(np.size(synsets))]

#%% Image information
# final size the images get saved at
final_size = [224, 224]

# making a circular mask with cosine fading to background
cos_mask = np.zeros(final_size)
values = final_size[0]/2*np.linspace(-1,1,final_size[0])
gridx,gridy = np.meshgrid(values,values)
r = np.sqrt(gridx**2+gridy**2)

# creating three ring sections based on distance from center
outer_range = 100
inner_range = 50

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

mask_to_add = np.concatenate((_R_MEAN*np.ones([224,224,1]), _G_MEAN*np.ones([224,224,1]),_B_MEAN*np.ones([224,224,1])), axis=2)
mask_to_add = mask_to_add*(1-mask_image)

#%% which rotations to do here?
rots = [0,22,45]

n_synset_train = 1300
n_synset_val = 50

#%% loop over rotations - create the new images
for rr in rots:
        
    # loop over synsets
    for ss in synsets:
        
        #%% first do validation images
        
        raw_subfolder = os.path.join(image_folder, 'validation',ss)
        rot_subfolder = os.path.join(image_folder, 'validation_rot_%d_cos'%rr, ss )
        
        if not os.path.isdir(rot_subfolder):
            os.mkdir(rot_subfolder)
        else:
            print('folder already exists, deleting it now')
            shutil.rmtree(rot_subfolder)
            os.mkdir(rot_subfolder)
            
        orig_images = os.listdir(raw_subfolder)
        
#        assert(np.size(orig_images)==n_synset_val)
        
        print('working on validation synset %s, rotation %d\n' %(ss, rr))
        print('   saving %d images to %s\n'%(n_synset_val, rot_subfolder))
        
        for ii in range(np.size(orig_images)):
               
            # Read the image file.
            image = Image.open(os.path.join(raw_subfolder, orig_images[ii]))
            
            # trim off the extension. this way all images get saved w same extension even though some are png originally
            image_name = os.path.splitext(orig_images[ii])[0]
            image_name = image_name + '.jpeg'
            
            # this is a cymk image, convert it first                
            if not image.mode == 'RGB':             
                image = image.convert('RGB')
            
            orig_dims = np.asarray(np.shape(image))[0:2]
            smaller_dim = np.min(orig_dims)
            smaller_dim_ind = np.argmin(orig_dims)
            
            # to make things fair across all rotations...we will eventually crop by the same amount regardless of what rotation we apply. 
            # the most extreme cropping will be required for a 45 degree rotation, so determine that box now
            smaller_dim_half = smaller_dim/2
            crop_box_half = np.floor(np.sqrt(np.power(smaller_dim_half, 2)/2))-1            
            image_center = orig_dims/2           
            crop_start = np.ceil(image_center-crop_box_half)
            crop_stop = crop_start + 2*crop_box_half          
            crop_start = crop_start.astype('int')
            crop_stop = crop_stop.astype('int')
            
            # rotate
            image_rot = image.rotate(rr)
            
            # crop
            image_rot_cropped = image_rot.crop((crop_start[1],crop_start[0],crop_stop[1],crop_stop[0]))            

            # scale
            image_rot_resize = image_rot_cropped.resize(final_size)
           
            # mask
            image_rot_masked = mask_image*image_rot_resize
            
            # adjust background
            image_rot_masked_adj = image_rot_masked+mask_to_add

            # put this new array in an image and save it
            image_final = Image.fromarray(image_rot_masked_adj.astype('uint8'))            
            image_final.save(os.path.join(rot_subfolder, image_name),format='JPEG')
            
        #%% next do training images
            
        raw_subfolder = os.path.join(image_folder, 'train',ss)
        rot_subfolder = os.path.join(image_folder, 'train_rot_%d_cos'%rr, ss )
        
        if not os.path.isdir(rot_subfolder):
            os.mkdir(rot_subfolder)
        else:
            print('folder already exists, deleting it now')
            shutil.rmtree(rot_subfolder)
            os.mkdir(rot_subfolder)
            
        orig_images = os.listdir(raw_subfolder)
        
#        assert(np.size(orig_images)==n_synset_train)
        
        print('working on training synset %s, rotation %d\n' %(ss, rr))
        print('   saving %d images to %s\n'%(n_synset_train, rot_subfolder))
        
        for ii in range(np.size(orig_images)):
               
             # Read the image file.
            image = Image.open(os.path.join(raw_subfolder, orig_images[ii]))
            
            # trim off the extension. this way all images get saved w same extension even though some are png originally
            image_name = os.path.splitext(orig_images[ii])[0]
            image_name = image_name + '.jpeg'
            
            # this is a cymk image, convert it first                
            if not image.mode == 'RGB':             
                image = image.convert('RGB')
             
            orig_dims = np.asarray(np.shape(image))[0:2]
                
            smaller_dim = np.min(orig_dims)
            smaller_dim_ind = np.argmin(orig_dims)
            
             
            # to make things fair across all rotations...we will eventually crop by the same amount regardless of what rotation we apply. 
            # the most extreme cropping will be required for a 45 degree rotation, so determine that box now
            smaller_dim_half = smaller_dim/2
            crop_box_half = np.floor(np.sqrt(np.power(smaller_dim_half, 2)/2))-1            
            image_center = orig_dims/2           
            crop_start = np.ceil(image_center-crop_box_half)
            crop_stop = crop_start + 2*crop_box_half          
            crop_start = crop_start.astype('int')
            crop_stop = crop_stop.astype('int')
            
            # rotate
            image_rot = image.rotate(rr)
            
            # crop
            image_rot_cropped = image_rot.crop((crop_start[1],crop_start[0],crop_stop[1],crop_stop[0]))            

            # scale
            image_rot_resize = image_rot_cropped.resize(final_size)
           
            # mask
            image_rot_masked = mask_image*image_rot_resize
            
            # adjust background
            image_rot_masked_adj = image_rot_masked+mask_to_add

            # put this new array in an image and save it
            image_final = Image.fromarray(image_rot_masked_adj.astype('uint8'))            
            image_final.save(os.path.join(rot_subfolder, image_name),format='JPEG')
            
   