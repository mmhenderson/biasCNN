# -*- coding: utf-8 -*-
"""
Spyder Editor

rotate images from the ILSVRC2012-CLS dataset by specified increments, so that 
we can re-train a neural network with rotated images and see how cardinal biases change. 
This version saves square images that are all cropped to same size.

MMH 10/29/2019

"""

import numpy as np
import os
import glob
#import matplotlib.pyplot as plt
from PIL import Image
import shutil

# find my root directory and define some paths here
root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

# mask file for my smoothed circular window
#mask_file = os.path.join(root,'biasCNN','code','image_proc_code','Smoothed_mask.png')

# path to where all the original and new images will be
image_folder = os.path.join(root, 'biasCNN','images','ImageNet','ILSVRC2012')

# find all the names of all my synsets, from folder names
synsets = glob.glob(os.path.join(image_folder, 'validation','n*'))
synsets = [synsets[ss].split('/')[-1] for ss in range(np.size(synsets))]

# final size the images get saved at
final_size = [224, 224]

# which rotations to do here?
rots = [0,22,45]

n_synset_train = 1300
n_synset_val = 50

for rr in rots:
        
    rot_folder_val = os.path.join(image_folder, 'validation_rot_%d_square'%rr)
        
    if not os.path.isdir(rot_folder_val):
        os.mkdir(rot_folder_val)
        
    rot_folder_train = os.path.join(image_folder, 'train_rot_%d_square'%rr)
        
    if not os.path.isdir(rot_folder_train):
        os.mkdir(rot_folder_train)
        
    for ss in synsets:
#    for ss in synsets[0:1]:
        
        #%% first do validation images
        
        raw_subfolder = os.path.join(image_folder, 'validation',ss)
        rot_subfolder = os.path.join(image_folder, 'validation_rot_%d_square'%rr, ss )
        
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
           
            image_rot_resize = np.reshape(image_rot_resize.getdata(),(final_size[0],final_size[1],3))
            # mask
#            image_rot_masked = mask_image*image_rot_resize
#            
#            # adjust background
#            image_rot_masked_adj = image_rot_masked+mask_to_add

            # put this new array in an image and save it
            image_final = Image.fromarray(image_rot_resize.astype('uint8'))            
            image_final.save(os.path.join(rot_subfolder, image_name),format='JPEG')
            
        #%% next do training images
            
        raw_subfolder = os.path.join(image_folder, 'train',ss)
        rot_subfolder = os.path.join(image_folder, 'train_rot_%d_square'%rr, ss )
        
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
           
            image_rot_resize = np.reshape(image_rot_resize.getdata(),(final_size[0],final_size[1],3))
            # mask
#            image_rot_masked = mask_image*image_rot_resize
#            
#            # adjust background
#            image_rot_masked_adj = image_rot_masked+mask_to_add

            # put this new array in an image and save it
            image_final = Image.fromarray(image_rot_resize.astype('uint8'))            
            image_final.save(os.path.join(rot_subfolder, image_name),format='JPEG')
            
                        
            #%% extra code for visualizing
            
            #            #%%
#            plt.figure();plt.imshow(image_rot)
#            plt.plot(crop_start[1],crop_start[0],'ro')           
#            plt.plot(crop_stop[1],crop_start[0],'ro')
#            plt.plot(crop_stop[1],crop_stop[0],'ro')
#            plt.plot(crop_start[1],crop_stop[0],'ro')
#            #%%
#            plt.figure();plt.imshow(image_rot_cropped)
#            
#            