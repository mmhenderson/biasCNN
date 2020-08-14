# biasCNN
Evaluating low-level perceptual biases in convolutional neural networks.

### code

#### analysis_code 
   Contains all code to compute Fisher information and single-unit tuning properties.
  
#### eval_models
   Shell scripts to launch model evaluation.
  
#### get_image_stats
   Analyze and plot the distribution of orientations in upright and rotated training images.
  
#### make_eval_images
   Create grating-like images by filtering within narrow orientation range.
  
#### make_training_images
    Rotate ImageNet images to generate different training sets.
  
#### tf_code
    Tensorflow code to train and evaluate the VGG-16 network. Modified from tensorflow slim model library.
  
#### train_models
    Shell scripts to launch model training.
