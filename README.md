# biasCNN
Evaluating low-level perceptual biases in convolutional neural networks.

code/analysis_code
  Contains all code to compute Fisher information and single-unit tuning properties.
  
code/eval_models
  Shell scripts to launch model evaluation.
  
code/get_image_stats
  Analyze and plot the distribution of orientations in upright and rotated training images.
  
code/make_eval_images
  Create grating-like images by filtering within narrow orientation range.
  
code/make_training_images
  Rotate ImageNet images to generate different training sets.
  
code/tf_code
  Tensorflow code to train and evaluate the VGG-16 network. Modified from tensorflow slim model library.
  
code/train_models
  Shell scripts to launch model training.
