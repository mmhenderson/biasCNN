Code related to the paper:<br />
Henderson, M. & Serences, J.T. (2022) Biased orientation representations can be explained by experience with non-uniform training set statistics. <i>Journal of Vision.</i> https://doi.org/10.1167/jov.21.8.10.
<br /><br />
**Contents**:<br />
- **analysis_code:** python code to compute Fisher information and single-unit tuning properties.
- **get_image_stats:** analyze and plot the distribution of orientations in upright and rotated training images.  
- **make_eval_images:** create evaluation images, by filtering ImageNet images within specified orientation range.
- **make_training_images:** rotate ImageNet images to generate different training sets. 
- **plot_figures:** jupyter notebooks to reproduce figures for all analyses.
- **run_analysis:** shell scripts to launch model "evaluation" and further analysis of network responses.
- **tf_code:** Tensorflow code to train and evaluate the VGG-16 network. Modified from tensorflow slim model library. 
- **train_models:** shell scripts to launch model training.
<br />
Any questions or concerns should be directed to:<br />
mmhender@cmu.edu
