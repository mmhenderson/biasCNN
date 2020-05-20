% make some filtered noise images with varying spatial frequency and orientation content

%% Set up parameters here
clear
close all hidden

randseed = 435679;
% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

image_set = 'FiltNoiseCos8';
scale_by = 1;

% define where to save the newly created images
image_save_path = fullfile(root,'biasCNN/images/gratings/',image_set);
if ~isdir(image_save_path)
    mkdir(image_save_path)
end

% this is the height and width of the final images
image_size = 224*scale_by;
size_pix = [image_size,image_size];

%% making a circular mask with cosine fading to background
cos_mask = zeros(image_size);
values = image_size./2*linspace(-1,1,image_size);
[gridx,gridy] = meshgrid(values,values);
r = sqrt(gridx.^2+gridy.^2);
% creating three ring sections based on distance from center
outer_range = 100*scale_by;
inner_range = 50*scale_by;
% inner values: set to 1
cos_mask(r<inner_range) = 1;
% middle values: create a smooth fade
faded_inds = r>=inner_range & r<outer_range;
cos_mask(faded_inds) = 0.5*cos(pi/(outer_range-inner_range).*(r(faded_inds)-inner_range)) + 0.5;
% outer values: set to 0
cos_mask(r>=outer_range) = 0;

% make it three color channels
mask_image = repmat(cos_mask,1,1,3);

% also want to change the background color from 0 (black) to a mid gray color 
% (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
% will be subtracted when the images are centered during preproc.
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

mask_to_add = permute(repmat([R_MEAN,G_MEAN,B_MEAN], image_size,1,image_size),[1,3,2]);
mask_to_add = mask_to_add.*(1-mask_image);


%% enter parameters here
orient_vals_deg = linspace(0,179,180);
nOrient = numel(orient_vals_deg);
% what spatial frequencies do you want? these will each be in a separate
% folder. Units are cycles per pixel.
freq_levels_cpp_orig = logspace(log10(0.02),log10(0.4),6);
% adjusting these so that they'll be directly comparable with an older
% version of the experiment (in which we had smaller 140x140 images)
freq_levels_cycles_per_image = freq_levels_cpp_orig*140;

% these are the actual cycles-per-pixel that we want, so that we end up
% with the same number of cycles per image as we had in the older version.
freq_levels_cpp = freq_levels_cycles_per_image/image_size;

nSF = numel(freq_levels_cpp);
% specify different contrast levels
contrast_levels = [0.8];

% how many random instances do you want to make?
numInstances = 4;

%% define some parameters that are the same for all images
params.freq_sd_cpp = 0.005;
params.orient_kappa_deg = 1000;       
params.size_pix = size_pix;
nImsAtATime = 8;
%% loop over images 

for ff = 1:nSF
    
    thisdir = fullfile(image_save_path,sprintf('SF_%.2f/', freq_levels_cpp(ff)));
    if ~isdir(thisdir)
        mkdir(thisdir)
    end
    
    for oo = 1:nOrient

        params.freq_mean_cpp = freq_levels_cpp(ff);
        params.orient_mean_deg = orient_vals_deg(oo);
        
        randseed = randseed+1;
        
        images = get_filtered_noise(params, nImsAtATime, randseed);
        
        % loop over the individual images and save them
        for ii=1:nImsAtATime
            image = images{ii};
            % rescale this image now to 0-1
            % originally the points were gaussian distributed, with mean of 0
            % and SD of 1, so scale down a bit.
            image_scaled = image./max(abs(image(:)));
            image_scaled = (image_scaled+1)./2;

            % now mask with the smoothed circle
            image_masked = repmat(image_scaled,1,1,3).*mask_image;
            image_masked = (image_masked*255)+mask_to_add;
            image_masked = uint8(image_masked);
            fn2save = fullfile(thisdir,sprintf('FiltNoise_ex%d_%ddeg.png',ii,orient_vals_deg(oo)));
            fprintf('saving to %s...\n', fn2save)
            imwrite(image_masked, fn2save)
        end
        
    end
end
        


