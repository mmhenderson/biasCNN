% make some filtered noise images with varying spatial frequency and orientation content

%% Set up parameters here
clear
close all hidden

randseed = 987987;
% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

image_set = 'FiltNoiseSquare8';
scale_by = 1;

% define where to save the newly created images
image_save_path = fullfile(root,'biasCNN/images/gratings/',image_set);
if ~isdir(image_save_path)
    mkdir(image_save_path)
end

% this is the height and width of the final images
image_size = 224*scale_by;
size_pix = [image_size,image_size];


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

            image_final = uint8(image_scaled.*255);
           
            fn2save = fullfile(thisdir,sprintf('FiltNoise_ex%d_%ddeg.png',ii,orient_vals_deg(oo)));
            fprintf('saving to %s...\n', fn2save)
            imwrite(image_final, fn2save)
        end
        
    end
end
        


