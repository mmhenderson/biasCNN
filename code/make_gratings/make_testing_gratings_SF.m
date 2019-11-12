% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project
clear
close all

rndseed = 837987;
rng(rndseed)

% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

% define where to save the newly created images
image_save_path = fullfile(root,'biasCNN/images/gratings/SpatFreqGratings/');
if ~isdir(image_save_path)
    mkdir(image_save_path)
end

mask_file = fullfile(root,'biasCNN/code/image_proc_code/Smoothed_mask.png');

% this is a mask of range 0-255 - use this to window the image
mask_image = imread(mask_file);     
mask_image = repmat(mask_image,1,1,3);
mask_image = double(mask_image)./255; % change to 0-1 range

% also want to change the background color from 0 (black) to a mid gray color 
% (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
% will be subtracted when the images are centered during preproc.
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

mask_to_add = cat(3, R_MEAN*ones(224,224,1),G_MEAN*ones(224,224,1),B_MEAN*ones(224,224,1));
mask_to_add = mask_to_add.*(1-mask_image);
%% enter parameters here

% what spatial frequencies do you want? these will each be in a separate
% folder. Units are cycles per pixel.
freq_levels_cpp = logspace(log10(0.02),log10(0.4),6);

% specify different amounts of noise
noise_levels = [0.01];

% specify different contrast levels
contrast_levels = [0.8];

% how many random instances do you want to make?
numInstances = 4;

% two opposite phases
phase_levels = [0,180];

% this is the height and width of the final images
image_size = 224;

% start with a meshgrid
X=-0.5*image_size+.5:1:.5*image_size-.5; Y=-0.5*image_size+.5:1:.5*image_size-.5;
[x,y] = meshgrid(X,Y);


%% make and save the individual images
nn=1;

for cc=1:length(contrast_levels)

    for ff = 1:length(freq_levels_cpp)

        thisdir = sprintf('%sSF_%.2f_Contrast_%.2f/', image_save_path, freq_levels_cpp(ff), contrast_levels(cc));
        if ~isdir(thisdir)
            mkdir(thisdir)
        end

        this_freq_cpp = freq_levels_cpp(ff);

        orient_vals = linspace(0,179,180);

        for oo=1:length(orient_vals)
            
            for pp=1:length(phase_levels)

                phase_vals = ones(numInstances,1)*phase_levels(pp)*pi/180;

                for ii = 1:numInstances

                    %% make the full field grating
                    % range is [-1,1] to start
                    sine = (sin(this_freq_cpp*2*pi*(y.*sin(orient_vals(oo)*pi/180)+x.*cos(orient_vals(oo)*pi/180))-phase_vals(ii)));

                    % make the values range from 1 +/-noise to
                    % -1 +/-noise
                    sine = sine+ randn(size(sine))*noise_levels(nn);

                    % now scale it down (note the noise also gets scaled)
                    sine = sine*contrast_levels(cc);

                    % shouldnt ever go outside the range [-1,1] so values won't
                    % get cut off (requires that noise is low if contrast is
                    % high)
                    assert(max(sine(:))<=1 && min(sine(:))>=-1)

                    % change the scale from [-1, 1] to [0,1]
                    % the center is exactly 0.5 - note the values may not
                    % span the entire range [0,1] but will be centered at
                    % 0.5.
                    stim_scaled = (sine+1)./2;

                    % convert from [0,1] to [0,255]
                    stim_scaled = stim_scaled.*255;
                    
                    % now multiply it by the donut (circle) to get gaussian envelope
                    stim_masked = stim_scaled.*mask_image;
                    
                    % finally add a mid-gray background color.
                    stim_masked_adj = uint8(stim_masked + mask_to_add);
                   
                    assert(all(squeeze(stim_masked_adj(1,1,:))==[R_MEAN;G_MEAN;B_MEAN]))
                    
                    fn2save = fullfile(thisdir,sprintf('Gaussian_phase%d_ex%d_%ddeg.png',phase_levels(pp),ii,orient_vals(oo)));

                    imwrite(stim_masked_adj, fn2save)
                    fprintf('saving to %s...\n', fn2save)
                    
                end
            end
        end
    end
end