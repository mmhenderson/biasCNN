%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_set = 'CosGratings';

image_path = fullfile(root, 'images','gratings',image_set);
save_path = fullfile(root, 'image_stats','gratings',image_set);

if ~isfolder(save_path)
    mkdir(save_path)
end

% list the ground truth spat freq for gratings
meas_sf_list = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
process_at_size = 224;

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25];
nSF = numel(sf_vals);

ori_vals_deg = linspace(0,179,180);
nOri = numel(ori_vals_deg);
nImsPerOri=1;
phase=0;
ex=1;

%% specify the spatial frequencies and orientations to filter at

[wavelength_list,sorder] = sort(1./meas_sf_list,'ascend');
meas_sf_list = meas_sf_list(sorder);

meas_ori_list = 5:5:180;

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

%% make the filter bank (will be same for all images we look at)

fprintf('making filters...\n')
tic

GaborBank = gabor(wavelength_list.*resize_factor,meas_ori_list);

sizeLargestKernel = size(GaborBank(end).SpatialKernel);
% Gabor always returns odd length kernels
padding_needed = (sizeLargestKernel-1)/2;

max_pix = process_at_size;  % maximum size of any image dimension
% FIX this so that we can make the filters ahead of time
size_after_pad = max_pix*resize_factor+padding_needed*2;
size_after_pad = size_after_pad + mod(size_after_pad,2);
% making a matrix [nPix x nPix x nFilters]
filters_freq = zeros([size_after_pad,length(GaborBank)]);

for p = 1:length(GaborBank)

    H = makeFrequencyDomainTransferFunction_MMH(GaborBank(p),size_after_pad);
    filters_freq(:,:,p) = ifftshift(H);

end
toc

% nIms = zeros(length(true_sflist),1);
total_time = zeros(length(meas_sf_list),6);

%% define more paths

fn2save = fullfile(save_path, sprintf('AllIms_allstats_highdensity.mat'));

%% loop over images

for sf = 1:nSF
    
    clear image_stats
    
    image_folder = dir(fullfile(image_path, sprintf('SF_%.2f*',sf_vals(sf))));     
    fn2save = fullfile(save_path, sprintf('%s_allstats_highdensity.mat',image_folder.name));
    image_folder = fullfile(image_folder(1).folder, image_folder(1).name);

    for oo=1:nOri
        
        
        fn2load = fullfile(image_folder,sprintf('Gaussian_phase%d_ex%d_%ddeg.png',phase,ex,ori_vals_deg(oo)));

        %% loop over images and process

        fprintf('loading from %s\n',fn2load)
        try
            image = imread(fn2load);
        catch err
            fprintf('image %d could not be loaded!\n',ii)
            continue
        end
        image_orig = image;

        % make this params struct to pass into my function
        params.R_MEAN = R_MEAN;
        params.G_MEAN = G_MEAN;
        params.B_MEAN = B_MEAN;

        params.process_at_size = process_at_size;
        params.size_after_pad = size_after_pad;
        params.filters_freq = filters_freq;

        params.ori_list = meas_ori_list;
        params.wavelength_list = wavelength_list;

        %% do the processing in a separate function
        out = process_image(image, params);
        out.true_ori = ori_vals_deg(oo);
        out.true_sf = sf_vals(sf);
        out.mag = [];
        out.phase = [];

        image_stats(oo) = out;
       
    end
    
    save(fn2save, 'image_stats');
    fprintf('saving to %s\n',fn2save);
    
end


