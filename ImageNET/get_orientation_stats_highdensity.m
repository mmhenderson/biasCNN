%% get the orientation content of a subset of the images from ImageNET database
% these are the images that were used to train VGG-16 in 2014 ILSVRC (DET training set)

clear
close all

if isempty(gcp('nocreate'))
    parpool(8);
end

root = pwd;

% find all the folders in this directory
image_folders = [dir(fullfile(root, 'ILSVRC2014_DET_train','n*')); dir(fullfile(root, 'ILSVRC2014_DET_train','ILSVRC*'))];
good_inds = find(~contains({image_folders.name},'.tar'));
image_folders = image_folders(good_inds);

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

syn_file = fullfile(root, 'ILSVRC2014_devkit/data/meta_det.mat');
load(syn_file)

% how big are the images when we do all this processing? this is the same
% as the size that VGG-16 preprocesing resizes them to.
process_at_size = 224;
%% specify the spatial frequencies and orientations to filter at

freq_list = logspace(log10(0.02), log10(.2),10);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;

all_im_fns = [];

%% make the filter bank 

fprintf('making filters...\n')
tic

GaborBank = gabor(wavelength_list.*resize_factor,ori_list);
freq_inds = repelem(1:length(freq_list),numel(ori_list));
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
total_time = zeros(length(image_folders),6);
%% loop over images
for ff=1:length(image_folders)
    
    
    %% figure out which set of images is in this folder
    if image_folders(ff).name(1)=='n'
        ind = find(ismember({synsets.WNID},image_folders(ff).name));
        if ~isempty(ind)
            info = synsets(ff);
            fprintf('processing synset %d, %s\n',ind, synsets(ind).name);
            fn2save = fullfile(root, 'ImageStats',sprintf('synset%d_allstats_highdensity.mat',ind));
        else
            info = [];
            fprintf('processing synset %s\n',image_folders(ff).name);
            fn2save = fullfile(root, 'ImageStats',sprintf('synset_%s_allstats_highdensity.mat',image_folders(ff).name));   
        end
    else
        info = [];
        fprintf('processing images in %s\n',image_folders(ff).name);
        fn2save = fullfile(root, 'ImageStats',sprintf('%s_allstats_highdensity.mat',image_folders(ff).name));   
    end
   
    image_folder = fullfile(image_folders(ff).folder, image_folders(ff).name);
    
    imlist = dir([image_folder, filesep, '*.JPEG']);
    
    fprintf('found %d images in folder\n',length(imlist));
    
    nIms(ff) = length(imlist);

    image_stats = [];
    c_start_all = clock;
    
    %% loop over images and process
    parfor ii = 1:length(imlist)
       
        fprintf('loading image %d of %d\n',ii,length(imlist));
        im_file = fullfile(imlist(ii).folder, imlist(ii).name);
        try
            image = imread(im_file);
        catch err
            fprintf('image %d could not be loaded!\n',ii)
            continue
        end
        image_orig = image;
       
        %% Pre-processing
        
        % make it grayscale
        if size(image,3)==3
            image=rgb2gray(image);
        end
        orig_size = size(image);
        
        % resize it so the smallest side is 256
        % this is equivalent to the function _aspect_preserving_resize
        scaled_size = 256; % this is equivalent to _RESIZE_SIDE_MIN
        [smaller_dim, smaller_dim_ind] = min(orig_size);
        image = imresize(image, scaled_size/smaller_dim);
        assert(size(image,smaller_dim_ind)==scaled_size);
        scaled_size = size(image);
        
        % now crop it to exactly 224 x 224
        % this is equivalent to _central_crop
        crop_by = floor(max(((scaled_size-process_at_size)/2),0));
        inds_1 = crop_by(1)+1:crop_by(1)+process_at_size;
        inds_2 = crop_by(2)+1:crop_by(2)+process_at_size;
        image = image(inds_1, inds_2);
        cropped_size = size(image);
        assert(all(cropped_size==process_at_size))
        
        % z score all the luminance values across all pixels
        image = im2double(image);
        image = reshape(zscore(image(:)),size(image));
        
        % pad it so we can apply the filters at the correct size
        pad_by = (size_after_pad - size(image))./2;        
        n2pad = [floor(pad_by'), ceil(pad_by')];        
        image_padded = [repmat(image(:,1), 1, n2pad(2,1)), image, repmat(image(:,end), 1, n2pad(2,2))];
        image_padded = [repmat(image_padded(1,:), n2pad(1,1), 1); image_padded; repmat(image_padded(end,:), n2pad(1,2),1)];
        
        padded_size = size(image_padded);
        assert(all(padded_size==size_after_pad));
        
        %% Filtering
        
        % fft into frequency domain
        image_fft = fft2(image_padded);

        % Apply all my filters all at once
        filtered_freq_domain = image_fft.*filters_freq;

        % get back to the spatial domain
        out = ifft2(filtered_freq_domain);
        
        % un-pad the image (back to its down-sampled size)
        out = out(n2pad(1,1)+1:n2pad(1,1)+cropped_size(1), n2pad(2,1)+1:n2pad(2,1)+cropped_size(2),:);
        assert(size(out,1)==cropped_size(1) && size(out,2)==cropped_size(2));
        
        mag = abs(out);
        phase = angle(out);
        
        %%  add all this info to my structure
       
        image_stats(ii).mean_phase = squeeze(mean(mean(phase,2),1));
        image_stats(ii).mean_mag = squeeze(mean(mean(mag,2),1));
        image_stats(ii).ori_list = ori_list;
        image_stats(ii).wavelength_list = wavelength_list;
        image_stats(ii).orig_size = orig_size;
        image_stats(ii).cropped_size = cropped_size;
        image_stats(ii).scaled_size = scaled_size;
        image_stats(ii).padded_size = padded_size;

    end
    c_end_all = clock;
    total_time(ff,:) = c_end_all - c_start_all;
    save(fn2save, 'info','image_stats','total_time');
    
end