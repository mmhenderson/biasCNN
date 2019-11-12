%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

% if isempty(gcp('nocreate'))
%     parpool(8);
% end

curr_folder = pwd;
filesepinds=find(curr_folder==filesep);
root = curr_folder(1:filesepinds(end));
im_path = fullfile(root, 'images','SpatFreqGratings');

% where will i save these image statistics?
folder2save = fullfile(curr_folder, 'image_stats', sprintf('SpatFreqGratings'));
if ~isfolder(folder2save)
    mkdir(folder2save)
end

% list the ground truth spat freq for gratings
true_sflist = [0.02, 0.04, 0.07, 0.12, 0.22, 0.40];

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
process_at_size = 224;

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

%% specify the spatial frequencies and orientations to filter at

freq_list = logspace(log10(0.02), log10(.2),4);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;

%% make the filter bank (will be same for all images we look at)

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

nIms = zeros(length(true_sflist),1);
total_time = zeros(length(true_sflist),6);

%% now loop over the three different training sets
for ff = 1:length(true_sflist)
    
    % find all the folders in this directory
    image_folder = dir(fullfile(im_path, sprintf('*%.2f*',true_sflist(ff))));
    
    fn2save = fullfile(folder2save, sprintf('%s_allstats_highdensity.mat',image_folder.name));
  
    %% loop over images
   
    fprintf('processing folder %d of %d\n',ff,length(true_sflist));
        
    imlist = dir(fullfile(image_folder.folder, image_folder.name, '*.png'));

    fprintf('found %d images in folder\n',length(imlist));

    nIms(ff) = length(imlist);

    image_stats = [];
    c_start_all = clock;
    ori_done = [];
    %% loop over images and process
    for ii = 1:length(imlist)

        fprintf('loading image %d of %d\n',ii,length(imlist));
        im_file = fullfile(imlist(ii).folder, imlist(ii).name);
        
        % figure out the orientation from the name here
        ind1 = find(im_file=='_');ind1 =ind1(end);
        ind2 = strfind(im_file, 'deg');
        true_ori = im_file(ind1+1:ind2-1);
        true_ori = str2double(true_ori);
        if any(true_ori==ori_done)
            fprintf('skipping %s\n',im_file)
            continue
        else
            fprintf('loading %s\n',im_file);
            ori_done = [ori_done, true_ori];
        end
        
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

        assert(orig_size(1)==process_at_size && orig_size(2)==process_at_size)

        % z score all the luminance values across all pixels
        image = im2double(image);
        image = reshape(zscore(image(:)),size(image));

        % pad it so we can apply the filters at the correct size
        pad_by = (size_after_pad - size(image))./2;        
        n2pad = [floor(pad_by'), ceil(pad_by')];        
        image_padded = [repmat(image(:,1), 1, n2pad(2,1)), image, repmat(image(:,end), 1, n2pad(2,2))];
        image_padded = [repmat(image_padded(1,:), n2pad(1,1), 1); image_padded; repmat(image_padded(end,:), n2pad(1,2),1)];
%         image_padded = [repmat(zeros(size(image(:,1))), 1, n2pad(2,1)), image, repmat(zeros(size(image(:,end))), 1, n2pad(2,2))];
%         image_padded = [repmat(zeros(size(image_padded(1,:))), n2pad(1,1), 1); image_padded; repmat(zeros(size(image_padded(end,:))), n2pad(1,2),1)];

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
        out = out(n2pad(1,1)+1:n2pad(1,1)+process_at_size, n2pad(2,1)+1:n2pad(2,1)+process_at_size,:);
        assert(size(out,1)==process_at_size && size(out,2)==process_at_size);

        mag = abs(out);
        phase = angle(out);

        %%  add all this info to my structure

        image_stats(ii).mean_phase = squeeze(mean(mean(phase,2),1));
        image_stats(ii).mean_mag = squeeze(mean(mean(mag,2),1));
        image_stats(ii).ori_list = ori_list;
        image_stats(ii).wavelength_list = wavelength_list;
        image_stats(ii).orig_size = orig_size;
%             image_stats(ii).scaled_size = scaled_size;
%             image_stats(ii).cropped_size = cropped_size;
        image_stats(ii).padded_size = padded_size;
        image_stats(ii).true_ori = true_ori;
        image_stats(ii).true_sf = true_sflist(ff);

    end
    c_end_all = clock;

    total_time(ff,:) = c_end_all - c_start_all;
    save(fn2save, 'image_stats','total_time');
   
end