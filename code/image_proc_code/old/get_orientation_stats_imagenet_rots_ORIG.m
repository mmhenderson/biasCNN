%% get the orientation content of a subset of the images from ImageNET database
% making sure that our image set rotations affected the prior in the
% expected way (peak at cardinals, or cardinals+rotation)
%%

clear
close all

if isempty(gcp('nocreate'))
    parpool(8);
end

rot_list = [45];
% rot_list = [0,22,45];

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root,'images','ImageNet','ILSVRC2012');
save_path = fullfile(root,'image_stats','ImageNet','ILSVRC2012');

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

%     all_im_fns = [];
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

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

%% now loop over the three different training sets
for rr = 1:length(rot_list)
    
    
    % find all the folders in this directory
    image_folders = dir(fullfile(image_path, sprintf('train_rot_%d',rot_list(rr)),'n*'));
    
    folder2save = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)));
    if ~isfolder(folder2save)
        mkdir(folder2save)
    end
    
    nIms = zeros(length(image_folders),1);
    total_time = zeros(length(image_folders),6);
    
    %% loop over folders and images
    for ff=1:length(image_folders)

        fprintf('processing folder %d of %d\n',ff,length(image_folders));
        
        fn2save = fullfile(folder2save,sprintf('%s_allstats_highdensity.mat',image_folders(ff).name));  
      
        imlist = dir(fullfile(image_folders(ff).folder, image_folders(ff).name, '*.jpeg'));

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

            % subtract the background color (this is the color at a corner
            % pixel)
            image = image-uint8(permute(repmat([R_MEAN;G_MEAN;B_MEAN],1, size(image,1),size(image,2)), [2,3,1]));
%             assert(all(image(1,1,:)==0))    % make sure we have a zero here
            
            % make it grayscale
            if size(image,3)==3
                image=rgb2gray(image);
            end
%             if image(1,1)~=0    % make sure we have a zero here
%                 error('bad value in corner of image')
%             end
            
            image = im2double(image);
            
            orig_size = size(image);
    
            % no need to resize, but make sure it's teh size we expect
            assert(orig_size(1)==process_at_size && orig_size(2)==process_at_size)

            % pad it so we can apply the filters at the correct size
            pad_by = (size_after_pad - size(image))./2;        
            n2pad = [floor(pad_by'), ceil(pad_by')];        

            % Zero-pad the image for filtering
            image_padded = [repmat(zeros(size(image(:,1))), 1, n2pad(2,1)), image, repmat(zeros(size(image(:,end))), 1, n2pad(2,2))];
            image_padded = [repmat(zeros(size(image_padded(1,:))), n2pad(1,1), 1); image_padded; repmat(zeros(size(image_padded(end,:))), n2pad(1,2),1)];

            padded_size = size(image_padded);
            assert(all(padded_size==size_after_pad));

            %% Filtering

            % fft into frequency domain
            image_fft = fft2(image_padded);

            % Apply all my filters all at once
            filtered_freq_domain = image_fft.*filters_freq;

            % get back to the spatial domain
            out_full = ifft2(filtered_freq_domain);

            % un-pad the image (back to its down-sampled size)
            out = out_full(n2pad(1,1)+1:n2pad(1,1)+process_at_size, n2pad(2,1)+1:n2pad(2,1)+process_at_size,:);
            assert(size(out,1)==process_at_size && size(out,2)==process_at_size);

            mag = abs(out);
            phase = angle(out);

            %%  add all this info to my structure

            image_stats(ii).mean_phase = squeeze(mean(mean(phase,2),1));
            image_stats(ii).phase = phase;
            image_stats(ii).mean_mag = squeeze(mean(mean(mag,2),1));
            image_stats(ii).mag = mag;
            image_stats(ii).ori_list = ori_list;
            image_stats(ii).wavelength_list = wavelength_list;
            image_stats(ii).orig_size = orig_size;
%             image_stats(ii).scaled_size = scaled_size;
%             image_stats(ii).cropped_size = cropped_size;
            image_stats(ii).padded_size = padded_size;

        end
        c_end_all = clock;
        
        total_time(ff,:) = c_end_all - c_start_all;
        save(fn2save, 'image_stats','total_time');

    end
    
end