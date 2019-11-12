%% get the orientation content of a subset of the images from ImageNET database
% making sure that our image set rotations affected the prior in the
% expected way (peak at cardinals, or cardinals+rotation)
%%

clear
close all

rot_list = [0,22,45];

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

% which sets should we look at now? don't do all of them yet, because it'll
% take too long.
sets2do = [1:100];
%% specify the spatial frequencies and orientations to filter at

freq_list = logspace(log10(0.02), log10(.2),4);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;

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

%% find the names of all image sets (these are identical across rotations)
% want to make sure we grab the same ones in the same order across all
% rotations, so make the list now.
set_folders = dir(fullfile(image_path, sprintf('train_rot_0'),'n*'));
set_folders = {set_folders.name}; 

%% loop over image sets first
for ff=[sets2do]

    fprintf('processing image set %d of %d\n',ff,length(set_folders));

    %% now loop over the three different training sets
    for rr = 1:length(rot_list)

        image_dir_this_rot = fullfile(image_path, sprintf('train_rot_%d',rot_list(rr)));

        folder2save = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)));
        if ~isfolder(folder2save)
            mkdir(folder2save)
        end
        
        fn2save = fullfile(folder2save,sprintf('%s_allstats_highdensity.mat',set_folders{ff}));  
      
        imlist = dir(fullfile(image_dir_this_rot, set_folders{ff}, '*.jpeg'));

        % make this params struct to pass into my function
        params.R_MEAN = R_MEAN;
        params.G_MEAN = G_MEAN;
        params.B_MEAN = B_MEAN;

        params.process_at_size = process_at_size;
        params.size_after_pad = size_after_pad;
        params.filters_freq = filters_freq;
 
        params.ori_list = ori_list;
        params.wavelength_list = wavelength_list;
        
        %% loop over images and process
        clear image_stats
        parfor ii = 1:length(imlist)
%         parfor ii =1:10
            
            im_file = fullfile(imlist(ii).folder, imlist(ii).name);
            fprintf('loading image %d of %d\n',ii,length(imlist));
            fprintf('    from %s\n',im_file);
            
            try
                image = imread(im_file);
            catch err
                fprintf('image %d could not be loaded!\n',ii)
                continue
            end
          
            %% do the processing in a separate function
            image_stats(ii) = process_image(image, params);
            
        end

        save(fn2save, 'image_stats');
        fprintf('saving to %s\n',fn2save);
    end
    
end


