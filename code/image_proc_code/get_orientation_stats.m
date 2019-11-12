%% get the orientation content of a subset of the images from ImageNET database
% these are the images that were used to train VGG-16 in 2014 ILSVRC (DET training set)
clear
close all

if isempty(gcp('nocreate'))
    parpool(12);
end

root = pwd;

% find all the folders in this directory
image_folders = [dir(fullfile(root, 'ILSVRC2014_DET_train','n*')); dir(fullfile(root, 'ILSVRC2014_DET_train','ILSVRC*'))];
good_inds = find(~contains({image_folders.name},'.tar'));
image_folders = image_folders(good_inds);

% set an amount of downsampling, for speed of processing
resize_factor = 0.25;

syn_file = fullfile(root, 'ILSVRC2014_devkit/data/meta_det.mat');
load(syn_file)

%% specify the spatial frequencies and orientations to filter at

freq_list = logspace(log10(0.01), log10(.1),3);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;

all_im_fns = [];

%% loop over images
for ff=1:length(image_folders)
    
    if image_folders(ff).name(1)=='n'
        ind = find(ismember({synsets.WNID},image_folders(ff).name));
        if ~isempty(ind)
            fprintf('processing synset %d, %s\n',ind, synsets(ind).name);
            fn2save = fullfile(root, 'ImageStats',sprintf('synset%d_allstats.mat',ind));
        else
            fprintf('processing synset %s\n',image_folders(ff).name);
            fn2save = fullfile(root, 'ImageStats',sprintf('synset_%s_allstats.mat',image_folders(ff).name));   
        end
    else
        fprintf('processing images in %s\n',image_folders(ff).name);
        fn2save = fullfile(root, 'ImageStats',sprintf('%s_allstats.mat',image_folders(ff).name));   
    end
   
    image_folder = fullfile(image_folders(ff).folder, image_folders(ff).name);
    
    imlist = dir([image_folder, filesep, '*.JPEG']);
    
    fprintf('found %d images in folder\n',length(imlist));
    
    nIms(ff) = length(imlist);

    image_stats = [];
    
    for ii = 1:length(imlist)
        
        fprintf('processing image %d of %d\n',ii,length(imlist));
        
%         im_file = fullfile(root, 'ILSVRC2014_DET_train',[im_list{ii} '.JPEG']);
        im_file = fullfile(imlist(ii).folder, imlist(ii).name);
        try
            image = imread(im_file);
        catch err
            fprintf('image %d could not be loaded!\n',ii)
            continue
        end
        
        if size(image,3)==3
            image=rgb2gray(image);
        end
        image = im2double(image);
        image = reshape(zscore(image(:)),size(image));
        image = imresize(image,resize_factor);
        tic
        mag = zeros(size(image,1),size(image,2),length(wavelength_list),length(ori_list));

        parfor oo=1:length(ori_list)

            gaborbank1 = gabor(wavelength_list.*resize_factor,ori_list(oo));
            [this_mag,~] = imgaborfilt(image,gaborbank1);
            mag(:,:,:,oo) = this_mag;
        end
        toc
        image_stats(ii).mean_mag = squeeze(mean(mean(mag,2),1));
        image_stats(ii).ori_list = ori_list;
        image_stats(ii).wavelength_list = wavelength_list;
    end

    
    info = synsets(ff);
    save(fn2save, 'info','image_stats');
    
end