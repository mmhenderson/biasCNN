%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_set = 'CosGratings';

image_path = fullfile(root, 'images','gratings','CosGratings');
save_path = fullfile(root, 'image_stats','gratings','CosGratings');

if ~isfolder(save_path)
    mkdir(save_path)
end

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
process_at_size = 224;

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

%% specify the spatial frequencies and orientations to filter at

true_sf_list = true_sflist;
[wavelength_list,sorder] = sort(1./true_sf_list,'ascend');
true_sf_list = true_sf_list(sorder);

ori_list = 5:5:180;

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

%% make the filter bank (will be same for all images we look at)

fprintf('making filters...\n')
tic

GaborBank = gabor(wavelength_list.*resize_factor,ori_list);

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
    image_folder = dir(fullfile(image_path, sprintf('*%.2f*',true_sflist(ff))));
    
    fn2save = fullfile(save_path, sprintf('%s_allstats_highdensity.mat',image_folder.name));
  
    %% loop over images
   
    fprintf('processing folder %d of %d\n',ff,length(true_sflist));
        
    imlist = dir(fullfile(image_folder.folder, image_folder.name, '*.png'));

    fprintf('found %d images in folder\n',length(imlist));

    nIms(ff) = length(imlist);

    ori_done = [];
    %% loop over images and process
    clear image_stats
    
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

        % make this params struct to pass into my function
        params.R_MEAN = R_MEAN;
        params.G_MEAN = G_MEAN;
        params.B_MEAN = B_MEAN;

        params.process_at_size = process_at_size;
        params.size_after_pad = size_after_pad;
        params.filters_freq = filters_freq;
 
        params.ori_list = ori_list;
        params.wavelength_list = wavelength_list;
                
        %% do the processing in a separate function
        out = process_image(image, params);
        out.true_ori = true_ori;
        out.true_sf = true_sflist(ff);
        out.mag = [];
        out.phase = [];
        
        image_stats(ii) = out;
        
    end

    save(fn2save, 'image_stats');
    fprintf('saving to %s\n',fn2save);
end