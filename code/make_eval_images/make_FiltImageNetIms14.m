% Make a set of filtered images (load random ImageNet images and 
% filter in the frequency domain). 
% Each image has broadband spatial frequency content, and orientation 
% content varying between 0-180 deg.
% Will create 4 unique random versions of this image set, each with 
% 180*48 = 8640 images.
% The codename of this image set is "FiltIms14AllSFCos", referenced
% elsewhere in this project code.

% MMH April 2020

%% Set up parameters here
clear
close all hidden

nSetsToMake = 4;
randseeds = [345349,793278,934899,578484];
sets2make_now = [1:4];

% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

% name under which i'll save this image set
image_set_root = 'FiltIms14AllSFCos';
% path where the ImageNet images are located
image_path = fullfile(root,'biasCNN','images','ImageNet','ILSVRC2012');
% load this file that has the names of all synsets and the number of images
% in each...they're not all exactly the same size
load(fullfile(root,'biasCNN','code','make_eval_images','nImsPerSetTraining.mat'));
nSynsets = 1000;
assert(numel(set_folders)==nSynsets)

% define the orientations to create
orient_vals_deg = linspace(0,179,180);
nOri = numel(orient_vals_deg);
orient_kappa_deg=1000;
nImsPerOri=48;
nImsTotal=nOri*nImsPerOri;

% this is the height and width of the final images
scale_by=1;
final_size_pix = 224*scale_by;

%% making a circular mask with cosine fading to background
cos_mask = zeros(final_size_pix);
values = final_size_pix./2*linspace(-1,1,final_size_pix);
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

mask_image=cos_mask;

%% background color
% also want to change the background color from 0 (black) to a mid gray color 
% (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
% will be subtracted when the images are centered during preproc.
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

channel_means = [R_MEAN, G_MEAN, B_MEAN];
channel_sd = [12,12,12];

%% define params for SF filtering

freq_sd_cpp = 0.005; % sigma for the gaussian fall-off
freq_min_cpp = 0.02;
freq_max_cpp = 0.25;
assert(freq_min_cpp>=freq_sd_cpp*4);
% Figure out how big image must be to accomodate lowest SF:
% code copied/adapted from gabor.m
BW = 1;
spatial_aspect_ratio=0.5;
wavelength = (1./freq_min_cpp);
sigmaX = wavelength/pi*sqrt(log(2)/2)*(2^BW+1)/(2^BW-1);    
sigmaY = sigmaX ./ spatial_aspect_ratio;    % bandwidth of spatial frequency - units of pixels in spectral domain

% SpatialKernel needs large (7 sigma radial) falloff of
% Gaussian in spatial domain for frequency domain and spatial
% domain computations to be equivalent within floating point
% round off error.
rx = ceil(7*sigmaX);
ry = ceil(7*sigmaY);

r = max(rx,ry);
padding_needed=r;

max_pix = final_size_pix;  % maximum size of any image dimension
% This is same for all images, so that we can make the filters ahead of time
size_after_pad = max_pix+padding_needed*2;
size_after_pad = size_after_pad + mod(size_after_pad,2);

%% Define spectral domain grid/sampling rate etc
% first, define sampling rate and frequency axis
samp_rate_pix = 1;   % samples per pixel, always one.
% get frequency axis
nframes = size_after_pad;
nyq = .5*samp_rate_pix;         
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpp = samp_rate_pix/nframes;  
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;
center_pix = find(abs(fax)==min(abs(fax)));

% next, we're listing the frequencies corresponding to each point in
% the FFT representation, in grid form
[ x_freq, y_freq ] = meshgrid(fax,fax);
x_freq = x_freq(:);y_freq = y_freq(:);
% converting these into a polar format, where angle is orientation and
% magnitude is spatial frequency
[ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
% adjust the angles so that they go from 0-pi in rads
ang_grid = mod(ang_grid,pi);
ang_grid = reshape(ang_grid,size_after_pad,size_after_pad);
mag_grid = reshape(mag_grid,size_after_pad,size_after_pad);

%% Define SF filter in spectral domain
tar_mag = normcdf(mag_grid, freq_min_cpp, freq_sd_cpp).*1-normcdf(mag_grid, freq_max_cpp,freq_sd_cpp);

%% loop over sets, create the images.

for ss = [sets2make_now]
    
    image_set = sprintf('%s_rand%d',image_set_root,ss);
    
    % for each image - choose a random synset and a random image from that
    % synset. 
    rng(randseeds(ss));
    rand_sets = datasample(1:nSynsets,nImsTotal,'replace',true);
    rand_ims = zeros(size(rand_sets));
    for ii=1:nImsTotal
        rand_ims(ii) = datasample(1:nImsPerSet(rand_sets(ii)),1);
    end
    rand_ims = reshape(rand_ims,nOri,nImsPerOri);
    rand_sets = reshape(rand_sets,nOri, nImsPerOri);
    
    % for each image - defne how much to randomly rotate it by, prior to
    % filtering.
    random_rots = reshape(datasample(0:179, nOri*nImsPerOri), nOri, nImsPerOri);
    
    % define where to save the newly created images
    image_save_path = fullfile(root,'biasCNN','images','gratings',image_set);
    if ~isdir(image_save_path)
        mkdir(image_save_path)
    end
    
    thisdir = fullfile(image_save_path,'AllIms');
    if ~isdir(thisdir)
        mkdir(thisdir)
    end
    
    for oo = 1:nOri
        
        %% Make a gaussian orientation filter in spectral domain
        tar_ang = reshape(circ_vmpdf(ang_grid*2, orient_vals_deg(oo)*pi/180*2, orient_kappa_deg*pi/180*2),size_after_pad,size_after_pad);
        tar_ang = tar_ang./max(tar_ang(:));
        % important - make sure that the filter has a constant value at the very center
        % of the frequency space. because this value has no angular
        % meaning.
        tar_ang(center_pix,center_pix) = 0;

        %% loop over images and filter each one
        for ii = 1:nImsPerOri

            if ispc
                imlist = dir(fullfile(image_path, 'train',set_folders{rand_sets(oo,ii)}, '*.jpeg'));
            else
                imlist = dir(fullfile(image_path, 'train',set_folders{rand_sets(oo,ii)}, '*.JPEG'));
            end
            imlist = {imlist.name};
            imfn = fullfile(image_path,'train',set_folders{rand_sets(oo,ii)}, imlist{rand_ims(oo,ii)});

            %% load and preprocess the image
            try
                image = imread(imfn);
            catch
                fprintf('Cannot load the specified image, loading a different one instead...\n')
                rset=datasample(1:nSynsets,1);
                rim=datasample(1:nImsPerSet(rset),1);
                if ispc
                    imlist = dir(fullfile(image_path, 'train',set_folders{rset}, '*.jpeg'));
                else
                    imlist = dir(fullfile(image_path, 'train',set_folders{rset}, '*.JPEG'));
                end
                imlist = {imlist.name};
                imfn = fullfile(image_path,'train',set_folders{rset}, imlist{rim});
                image = imread(imfn);
            end

            if size(image,3)==3
                image = rgb2gray(image);
            end
            image = double(image);

            % rotate by a random amount
            % this should break any dependencies between spatial frequency,
            % luminance contrast and orientation in the final images.
            image = imrotate(image, random_rots(oo,ii),'nearest','crop');

            % crop centrally to a square - this is the largest possible
            % square that would work for all rotations (45 degree is the
            % limiting rotation)
            orig_dims = size(image);
            [smaller_dim,smaller_dim_ind]= min(orig_dims);
            [larger_dim,larger_dim_ind]= max(orig_dims);
            smaller_dim_half = smaller_dim/2;
            crop_box_half = floor(sqrt(smaller_dim_half.^2/2))-1;         
            image_center = orig_dims/2;    
            crop_start = ceil(image_center-crop_box_half);
            crop_stop = crop_start + 2*crop_box_half;

            image_cropped = image(crop_start(1):crop_stop(1), crop_start(2):crop_stop(2));
            assert(size(image_cropped,1)==size(image_cropped,2));

            % rescale to its final size
            image_resized = imresize(image_cropped,[final_size_pix,final_size_pix]);
            assert(all(size(image_resized)==final_size_pix));

            image=image_resized;

            %% z-score all pixels and apply circular mask to remove edges

            % z-score, mean of 0, sd of 1
            image_scaled = reshape(zscore(image(:)), size(image));
            image_scaled=image_scaled.*mask_image;

            image=image_scaled;
            assert(image(1,1)==0)

            %% pad image with zeros so we can apply the filters at the correct size

            pad_by = (size_after_pad - size(image))./2;        
            n2pad = [floor(pad_by'), ceil(pad_by')];        

            % Zero-pad the image for filtering
            image_padded = [repmat(zeros(size(image(:,1))), 1, n2pad(2,1)), image, repmat(zeros(size(image(:,end))), 1, n2pad(2,2))];
            image_padded = [repmat(zeros(size(image_padded(1,:))), n2pad(1,1), 1); image_padded; repmat(zeros(size(image_padded(end,:))), n2pad(1,2),1)];

            padded_size = size(image_padded);
            assert(all(padded_size==size_after_pad));

            image=image_padded;
            
            %% FFT and filter the image
            [out] = fft2(image);
            image_fft = fftshift(out);  
            
            % apply the filters
            image_fft_filt = image_fft.*tar_mag.*tar_ang;

            mag = abs(image_fft_filt);
            % replace phase values with random numbers between -pi +pi
            fake_phase = (rand(size(image_fft_filt))-0.5)*2*pi;

            % create the full complex array again
            image_fft_filt_shuff = complex(mag.*cos(fake_phase), mag.*sin(fake_phase));

            % back to spatial domain
            image_filtered_full = real(ifft2(fftshift(image_fft_filt_shuff)));  

            % un-pad the image (back to its original size)
            image_filtered = image_filtered_full(n2pad(1,1)+1:n2pad(1,1)+final_size_pix, n2pad(2,1)+1:n2pad(2,1)+final_size_pix,:);
            assert(size(image_filtered,1)==final_size_pix && size(image_filtered,2)==final_size_pix);

            %% Convert to final format

            % go through each image channel and set mean and SD of the
            % luminance distribution to desired values.
            image_final = zeros(final_size_pix,final_size_pix,3);           
            for cc=1:3
                
                % mask, then z-score only the nonzero values (trying to
                % be as accurate as possible with mean and SD of the
                % final image)
                inds2use = mask_image>0;
                image_z = image_filtered.*mask_image;
                image_z(inds2use) = zscore(image_z(inds2use),1);

                % set mean and stdev
                image_z_scaled = image_z*channel_sd(cc) + channel_means(cc);

                % make sure all values are in possible range
                image_z_scaled = min(max(image_z_scaled, 0), 255);
                image_final(:,:,cc) = image_z_scaled;

            end

            % convert to integer format here
            image_final = uint8(image_final);

            % save the final image.
            fn2save = fullfile(thisdir,sprintf('FiltImage_ex%d_%ddeg.png',ii,orient_vals_deg(oo)));
            fprintf('saving to %s...\n', fn2save)
            imwrite(image_final, fn2save)
        end

        
    end
end
        


