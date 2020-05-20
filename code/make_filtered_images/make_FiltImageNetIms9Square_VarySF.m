% make some filtered images with varying orientation content

%% Set up parameters here
clear
close all hidden

nSetsToMake = 4;
randseeds = [198729,102938,698357,304983];
sets2make_now = [1:4];


% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

image_set_root = 'FiltIms9Square';
image_path = fullfile(root,'biasCNN','images','ImageNet','ILSVRC2012');
% load this file that has the names of all synsets and the number of images
% in each...they're not all exactly the same size
load(fullfile(root,'biasCNN','code','make_filtered_images','nImsPerSetTraining.mat'));
nSynsets = 1000;
assert(numel(set_folders)==nSynsets)

orient_vals_deg = linspace(0,179,180);
nOri = numel(orient_vals_deg);
orient_kappa_deg=1000;
nImsPerOri=48;
nImsTotal=nOri*nImsPerOri;

% this is the height and width of the final images
scale_by=1;
final_size_pix = 224*scale_by;

% what spatial frequencies do you want? these will each be in a separate
% folder. Units are cycles per pixel.
freq_levels_cpp_orig = logspace(log10(0.02),log10(0.4),6);
% adjusting these so that they'll be directly comparable with an older
% version of the experiment (in which we had smaller 140x140 images)
freq_levels_cycles_per_image = freq_levels_cpp_orig*140;
% these are the actual cycles-per-pixel that we want, so that we end up
% with the same number of cycles per image as we had in the older version.
freq_levels_cpp = freq_levels_cycles_per_image/final_size_pix;

nSF = numel(freq_levels_cpp);
freq_sd_cpp = 0.005;

% %% making a circular mask with cosine fading to background
% cos_mask = zeros(final_size_pix);
% values = final_size_pix./2*linspace(-1,1,final_size_pix);
% [gridx,gridy] = meshgrid(values,values);
% r = sqrt(gridx.^2+gridy.^2);
% % creating three ring sections based on distance from center
% outer_range = 100*scale_by;
% inner_range = 50*scale_by;
% % inner values: set to 1
% cos_mask(r<inner_range) = 1;
% % middle values: create a smooth fade
% faded_inds = r>=inner_range & r<outer_range;
% cos_mask(faded_inds) = 0.5*cos(pi/(outer_range-inner_range).*(r(faded_inds)-inner_range)) + 0.5;
% % outer values: set to 0
% cos_mask(r>=outer_range) = 0;
% 
% mask_image=cos_mask;

% smooth the filter even more here
% Smooth_size = round(10); %size of fspecial smoothing kernel
% Smooth_sd = round(5); %smoothing kernel sd
% mask_image = filtfilt(fspecial('gaussian', Smooth_size, Smooth_sd), mask_image);
% mask_image = filter2(fspecial('gaussian', Smooth_size, Smooth_sd), mask_image);
% % filt in both fwd and reverse directions
% mask_image = filter2(fspecial('gaussian', Smooth_size, Smooth_sd), fliplr(flipud(mask_image)));

% figure;imagesc(mask_image-cos_mask);axis square;colormap('Gray');colorbar()

%%
% also want to change the background color from 0 (black) to a mid gray color 
% (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
% will be subtracted when the images are centered during preproc.
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

channel_means = [R_MEAN, G_MEAN, B_MEAN];
channel_sd = [12,12,12];

% mask_to_add = permute(repmat([R_MEAN,G_MEAN,B_MEAN], final_size_pix,1,final_size_pix),[1,3,2]);
% mask_to_add = mask_to_add.*(1-mask_image);

%% get ready for filtering
% first, define sampling rate and frequency axis
samp_rate_pix = 1;   % samples per pixel, always one.
% get frequency axis
nframes = final_size_pix;
nyq = .5*samp_rate_pix;         
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpp = samp_rate_pix/nframes;  
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;
center_pix = find(fax==min(abs(fax)));

% next, we're listing the frequencies corresponding to each point in
% the FFT representation, in grid form
[ x_freq, y_freq ] = meshgrid(fax,fax);
x_freq = x_freq(:);y_freq = y_freq(:);
% converting these into a polar format, where angle is orientation and
% magnitude is spatial frequency
[ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
% adjust the angles so that they go from 0-pi in rads
ang_grid = mod(ang_grid,pi);
ang_grid = reshape(ang_grid,final_size_pix,final_size_pix);
mag_grid = reshape(mag_grid,final_size_pix,final_size_pix);

% freq_sd_cpp=0.001;
% freq_mean_cpp = freq_step_cpp*2;
% 
% tar_mag = normcdf(mag_grid, freq_mean_cpp, freq_sd_cpp);

%% loop over sets, create the images.
maxvals = zeros(length(randseeds),nSF,nOri,nImsPerOri,3);
minvals = zeros(length(randseeds),nSF,nOri,nImsPerOri,3);


for ss = [sets2make_now]
  
    for sf=1
%     for sf = 1:nSF

        image_set = sprintf('%s_SF_%.2f_rand%d',image_set_root,freq_levels_cpp(sf),ss);

        % for each image - choose a random synset and a random image from that
        % synset. 
        rng(randseeds(ss)+sf);
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
        meanlum = zeros(nOri,nImsPerOri);
        
        %% make a gaussian spatial frequency filter
        freq_mean_cpp = freq_levels_cpp(sf);
        tar_mag = normpdf(mag_grid, freq_mean_cpp, freq_sd_cpp);

        for oo = 1:nOri

            %% Make a gaussian orientation filter
            tar_ang = reshape(circ_vmpdf(ang_grid*2, orient_vals_deg(oo)*pi/180*2, orient_kappa_deg*pi/180*2),final_size_pix,final_size_pix);
            % important - make sure that the filter has a 1 at the very center
            % of the frequency space. because this value has no angular
            % meaning.
            tar_ang(center_pix,center_pix) = 1;
%             tar_ang = tar_ang./100;
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

                %% z-score all pixels and apply circular mask

                % z-score, mean of 0, sd of 1
                image_scaled = reshape(zscore(image_resized(:)), size(image_resized));

                % mask background values to zero (get rid of edges before fFT)
%                 image_masked = image_scaled.*mask_image;
                image_masked=image_scaled;
                %% FFT and filter the image
                [out] = fft2(image_masked);
                image_fft = fftshift(out);  
%                 image_fft_filt = image_fft.*tar_ang;
                image_fft_filt = image_fft.*tar_ang.*tar_mag;

                mag = abs(image_fft_filt);
%                 phase = angle(image_fft_filt);
                % shuffle the phase randomly
%                 phase_shuff = reshape(phase(randperm(numel(phase))),size(phase));
                fake_phase = (rand(size(tar_ang))-0.5)*2*pi;
                % create the full complex array again
                image_fft_filt_shuff = complex(mag.*cos(fake_phase), mag.*sin(fake_phase));

                % bcck to spatial domain
                image_filtered = real(ifft2(fftshift(image_fft_filt_shuff)));  


    %             image_filtered = real(ifft2(fftshift(image_fft_filt)));            

                %% Convert to final format

                % Apply the mask again
                % values are double (negative and positive) here, mean at zero
%                 image_filt_masked = image_filtered.*mask_image;
                image_filt_masked=image_filtered;

                % go through each image channel and set mean and SD of the
                % luminance distribution to desired values.
                image_final = zeros(final_size_pix,final_size_pix,3);           
                for cc=1:3
                    % z-score first
                    image_z = reshape(zscore(image_filt_masked(:)),size(image_filt_masked));  
                    
                    % now mask
%                     image_z = image_z.*mask_image;
                    
                    % set mean and stdev
                    image_z_scaled = image_z*channel_sd(cc) + channel_means(cc);
                    % make sure all values are in correct range here
%                     minvals(ss,sf,oo,ii,cc) = min(image_z_scaled(:));
%                     maxvals(ss,sf,oo,ii,cc) = max(image_z_scaled(:));
%                     if min(image_z_scaled(:))<0 || max(image_z_scaled(:))>255
%                         fprintf('%d deg\n',oo);
% %                         pause
%                     end
                    image_z_scaled = min(max(image_z_scaled, 0), 255);
                    image_final(:,:,cc) = image_z_scaled;

                    % adjust slightly so background vals are
                    % exactly equal over all images.
%                     image_bg_adj = image_z_scaled + (channel_means(cc)-image_z_scaled(1,1))*ones(size(image_z_scaled));
%                     image_final(:,:,cc) = image_bg_adj;
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
end
        


