%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root, 'images','gratings','SpatFreqGratings');

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);

orilist = [0:179];
nSF = length(true_sflist);
nOri =180;



image_size = 224;

% start with a meshgrid
X=-0.5*image_size+.5:1:.5*image_size-.5; Y=-0.5*image_size+.5:1:.5*image_size-.5;
[x,y] = meshgrid(X,Y);

mask_file = fullfile(root,'code/image_proc_code/Smoothed_mask.png');

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

do_mask = 0;

noise_levels = [0.1];
nn=1;
contrast_levels = [0.50];
cc=1;

nEx = 1;

sf2do = [1:6];

discrim = zeros(nSF,nOri,nOri,nEx);
discrim_curve = zeros(nSF,nOri-1,nEx);


%% loop over spatial frequencies
for ff = sf2do
    
    % find all the folders in this directory
    image_folder = dir(fullfile(image_path, sprintf('*%.2f*',true_sflist(ff))));
    
    %% loop over images
   
    fprintf('processing spatial frequency %d\n',ff);
%     fprintf('processing folder %d of %d\n',ff,length(true_sflist));
        
    images = zeros(nOri,224,224,nEx);
    
    for oo = 1:nOri
        
        for ee = 1:nEx

             %% make the full field grating
            % range is [-1,1] to start
            sine = (sin(true_sflist(ff)*2*pi*(y.*sin(orilist(oo)*pi/180)+x.*cos(orilist(oo)*pi/180))));

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

            if do_mask
                % now multiply it by the donut (circle) to get gaussian envelope
                stim_masked = stim_scaled.*mask_image(:,:,1);

                % finally add a mid-gray background color.
                stim_masked_adj = uint8(stim_masked + mask_to_add(:,:,1));

                assert(all(squeeze(stim_masked_adj(1,1,1))==[R_MEAN]))
            else
                stim_masked_adj = stim_scaled;
            end
            images(oo,:,:,ee) = stim_masked_adj;
        end
%        fn2load = fullfile(image_folder(1).folder, image_folder(1).name,sprintf('Gaussian_phase0_ex1_%ddeg.png',orilist(oo)));
       
%        images(oo,:,:) = rgb2gray(imread(fn2load));
        
    end
    for ee = 1:nEx
        for or1 = 1:nOri
            for or2 = or1+1:nOri
                image1 = images(or1,:,:,ee);
                image2 = images(or2,:,:,ee);
                dist = sqrt(sum((image1(:)-image2(:)).^2));
                % get euclidean distance
                discrim(ff,or1,or2,ee) = dist;
                discrim(ff,or2,or1,ee) = dist;
                if abs(or1-or2)==1
                    fprintf('or1=%d, or2=%d, dist = %.2f\n',or1,or2,dist);
                    discrim_curve(ff,min([or1,or2]),ee) = dist;
                end

            end
        end
    end
    
    %% plot similarity matrix
%     figure;hold all;
%     imagesc(squeeze(discrim(ff,:,:)));
%     axis square 
%     xlim([0,180]);
%     ylim([0,180]);
%     set(gca,'XTick',[0:45:180]);
%     set(gca,'YTick',[0:45:180]);
%     title(sprintf('Pixelwise dissimilarity matrix\nSF = %.2f cpp',true_sflist(ff)));
%     set(gcf,'Color','w')
    
    %% plot 1-deg discriminability curve (off-diagonal)
%     figure;hold all;
%     plot(orilist(1:size(discrim_curve,2)),mean(discrim_curve(ff,:,:),3),'o');
%     set(gca,'XTick',[0:45:180]);
%     xlabel('orientation (deg)')
%     ylabel('euclidean distance');
%     set(gcf,'Color','w')
%     title(sprintf('Pixelwise discriminability of 1 degree steps\nSF = %.2f cpp',true_sflist(ff)));
    
    %%
    figure;hold all;
    for ee=1:nEx
        plot(orilist(1:size(discrim_curve,2)),discrim_curve(ff,:,ee),'o');
    end
    plot(orilist(1:size(discrim_curve,2)),mean(discrim_curve(ff,:,:),3),'-','Color','k');
    set(gca,'XTick',[0:45:180]);
    xlabel('orientation (deg)')
    ylabel('euclidean distance');
    set(gcf,'Color','w')
    title(sprintf('Pixelwise discriminability of 1 degree steps\nSF = %.2f cpp',true_sflist(ff)));
end