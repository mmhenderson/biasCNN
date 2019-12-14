% make a smoothed circular mask to apply to imagenet images, trying to
% remove edge artifacts. this one puts noise in the mask, to see if there
% are any edge artifacts.
clear
close all
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));
save_path = fullfile(root, 'code','image_proc_code');
% this is the height and width of the final images
image_size = 224;

%% make a gaussian mask

Smooth_size = round(10); %size of fspecial smoothing kernel
Smooth_sd = round(5); %smoothing kernel sd
% Radius of mask outside edge. Needs to be smaller than half the image size,
% because it will expand as it is blurred below.
OuterMaskRadius = (image_size/2)-(Smooth_size*0.6); 

% start with a meshgrid
X=-0.5*image_size+.5:1:.5*image_size-.5; Y=-0.5*image_size+.5:1:.5*image_size-.5;
[x,y] = meshgrid(X,Y);

% make a masked circle
mask_orig = x.^2 + y.^2 <= (OuterMaskRadius)^2;

% blur the edges

mask = filter2(fspecial('gaussian', Smooth_size, Smooth_sd), mask_orig);

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

mask_to_add = cat(3,R_MEAN*ones(size(mask)),G_MEAN*ones(size(mask)),B_MEAN*ones(size(mask)));
mask_to_add = mask_to_add.*(1-mask);

% make noise then apply the mask
noise = randn(size(x))*255/5+255/2;
noise = filter2(fspecial('gaussian',Smooth_size,Smooth_sd), noise);
noise = repmat(noise,1,1,3);
noise = max(min(noise,255),0);
noise_masked = noise.*mask;

im_final = noise_masked+mask_to_add;
im_final = uint8(im_final);
% check and make sure all edge pixels have gone to zero

fn2save = fullfile(save_path,sprintf('Smoothed_masked_noise.png'));

imwrite(im_final, fn2save)

