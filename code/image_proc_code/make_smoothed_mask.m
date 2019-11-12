% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project
clear
close all
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));
save_path = fullfile(root, '/tensorflow/models/research/slim/datasets/');
save_path2 = fullfile(root, 'biasCNN/image_stat_analyses/');
% this is the height and width of the final images
image_size = 224;

%% make a mask with gaussian blurred edges (applied to all images)
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

% check and make sure all edge pixels have gone to zero
assert(all(mask(1,:)==0) && all(mask(:,end)==0) && all(mask(1,:)==0) && all(mask(end,:)==0))

fn2save = fullfile(save_path,sprintf('Smoothed_mask.png'));

imwrite(mask, fn2save)

fn2save = fullfile(save_path2,sprintf('Smoothed_mask.png'));

imwrite(mask, fn2save)
                  