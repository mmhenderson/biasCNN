% make some filtered noise images with varying spatial frequency and orientation content

%% Set up parameters here
clear
close all hidden

% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

savedir = fullfile(root,'biasCNN','figures','ImageStats');

test_image_set = 'FiltImsAllSFCos_rand1';
train_image_path = fullfile(root,'biasCNN','images','ImageNet','ILSVRC2012');
% load this file that has the names of all synsets and the number of images
% in each...they're not all exactly the same size
load(fullfile(root,'biasCNN','code','make_filtered_images','nImsPerSetTraining.mat'));
nSynsets = 1000;
assert(numel(set_folders)==nSynsets)

nRots=3;
rot_list=[0,22,45];
nPerRot=5;
% orient_vals_deg = linspace(0,179,180);
% nOri = numel(orient_vals_deg);
% orient_kappa_deg=1000;
% nImsPerOri=1;
nImsTotal=nRots*nPerRot;

% this is the height and width of the final images
scale_by=1;
final_size_pix = 224*scale_by;



%% loop over sets, create the images.

% for each image - choose a random synset and a random image from that
% synset. 
rng(567687);
sets=[904,793,349,20,800];
ims=[938,223, 469,101,92];
rand_sets=sets;
rand_ims=ims;
% rand_sets = datasample(1:nSynsets,nPerRot,'replace',true);
% rand_ims = zeros(size(rand_sets));
% for ii=1:nPerRot
%     rand_ims(ii) = datasample(1:nImsPerSet(rand_sets(ii)),1);
% end
rand_ims = repmat(rand_ims,nRots,1);
rand_sets = repmat(rand_sets,nRots,1);
 
%% set up for figure
nsteps_h=nPerRot;
nsteps_v=nRots;
figure('Position',[0,0,nsteps_h*400,nsteps_v*400]);

hold all;axis off;axis equal

axspace_ratio_v = 4;

axsize_v = axspace_ratio_v/(axspace_ratio_v*nsteps_v+nsteps_v+1);
assert(axsize_v*nsteps_v+axsize_v/axspace_ratio_v*(nsteps_v+1) ==1)
axspace_v= axsize_v/axspace_ratio_v;
axpos_v = axspace_v:axsize_v+axspace_v:1-axspace_v;
axpos_v = fliplr(axpos_v);

axspace_ratio_h = 4;    % ratio of space to axes

axsize_h = axspace_ratio_h/(axspace_ratio_h*nsteps_h+nsteps_h+1);
assert(axsize_h*nsteps_h+axsize_h/axspace_ratio_h*(nsteps_h+1) ==1)
axspace_h= axsize_h/axspace_ratio_h;
axpos_h = axspace_h:axsize_h+axspace_h:1-axspace_h;

[axpos_x,axpos_y] = meshgrid(axpos_h, axpos_v);

axpos_full = [axpos_x(:), axpos_y(:), repmat(axsize_v,numel(axpos_x), 2)];

 %% load ims and make figure   
pp=0;
for ii=1:nPerRot
    for rr=1:nRots
    
    
        pp=pp+1;
            
%         if ispc
        imlist = dir(fullfile(train_image_path, sprintf('train_rot_%d_cos',rot_list(rr)),set_folders{rand_sets(rr,ii)}, '*.jpeg'));
%         else
%             imlist = dir(fullfile(train_image_path, sprintf('train_rot_%d',rot_list(rr)),set_folders{rand_sets(oo,ii)}, '*.JPEG'));
%         end
        imlist = {imlist.name};
        imfn = fullfile(train_image_path,sprintf('train_rot_%d_cos',rot_list(rr)),set_folders{rand_sets(rr,ii)}, imlist{rand_ims(rr,ii)});

        %% load and preprocess the image
        image = imread(imfn);

        ax = axes('Position',axpos_full(pp,:));hold all;

        imshow(image);

        if ii==1
            title(sprintf('%d deg CCW',rot_list(rr)))
        end
    end
   
end
        
%% end loop and save the image itself
    
set(gcf,'Color','w');
cdata = print('-RGBImage','-r300');
savepath = fullfile(savedir, 'TrainImages.pdf');
saveas(gcf,savepath,'pdf')
fprintf('saved image to %s\n',savepath);

