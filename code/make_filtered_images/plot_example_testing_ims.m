% make some filtered noise images with varying spatial frequency and orientation content

%% Set up parameters here
clear
close all hidden

% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

savedir = fullfile(root,'biasCNN','figures','ImageStats');

test_image_set = 'FiltIms2AllSFCos_rand2';
image_path = fullfile(root,'biasCNN','images','gratings',test_image_set,'AllIms');

nOri=5;
ori_list=[0,36,72,108,144];
nPerOri=3;
% orient_vals_deg = linspace(0,179,180);
% nOri = numel(orient_vals_deg);
% orient_kappa_deg=1000;
% nImsPerOri=1;
nImsTotal=nOri*nPerOri;

 
%% set up for figure
nsteps_h=nOri;
nsteps_v=nPerOri;
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
for oo=1:nOri
   for ii=1:nPerOri
   
    
    
        pp=pp+1;
        
        imfn = fullfile(image_path,sprintf('FiltImage_ex%d_%ddeg.png',ii,ori_list(oo)));
                 
        %% load and preprocess the image
        image = imread(imfn);

        ax = axes('Position',axpos_full(pp,:));hold all;

        imshow(image);

        if ii==1
            title(sprintf('%d deg CCW',ori_list(oo)))
        end
    end
   
end
        
%% end loop and save the image itself
    
set(gcf,'Color','w');
cdata = print('-RGBImage','-r300');
savepath = fullfile(savedir, 'TestImages.pdf');
saveas(gcf,savepath,'pdf')
fprintf('saved image to %s\n',savepath);

