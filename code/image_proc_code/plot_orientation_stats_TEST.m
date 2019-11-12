%% get the orientation content of a subset of the images from ImageNET database
% making sure that our image set rotations affected the prior in the
% expected way (peak at cardinals, or cardinals+rotation)
%%

clear
close all

% if isempty(gcp('nocreate'))
%     parpool(8);
% end

% rot_list = [45];
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

%% specify the spatial frequencies and orientations to filter at

freq_list = logspace(log10(0.02), log10(.2),4);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;

%     all_im_fns = [];
R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

Ims2Use = 1:10;
% nIms2Do = 2;
mag_by_ori = zeros(length(rot_list),length(Ims2Use),length(ori_list));

%% now loop over the three different training sets
for rr = 1:length(rot_list)
    

    image_folders = dir(fullfile(image_path, sprintf('train_rot_%d',rot_list(rr)),'n*'));
    
    folder2save = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)));
    if ~isfolder(folder2save)
        mkdir(folder2save)
    end

    %% loop over folders and images

    ff=1;

    fn2save = fullfile(folder2save,sprintf('%s_allstats_highdensity_TEST.mat',image_folders(ff).name));  
    load(fn2save);
    fprintf('loading from %s\n',fn2save);

    mean_mag = reshape([image_stats.mean_mag],length(freq_list),length(ori_list),numel(image_stats));

    mag_by_ori(rr,:,:) = permute(squeeze(mean(mean_mag(:,:,Ims2Use),1)),[2,1]);

end

%%
for ii=Ims2Use
    
figure;hold all;
plot(ori_list,squeeze(mag_by_ori(:,ii,:)));
% legend(rot_list);

end
