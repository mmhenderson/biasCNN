%% analyze the mean and standard dev of orientation content from ImageNet 
%images, using output of get_orientation_stats_imagenet_rots.m

clear
close all

saveFigs=1;

% rot_list = [45];
rot_list = [0,22, 45];

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root,'images','ImageNet','ILSVRC2012');
save_path = fullfile(root,'image_stats','ImageNet','ILSVRC2012');

% sets2do = [1,2];
sets2do = [1:1000];
nSets = length(sets2do);

%% find the names of all image sets (these are identical across rotations)
% want to make sure we grab the same ones in the same order across all
% rotations, so make the list now.
set_folders = dir(fullfile(image_path, sprintf('train_rot_0'),'n*'));
set_folders = {set_folders.name}; 

%% get information about the filters that were used
freq_list = logspace(log10(0.02), log10(.2),4);
[wavelength_list,~] = sort(1./freq_list,'ascend');
ori_list = 5:5:180;
resize_factor = 1;
GaborBank = gabor(wavelength_list.*resize_factor,ori_list);
% based on the GaborBank - list the orientations and SF of the filters
% note that this might be sorted differently than what we put in, this
% order is the correct one.
orilist_bank = [GaborBank.Orientation];
% this is the orientation of the filter, switched into the same coordinate
% system as the grating images I drew (clockwise from 0, where 0=vertical).
orilist_bank_fliptoCW = 180 - orilist_bank;
sflist_bank = 1./[GaborBank.Wavelength];
wavelist_bank = [GaborBank.Wavelength];

% reshaped list of the ori and SF that correspond to each filter
nSF_filt = numel(wavelength_list);
nOri_filt = numel(ori_list);
orilist_out = reshape(orilist_bank_fliptoCW, nSF_filt, nOri_filt);
sflist_out = reshape(sflist_bank, nSF_filt, nOri_filt);

ori_axis = orilist_out(1,:);
sf_axis = sflist_out(:,1);
nSF_filt = length(sf_axis);
nOri_filt = length(ori_axis);

%% load the images
for rr=1:length(rot_list)

    ori_mag_list = [];
    ori_mag_by_sf_list = [];
    
    nImsPerSet = zeros(nSets, 1);
    for ss=sets2do
        
        set_file = dir(fullfile(save_path,sprintf('ImageStats_train_rot_%d',rot_list(rr)),sprintf('%s*.mat',set_folders{ss})));

        set_file = set_file(1);
        
        fn2load = fullfile(set_file.folder, set_file.name);
        fprintf('loading %s\n',fn2load);
        load(fn2load)
        if isempty(image_stats)
            fprintf('NO IMAGES PRESENT\n')
            continue
        end
        empty =  find(cellfun(@isempty, {image_stats.mean_mag}));
        nIms = length(image_stats) - numel(empty);
        nImsPerSet(ss) = nIms;
        
        if ss==sets2do(1)
            mean_mag = zeros(nSets,nSF_filt,nOri_filt);
            var_mag = zeros(nSets,nSF_filt,nOri_filt);
        end

        all_mag = [image_stats.mean_mag];
        all_mag = reshape(all_mag, nSF_filt, nOri_filt, nIms);

%         mean_mag(ss,:,:) = mean(all_mag,3);   

        ims_by_ori = permute(squeeze(mean(all_mag,1)),[2,1]);
        ims_by_ori = zscore(ims_by_ori,[],2);
        
        % concatenate to a long list, nTotalIms x nOri
        ori_mag_list = [ori_mag_list; ims_by_ori];
        
        ims_by_ori_sf = permute(all_mag, [3,2,1]);
        ims_by_ori_sf = zscore(ims_by_ori_sf, [],2);
        
        ori_mag_by_sf_list = [ori_mag_by_sf_list; ims_by_ori_sf];
           
    end

    
    nImsTotal = sum(nImsPerSet);
    assert(size(ori_mag_list,1)==nImsTotal);

    if all(sets2do==1:1000)
        meanvals = mean(ori_mag_list,1);
        stdvals = std(ori_mag_list,[],1);
        ori_save_name = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)), 'AllIms_Ori_MeanSD.mat');
        fprintf('saving to %s\n',ori_save_name);
        save(ori_save_name, 'ori_axis','meanvals','stdvals');
        
        meanvals_bysf = zeros(nOri_filt,nSF_filt);
        stdvals_bysf = zeros(nOri_filt, nSF_filt);
        for sf = 1:nSF_filt
            vals = ori_mag_by_sf_list(:,:,sf);            
            meanvals_bysf(:,sf) = mean(vals,1);
            stdvals_bysf(:,sf) = std(vals,[],1);
        end
        ori_sf_save_name = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)), 'AllIms_OriSF_MeanSD.mat');
        fprintf('saving to %s\n',ori_sf_save_name);
        save(ori_sf_save_name, 'ori_axis','meanvals_bysf','stdvals_bysf');
    end   
    
end