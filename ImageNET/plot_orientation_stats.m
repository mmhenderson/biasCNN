%% plot the orientation content of a subset of the images used in the ImageNET database
% this includes all images in the DET 2014 training set
clear
close all

root = pwd;

% for some reason the synsets listed in the structure don't match with the
% list of synsets whose images actually are present here
syn_file = fullfile(root, 'ILSVRC2014_devkit/data/meta_det.mat');
load(syn_file)
wnid_folders = dir(fullfile(root, 'ILSVRC2014_DET_train','n*'));
wnid_folder_list = {wnid_folders.name};
extra_synset_folders = setdiff(wnid_folder_list, {synsets.WNID});
empty_synset_names = setdiff({synsets.WNID},wnid_folder_list);
synsets = synsets(~ismember({synsets.WNID},empty_synset_names));

% first set of folders: images that correspond to a synset in WordNET
nSets1 = numel(synsets);
% also all these folders, which have synset names that are not included in the synsets file
nSets2 = numel(extra_synset_folders);

% also using all images from these extra folders - note these are not from
% ImageNET but are from Flickr.
extra_folders = dir(fullfile(root, 'ILSVRC2014_DET_train','ILSVRC*'));
good_inds = find(~contains({extra_folders.name},'.tar'));
extra_folders = extra_folders(good_inds);
nSets3 = numel(extra_folders);

nSets = nSets1 + nSets2 + nSets3;

nImsPerSet = zeros(nSets, 1);

ori_mag_list = [];
set_list = [];
% loop over images
for ss=1:nSets
    
    if ss<=nSets1     
        fprintf('loading synset %d, %s\n',ss, synsets(ss).name);
        fn2load = fullfile(root, 'ImageStats', sprintf('synset%d_allstats.mat',ss));
    elseif ss<=nSets1+nSets2
        fprintf('loading from %s\n',extra_synset_folders{ss-nSets1});
        fn2load = fullfile(root, 'ImageStats', sprintf('synset_%s_allstats.mat',extra_synset_folders{ss-nSets1}));
    else
        fprintf('loading from %s\n',extra_folders(ss-nSets1-nSets2).name);
        fn2load = fullfile(root, 'ImageStats', sprintf('%s_allstats.mat',extra_folders(ss-nSets1-nSets2).name));
    end
    
    load(fn2load)
    if isempty(image_stats)
        fprintf('NO IMAGES PRESENT\n')
        continue
    end
    empty =  find(cellfun(@isempty, {image_stats.mean_mag}));
    nIms = length(image_stats) - numel(empty);
    nImsPerSet(ss) = nIms;
    
    if ss==1
        ori_list = image_stats(1).ori_list;
        wavelength_list = image_stats(1).wavelength_list;
        freq_list = 1./wavelength_list;
        nSF = length(image_stats(1).wavelength_list);
        nOri = length(image_stats(1).ori_list);
        mean_mag = zeros(nSets,nSF,nOri);
        var_mag = zeros(nSets,nSF,nOri);
    end
    
    all_mag = [image_stats.mean_mag];
    all_mag = reshape(all_mag, nSF, nOri, nIms);
    
    mean_mag(ss,:,:) = mean(all_mag,3);   
    
    ims_by_ori = permute(squeeze(mean(all_mag,1)),[2,1]);
    ims_by_ori = zscore(ims_by_ori,[],2);
    
    % concatenate to a long list, nTotalIms x nOri
    ori_mag_list = [ori_mag_list; ims_by_ori];
    
    set_list=  [set_list; ss*ones(nIms,1)];
    
end

nImsTotal = sum(nImsPerSet);
assert(size(ori_mag_list,1)==nImsTotal);

clear legend_labs
for sf=1:nSF
    legend_labs{sf} = sprintf('%.2f cpp',freq_list(sf));
end
%% plot stats with error bars

figure;
hold all;
set(gcf,'Color','w')
meanvals = mean(ori_mag_list,1);

stdvals = std(ori_mag_list,[],1);
plot(ori_list,ori_mag_list(datasample(1:nImsTotal,100),:),'Color',[0.5,0.5,0.5],'LineStyle','-')
errorbar(ori_list, meanvals, stdvals,'Color','k','LineWidth',2);
title(sprintf('orientation content, %d images\n(mean+/-std)',nImsTotal));
xlabel('degrees');
ylabel('average magnitude')
xlim([min(ori_list),max(ori_list)])
set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

%% plot the mean stats, separated by spatial frequency

figure;set(gcf,'Color','w')
subplot(1,2,1); hold all;
cols = lines(3);

ori_lines = squeeze(mean(mean_mag,1));
ori_lines = zscore(ori_lines,[],2);
for sf = 1:nSF
    plot(ori_list,ori_lines(sf,:),'Color',cols(sf,:),'LineStyle','--')
end
% ori_hist = squeeze(mean(mean(mean_mag,1),2));
% [~,peak] = max(ori_hist);
% plot(ori_list,ori_hist,'Color','k','LineWidth',2)
% line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
title('orientation content');
xlabel('degrees');
ylabel('average magnitude')
xlim([min(ori_list),max(ori_list)])
set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

% line([exp_ori_peak,exp_ori_peak],get(gca,'YLim'),'Color','k')
subplot(1,2,2);hold all;

wave_hist = squeeze(mean(mean(mean_mag,1),3));
[~,peak] = max(wave_hist);
plot(freq_list, wave_hist,'Color','k')
% line([freq_list(peak),freq_list(peak)],get(gca,'YLim'),'Color','r')
title('spatial frequency content');
xlabel('frequency (cycles/pix)');
ylabel('average magnitude')
xlim([min(freq_list),max(freq_list)])
suptitle(sprintf('all images (%d)',nImsTotal))

%% make a figure to save

h=figure;set(gcf,'Color','w'); hold all;
cols = lines(3);

meanvals = mean(ori_mag_list,1);
stdvals = std(ori_mag_list,[],1);
bandedError_MMH(ori_list, meanvals, stdvals);
plot(ori_list, meanvals,'Color','k')
% errorbar(ori_list, meanvals, stdvals);
% plot(ori_list, ori_mag_list,'.')
% legend(legend_labs)
% ori_hist = squeeze(mean(mean(mean_mag,1),2));
% [~,peak] = max(ori_hist);
% plot(ori_list,ori_hist,'Color','k','LineWidth',2)
% line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
title('ImageNET statistics');
xlabel('Orientation (deg)');
ylabel('Magnitude (z-score)')
xlim([0,180])
set(gca,'XTick',[0,45,90,135,180],'XTickLabels',[0,45,90,135,180]);
set(gca,'YTick',[-1:1:3],'YTickLabels',[-1:1:3]);
savepath = '/usr/local/serenceslab/maggie/biasCNN/figures/Behavior/';
saveas(gcf, fullfile(savepath, 'ImageNET_orient.pdf'),'pdf');

%% plot some individual sets
%
figure;hold all;set(gcf,'Color','w')

sets2plot = datasample(1:200, 30,'replace',false);

for ss=1:length(sets2plot)
    
    subplot(5,6,ss);hold all
    ori_lines = squeeze(mean_mag(sets2plot(ss),:,:));
    ori_lines = zscore(ori_lines,[],2);
    for sf = 1:nSF
        plot(ori_list,ori_lines(sf,:),'Color',cols(sf,:),'LineStyle','-')
    end
    % ori_hist = squeeze(mean(mean(mean_mag,1),2));
    % [~,peak] = max(ori_hist);
    % plot(ori_list,ori_hist,'Color','k','LineWidth',2)
    % line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
    title(sprintf('%s (%d images)',synsets(sets2plot(ss)).name,nImsPerSet(sets2plot(ss))));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    if ss==length(sets2plot)
        legend(legend_labs)
    end
end

suptitle('orientation content, all images per set')
    
%% plot some individual sets -with lines for individual images.
%
figure;hold all;set(gcf,'Color','w')
sets2plot = datasample(1:200, 30,'replace',false);

for ss=1:length(sets2plot)
    
    subplot(5,6,ss);hold all
    
    ori_lines = ori_mag_list(set_list==ss,:);
    plot(ori_list,ori_lines(datasample(1:size(ori_lines,1),30),:), 'Color',[0.5, 0.5, 0.5],'LineStyle','-');
    plot(ori_list,mean(ori_lines,1),'Color','k','LineWidth',4);
    % ori_hist = squeeze(mean(mean(mean_mag,1),2));
    % [~,peak] = max(ori_hist);
    % plot(ori_list,ori_hist,'Color','k','LineWidth',2)
    % line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
    title(sprintf('%s (%d images)',synsets(sets2plot(ss)).name,nImsPerSet(sets2plot(ss))));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
%     if ss==length(sets2plot)
%         legend(legend_labs)
%     end
end

suptitle('orientation content, all images per set')
    

