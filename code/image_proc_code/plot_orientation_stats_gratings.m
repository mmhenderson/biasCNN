%% plot the orientation content of a subset of the images used in the ImageNET database
% this includes all images in the DET 2014 training set
clear
close all

true_sflist = [0.02, 0.04, 0.07, 0.12, 0.22, 0.40];

curr_folder = pwd;
filesepinds=find(curr_folder==filesep);
root = curr_folder(1:filesepinds(end));
im_path = fullfile(root, 'images','SpatFreqGratings');

nIms = 180;
nOriMeas = 36;

for ff=1:length(true_sflist)
     % find all the folders in this directory
    image_folder = dir(fullfile(im_path, sprintf('*%.2f*',true_sflist(ff))));
    
     % find all the folders in this directory
    folder2save = fullfile(curr_folder, 'image_stats', sprintf('SpatFreqGratings'));
    fn2save = fullfile(folder2save, sprintf('%s_allstats_highdensity.mat',image_folder.name));
  
    load(fn2save);
     
    nSFMeas = length(image_stats(1).wavelength_list);
    nOriMeas = length(image_stats(1).ori_list);    
    ori_mag_list = zeros(nIms, nOriMeas);
    sf_mag_list = zeros(nIms, nSFMeas);

    for ii=1:length(image_stats)
        
        % put this into a big array where the index is the true orientation
        % (different than ii because they don't go in order)
        mean_mag = reshape(image_stats(ii).mean_mag, nSFMeas,nOriMeas);
        if image_stats(ii).true_ori==0
            ori_mag_list(180,:) = mean(mean_mag,1);
            sf_mag_list(180,:) = mean(mean_mag,2);
        else
            ori_mag_list(image_stats(ii).true_ori,:) = mean(mean_mag,1);
            sf_mag_list(image_stats(ii).true_ori,:) = mean(mean_mag,2);
        end
        
        

    end
    
    ori_list = image_stats(1).ori_list;
    sf_list = 1./image_stats(1).wavelength_list;
  
    ori_mag_list = zscore(ori_mag_list, [],2);
    
    sf_mag_list_all(ff,:,:) = zscore(sf_mag_list, [],2);
    %% plot the mean stats, separated by true orientation

    figure;set(gcf,'Color','w')
    hold all;
    cols = parula(nIms);
    title(sprintf('sf=%.2f\norientation content',true_sflist(ff)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    ylim([-3, 4])
    for oo = 1:5:nIms
        plot(ori_list,ori_mag_list(oo,:),'Color',cols(oo,:),'LineStyle','-')
        line([180-oo,180-oo],get(gca,'YLim'),'Color',cols(oo,:))
    end
  
    
 
end
  
%% plot the mean stats, separated by true orientation

figure;set(gcf,'Color','w')
hold all;
cols = parula(length(true_sflist));
title('SF content, all images');
xlabel('SF (cpp)');
ylabel('average magnitude')
ylim([-2, 2])
xlim([0,0.50])
% set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
% ylim([-3, 4])
for ff = 1:length(true_sflist)
    plot(sf_list,mean(squeeze(sf_mag_list_all(ff,:,:)),1),'Color',cols(ff,:),'LineStyle','-')
    line([true_sflist(ff),true_sflist(ff)],get(gca,'YLim'),'Color',cols(ff,:))
end
  
