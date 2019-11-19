%% plot the orientation content of a subset of the images used in the ImageNET database
% this includes all images in the DET 2014 training set
clear
close all


% list the ground truth spat freq for gratings
true_sflist = round(logspace(log10(0.02), log10(.4),6),3);



curr_folder = pwd;
filesepinds=find(curr_folder==filesep);
root = curr_folder(1:filesepinds(end-1));
im_path = fullfile(root, 'image_stats','gratings','grating_ims_11');

nIms = 180;
nOriMeas = 36;

% which of the true spatial frequencies do i want to make detailed plots
% of?
truesf2plot = [4];

trueori2plotimagesc = [1];

plot2d = 1;
plot1d = 1;

for ff=truesf2plot
     % find all the folders in this directory
    image_file = dir(fullfile(im_path, sprintf('*%.2f_Contrast_0.80*',10*true_sflist(ff))));
    
    fn2save = fullfile(image_file.folder, image_file.name);
     % find all the folders in this directory
%     folder2save = fullfile(curr_folder, 'image_stats', sprintf('SpatFreqGratings'));
%     fn2save = fullfile(folder2save, sprintf('%s_allstats_highdensity.mat',image_folder.name));
  
    load(fn2save);
     
    % how many SF did we measure at?
    nSFMeas = length(image_stats(1).wavelength_list);
    % how many orientations did we measure at?
    nOriMeas = length(image_stats(1).ori_list);    
    ori_mag_list = zeros(nIms, nOriMeas);
    sf_mag_list = zeros(nIms, nSFMeas);
    orisf_mag_list = zeros(nIms, nSFMeas, nOriMeas);
    
    for ii=1:length(image_stats)
        
        % put this into a big array where the index is the true orientation
        % (different than ii because they don't go in order)
        mean_mag = reshape(image_stats(ii).mean_mag, nSFMeas,nOriMeas);
        if image_stats(ii).true_ori==0
            ori_mag_list(180,:) = mean(mean_mag,1);
            sf_mag_list(180,:) = mean(mean_mag,2);
            orisf_mag_list(180,:,:) = mean_mag;
        else
            ori_mag_list(image_stats(ii).true_ori,:) = mean(mean_mag,1);
            sf_mag_list(image_stats(ii).true_ori,:) = mean(mean_mag,2);
            orisf_mag_list(image_stats(ii).true_ori,:,:) = mean_mag;
        end
        
        

    end
    
    ori_list = image_stats(1).ori_list;
    
    % note these are going in reverse order (descending) since the
    % wavelength is ascending order in GaborBank function
    sf_list = 1./image_stats(1).wavelength_list;
  
    % nIms x nOriMeas
    % zscore across orientation axis
    ori_mag_list = zscore(ori_mag_list, [],2);
    
    % nIms x nSFMeas
    % zscore across SF axis
    sf_mag_list = zscore(sf_mag_list, [], 2);

    % nIms x nSFMeas x nOriMean
    % zscore across orientation only
    orisf_mag_list = zscore(orisf_mag_list, [],3);
    
    %% make imagesc plot
    if plot2d
        
    for oo=1:length(trueori2plotimagesc)
        figure;set(gcf,'Color','w')
        hold all;
        imagesc(squeeze(orisf_mag_list(trueori2plotimagesc(oo),:,:))');
        xlabel('filter SF (cpp)')
        ylabel('filter orient (deg)')
        set(gca,'XTick',1:6,'XTickLabels',sf_list,'XDir','rev', 'YTick',1:36, 'YTickLabels',ori_list)
        title(sprintf('true SF=%.2f, true ori =%.2f',true_sflist(ff),ori_list(oo)));
    end
    
    end
     %% plot orientation content - all FILTER sf togther

     if plot1d
         
    figure;set(gcf,'Color','w')
    hold all;
    cols = parula(nIms);
    title(sprintf('TRUE sf=%.2f\nfiltered at all SF\norientation content',true_sflist(ff)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    ylim([-3, 4])
    for oo = 1:5:nIms
        plot(ori_list,ori_mag_list(oo,:),'Color',cols(oo,:),'LineStyle','-')
        line([180-oo,180-oo],get(gca,'YLim'),'Color',cols(oo,:))
    end
    plot(ori_list, mean(ori_mag_list,1), 'Color','k','LineWidth',3)

    %% plot orientation content - separated by FILTER spatial frequency

    for sf = 1:length(sf_list)

        figure;set(gcf,'Color','w')
        hold all;
        cols = parula(nIms);
        title(sprintf('TRUE sf=%.2f\nfiltered at %.2f\norientation content',true_sflist(ff), sf_list(sf)));
        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])
        set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
        ylim([-3, 4])
        for oo = 1:5:nIms
            plot(ori_list,squeeze(orisf_mag_list(oo,sf,:)),'Color',cols(oo,:),'LineStyle','-')
            line([180-oo,180-oo],get(gca,'YLim'),'Color',cols(oo,:))
        end
        plot(ori_list, mean(squeeze(orisf_mag_list(:,sf,:)),1), 'Color','k','LineWidth',3)
    end

    %% plot SF content - all orientations combined

    figure;set(gcf,'Color','w')
    hold all;
    cols = parula(length(true_sflist));
    title(sprintf('SF content, all images with TRUE SF=%.2f',true_sflist(ff)));
    xlabel('SF (cpp)');
    ylabel('average magnitude')
    ylim([-2, 2])
    xlim([0,0.50])

    plot(sf_list,mean(sf_mag_list,1),'Color',cols(ff,:),'LineStyle','-')
    line([true_sflist(ff),true_sflist(ff)],get(gca,'YLim'),'Color',cols(ff,:))
  
     end
end
  

  
