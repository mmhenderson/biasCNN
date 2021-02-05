%% Plot the orientation content of each modified version of ImageNet dataset
% Verify that rotations change the prior in the expected way (peaks at
% cardinals or cardinals + rotation).
% This script load the files saved by
% analyze_orientation_stats_imagenet_rots.m

% MMH March 2020
%%
clear
close all

% rot_list = [45];
rot_list = [0,22, 45];
% Switch these into my usual coordinate system: start at 0 degrees, moving
% in the clockwise direction. This is how gratings were drawn so it will
% match those images. The images were actually rotated in a
% counter-clockwie direction, which corresponds to a negative rotation in
% this coord system.
new_card_axes = mod([0-rot_list', 90-rot_list'],180);

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

fig_folder = fullfile(root, 'figures','ImageStats');
stat_path = fullfile(root,'image_stats','ImageNet','ILSVRC2012');

plotFisherPredAll = 1;
plotOriDistAll=1;
plotOriDistEachSF=1;
saveFigs=1;

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

legend_labs=[];
for sf=1:nSF_filt
    legend_labs{sf} = sprintf('%.2f cpp',freq_list(sf));
end

%% load the images
for rr=1:length(rot_list)

    fn2load = fullfile(stat_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)), 'AllIms_Ori_MeanSD.mat');
    fprintf('loading %s\n',fn2load);
    load(fn2load);

    fn2load = fullfile(stat_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)), 'AllIms_OriSF_MeanSD.mat');
    fprintf('loading %s\n',fn2load);
    load(fn2load);

    
    %% plot stats with error bars
    if plotOriDistAll
        figure;
        set(gcf,'DefaultLegendAutoUpdate','off');
        hold all;
        set(gcf,'Color','w')
     
%         bandedError_MMH(ori_axis,meanvals, stdvals, [0,0,1],0.2);
        plot(ori_axis,meanvals,'Color','k')
        title(sprintf('ImageNet Orientation Content\n%d deg CCW',rot_list(rr)));
        xlabel('degrees');
        ylabel('Probability')
        xlim([0,180])
        ylim([0.01, 0.05])
        set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135],'YTick',[0, 0.02, 0.04]);

        line([new_card_axes(rr,1), new_card_axes(rr,1)], get(gca,'YLim'),'Color','k');
        line([new_card_axes(rr,2), new_card_axes(rr,2)], get(gca,'YLim'),'Color','k');

        if saveFigs
            
            handle = gcf;
            prepFigForExport(handle,1);
            set(handle.Children(1),'FontSize',30)
            set(handle,'Position',[0,0,1200,1200])
            saveas(handle,fullfile(fig_folder,sprintf('OriContentAll_rot%d.pdf',rot_list(rr))),'pdf');
        end
    end
    
    
    %% plot predicted fisher information
    if plotFisherPredAll
        figure;
        set(gcf,'DefaultLegendAutoUpdate','off');
        hold all;
        set(gcf,'Color','w')
        
        % fisher info should be prior^2 (e.g. Wei and Stocker 2015 Nat Neuro)
        prior = meanvals - min(meanvals);
        fisher = prior.^2;
                
%         bandedError_MMH(ori_axis,meanvals, stdvals, [0,0,1],0.2);
        plot(ori_axis,fisher,'Color','k')
        title(sprintf('Predicted Fisher information (Prior^2)\n%d deg CCW',rot_list(rr)));
        xlabel('degrees');
        ylabel('Fisher info')
        xlim([0,180])
%         ylim([0, 0.002])
        set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135],'YTick',[0, 0.001, 0.002]);

        line([new_card_axes(rr,1), new_card_axes(rr,1)], get(gca,'YLim'),'Color','k');
        line([new_card_axes(rr,2), new_card_axes(rr,2)], get(gca,'YLim'),'Color','k');

        if saveFigs
            
            handle = gcf;
            prepFigForExport(handle,1);
            set(handle.Children(1),'FontSize',30)
            set(handle,'Position',[0,0,1200,1200])
            saveas(handle,fullfile(fig_folder,sprintf('FisherInfoPred_rot%d.pdf',rot_list(rr))),'pdf');
        end
    end
    
    %% plot the mean stats, separated by spatial frequency
    if plotOriDistEachSF
        figure;set(gcf,'Color','w')
        set(gcf,'DefaultLegendAutoUpdate','off');
        hold all;
        cols = parula(nSF_filt);

        for sf = 1:nSF_filt
            subplot(2,2,sf);hold all;
                        
            mvals = meanvals_bysf(:,sf)';
            svals = stdvals_bysf(:,sf)';
%             bandedError_MMH(ori_axis,mvals, svals, [0,0,1],0.2);
            plot(ori_axis,meanvals,'Color','k')

            title(legend_labs{sf});
            xlabel('degrees');
            ylabel('magnitude (z-score)')
            xlim([0,180])
            ylim([0, 0.04])
%             ylim([-2.5, 2.5])
            set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135],'YTick',[0, 0.02, 0.04]);

            line([new_card_axes(rr,1), new_card_axes(rr,1)], get(gca,'YLim'),'Color','k');
            line([new_card_axes(rr,2), new_card_axes(rr,2)], get(gca,'YLim'),'Color','k');

        end

        suptitle((sprintf('ImageNet Orientation Content\n%d deg CCW',rot_list(rr))));      
        if saveFigs
           
            handle = gcf;
            prepFigForExport(handle,1);
            set(handle,'Position',[0,0,1200,1200])
            saveas(handle,fullfile(fig_folder,sprintf('OriContentEachSF_rot%d.pdf',rot_list(rr))),'pdf');
        end
    end
    
end