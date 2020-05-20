%% plot the orientation content of the testing image set (filtered imagenet images)

clear
close all

% list the ground truth spat freq for gratings
meas_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
% true_sf2plot = [3];

image_set='FiltIms11Cos_SF_0.25_rand1';
% image_set = 'FiltIms9AllSFCos_rand1';
image_set_plot =image_set;
image_set_plot(image_set_plot=='_') = ' ';

curr_folder = pwd;
filesepinds=find(curr_folder==filesep);
root = curr_folder(1:filesepinds(end-1));
im_stat_path = fullfile(root, 'image_stats','gratings',image_set);

true_ori_vals_deg = linspace(0,179,180);
nOri_true = numel(true_ori_vals_deg);
nImsPerOri=48;
nIms = nOri_true*nImsPerOri;

%% get information about the filters that were used

[wavelength_list,~] = sort(1./meas_sflist,'ascend');
meas_ori_list = 5:5:180;
resize_factor = 1;
GaborBank = gabor(wavelength_list.*resize_factor,meas_ori_list);
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
nOri_filt = numel(meas_ori_list);
orilist_out = reshape(orilist_bank_fliptoCW, nSF_filt, nOri_filt);
sflist_out = reshape(sflist_bank, nSF_filt, nOri_filt);

meas_ori_list = orilist_out(1,:);
sf_axis = sflist_out(:,1);
nSF = length(sf_axis);
nOri_filt = length(meas_ori_list);

%% load the image stats

image_file = fullfile(im_stat_path, sprintf('AllIms_allstats_highdensity.mat'));
load(image_file)

nOri_true=180;
% re-organize the image stats into these arrays
ori_mag_list = zeros(nOri_true,nImsPerOri, nOri_filt);
sf_mag_list = zeros(nOri_true,nImsPerOri, nSF_filt);
orisf_mag_list = zeros(nOri_true,nImsPerOri, nSF_filt, nOri_filt);

for ii=1:nImsPerOri
    for oo=1:nOri_true

        % put this into a big array where the index is the true orientation
        % (different than ii because they don't go in order)
        mean_mag = reshape(image_stats(ii,oo).mean_mag, nSF_filt,nOri_filt);
        if image_stats(ii,oo).true_ori==0
            ori_mag_list(180,ii,:) = mean(mean_mag,1);
            sf_mag_list(180,ii,:) = mean(mean_mag,2);
            orisf_mag_list(180,ii,:,:) = mean_mag;
        else
            ori_mag_list(image_stats(ii,oo).true_ori,ii,:) = mean(mean_mag,1);
            sf_mag_list(image_stats(ii,oo).true_ori,ii,:) = mean(mean_mag,2);
            orisf_mag_list(image_stats(ii,oo).true_ori,ii,:,:) = mean_mag;
        end
    end
end

% nIms x nOriMeas
% zscore across orientation axis
ori_mag_list = zscore(ori_mag_list, [],3);

% nIms x nSFMeas
% zscore across SF axis
sf_mag_list = zscore(sf_mag_list, [], 3);

%% plot ori content versus spatial frequency

figure;hold all;
set(gcf,'Color','w')
vals = squeeze(mean(mean(orisf_mag_list(:,:,:,:),2),4));
vals = zscore(vals,[],1);
imagesc(vals);
xlabel('Filter SF');
ylabel('True orientation');
% title(sprintf('Filtered at %d deg',meas_ori_list(oo)));
xlim([0.5, nSF_filt+0.5]);
ylim([0.5, nOri_true+0.5]);
set(gca','XTick',[1:nSF_filt],'XTickLabel',sf_axis)
set(gca','YTick',[1:45:nOri_true],'YTickLabel',true_ori_vals_deg(1:45:nOri_true))
    
title(sprintf('%s\nSpatial frequency content of images at various orientations\n(averaged over filter orients)',image_set_plot));

%% plot ori content versus filter orientation
figure;hold all;
set(gcf,'Color','w')
vals = squeeze(mean(mean(orisf_mag_list(:,:,:,:),2),3));
vals = zscore(vals,[],1);
imagesc(vals);
xlabel('Filter orientation');
ylabel('True orientation');
% title(sprintf('Filtered at %d deg',meas_ori_list(oo)));
xlim([0.5, nOri_filt+0.5]);
ylim([0.5, nOri_true+0.5]);
set(gca','XTick',[1:6:nOri_filt],'XTickLabel',meas_ori_list(1:6:nOri_filt))
set(gca','YTick',[1:45:nOri_true],'YTickLabel',true_ori_vals_deg(1:45:nOri_true))
    
title(sprintf('%s\nMeasured orientation content of images at various orientations\n(averaged over filter SF)',image_set_plot));

%% plot ori content versus filter orientation - each SF separately

figure;hold all;
set(gcf,'Color','w')

for sf=1:length(sf_axis)
    
    subplot(2,3,sf);hold all;
    
    vals = squeeze(mean(orisf_mag_list(:,:,sf,:),2));
    vals = zscore(vals,[],1);
    imagesc(vals);
    xlabel('Filter orientation');
    ylabel('True orientation');
    axis square
    % title(sprintf('Filtered at %d deg',meas_ori_list(oo)));
    xlim([0.5, nOri_filt+0.5]);
    ylim([0.5, nOri_true+0.5]);
    set(gca','XTick',[1:6:nOri_filt],'XTickLabel',meas_ori_list(1:6:nOri_filt))
    set(gca','YTick',[1:45:nOri_true],'YTickLabel',true_ori_vals_deg(1:45:nOri_true))

    title(sprintf('filtered at %.2f',sf_axis(sf)));

end
suptitle(sprintf('%s\nMeas orientation content of images at various orientations',image_set_plot));

 %% plot orientation content - all FILTER sf togther


figure;set(gcf,'Color','w')
hold all;

title(sprintf('%s\nfiltered at all SF\norientation content',image_set_plot));
xlabel('degrees');
ylabel('average magnitude')
xlim([min(meas_ori_list),max(meas_ori_list)])
set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
ylim([-3, 4])
ii=0;
ori2do=[0:5:179]+1;
% ori2do=[80:5:100]+1;
cols = parula(numel(ori2do));
for oo = ori2do
    ii=ii+1;
% for oo = 1:1:nOri_true
    vals = squeeze(ori_mag_list(oo,:,:));
    plot(meas_ori_list,mean(vals,1),'Color',cols(ii,:),'LineStyle','-')
    line([true_ori_vals_deg(oo),true_ori_vals_deg(oo)],get(gca,'YLim'),'Color',cols(ii,:))
end
   

%% plot orientation content - separated by FILTER spatial frequency
figure;set(gcf,'Color','w')
hold all;
for sf = 1:length(sf_axis)

    subplot(2,3,sf);hold all
%     cols = parula(nOri_true);
    title(sprintf('filtered at %.2f',sf_axis(sf)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(meas_ori_list),max(meas_ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    ylim([-3, 4])
    ii=0;
    ori2do=[0:5:179]+1;
    % ori2do=[80:5:100]+1;
    cols = parula(numel(ori2do));
    for oo = ori2do
        ii=ii+1;
        vals = squeeze(orisf_mag_list(oo,:,sf,:));
        vals = zscore(vals,[],2);
        plot(meas_ori_list,mean(vals,1),'Color',cols(ii,:),'LineStyle','-')
        line([true_ori_vals_deg(oo),true_ori_vals_deg(oo)],get(gca,'YLim'),'Color',cols(ii,:))
    end
    vals = squeeze(orisf_mag_list(:,:,sf,:));
    vals=zscore(vals,[],3);
    plot(meas_ori_list,squeeze(mean(mean(vals,1),2)),'Color','k','LineStyle','-')
end
suptitle(sprintf('%s\norientation content',image_set_plot));
    
%% plot SF content - all filter orientations combined

figure;set(gcf,'Color','w')
hold all;
cols = parula(nOri_true);
title(sprintf('%s\nSF content, all images',image_set_plot));
xlabel('SF (cpp)');
ylabel('average magnitude')
xlim([min(sf_axis),max(sf_axis)])
ll=[];ii=0;
for oo = 1:22:nOri_true
    vals = squeeze(sf_mag_list(oo,:,:));
    plot(sf_axis,mean(vals,1),'Color',cols(oo,:),'LineStyle','-')
    ii=ii+1;
    ll{ii} = sprintf('%d deg',true_ori_vals_deg(oo));
end

legend(ll);


  
