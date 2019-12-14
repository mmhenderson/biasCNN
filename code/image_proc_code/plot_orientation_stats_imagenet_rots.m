%% plot the orientation content of a subset of the images used in the ImageNET database
% this includes all images in the DET 2014 training set
clear
close all

% rot_list = [45];
rot_list = [0,22,45];

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root,'images','ImageNet','ILSVRC2012');
save_path = fullfile(root,'image_stats','ImageNet','ILSVRC2012');

sets2do = [170:180];
nSets = length(sets2do);
%% find the names of all image sets (these are identical across rotations)
% want to make sure we grab the same ones in the same order across all
% rotations, so make the list now.
set_folders = dir(fullfile(image_path, sprintf('train_rot_0'),'n*'));
set_folders = {set_folders.name}; 

for rr=1:length(rot_list)

    ori_mag_list = [];
    set_list = [];

    nImsPerSet = zeros(nSets, 1);
    for ss=sets2do
        
        set_file = dir(fullfile(save_path,sprintf('ImageStats_train_rot_%d',rot_list(rr)),sprintf('%s*.mat',set_folders{ss})));

        set_file = set_file(1);
        
        fn2load = fullfile(set_file.folder, set_file.name);
        load(fn2load)
        if isempty(image_stats)
            fprintf('NO IMAGES PRESENT\n')
            continue
        end
        empty =  find(cellfun(@isempty, {image_stats.mean_mag}));
        nIms = length(image_stats) - numel(empty);
        nImsPerSet(ss) = nIms;
        
        if ss==sets2do(1)
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
     title(sprintf('Orientation content: ims rotated %d deg\n%d images (mean+/-std)', rot_list(rr), nImsTotal));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

    line([90+rot_list(rr), 90+rot_list(rr)], get(gca,'YLim'),'Color','k');
    line([0+rot_list(rr), 0+rot_list(rr)], get(gca,'YLim'),'Color','k');
    %% one example image
    % this plot should be shifted when the image is rotated
    figure;
    hold all;
    set(gcf,'Color','w')

    plot(ori_list,ori_mag_list(1,:),'Color',[0.5,0.5,0.5],'LineStyle','-')

    title(sprintf('Orientation content: ims rotated %d deg\none example image', rot_list(rr)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

    line([90+rot_list(rr), 90+rot_list(rr)], get(gca,'YLim'),'Color','k');
    line([0+rot_list(rr), 0+rot_list(rr)], get(gca,'YLim'),'Color','k');

    %% one example image - each SF separately
    % this plot should be shifted when the image is rotated
    figure;
    hold all;
    set(gcf,'Color','w')
    cols = parula(nSF);
    for sf = 1:nSF
        vals = squeeze(all_mag(sf,:,1));       
        vals = zscore(vals,[],2);
        plot(ori_list,vals,'Color',cols(sf,:),'LineStyle','-')

    end
    title(sprintf('Orientation content: ims rotated %d deg\none example image', rot_list(rr)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    legend(legend_labs)
    line([90+rot_list(rr), 90+rot_list(rr)], get(gca,'YLim'),'Color','k');
    line([0+rot_list(rr), 0+rot_list(rr)], get(gca,'YLim'),'Color','k');

    %% plot the mean stats, separated by spatial frequency

    figure;set(gcf,'Color','w')
    hold all;
    cols = parula(nSF);

    ori_lines = squeeze(mean(mean_mag,1));
    ori_lines = zscore(ori_lines,[],2);
    for sf = 1:nSF
        plot(ori_list,ori_lines(sf,:),'Color',cols(sf,:),'LineStyle','-')
    end
    
    title(sprintf('Orientation content: ims rotated %d deg', rot_list(rr)));
    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])
    set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);
    legend(legend_labs)
    line([90+rot_list(rr), 90+rot_list(rr)], get(gca,'YLim'),'Color','k');
    line([0+rot_list(rr), 0+rot_list(rr)], get(gca,'YLim'),'Color','k');

    %% plot SF distribution
    figure ;hold all;

    wave_hist = squeeze(mean(mean(mean_mag,1),3));
    [~,peak] = max(wave_hist);
    plot(freq_list, wave_hist,'Color','k')
    % line([freq_list(peak),freq_list(peak)],get(gca,'YLim'),'Color','r')
    title(sprintf('Spatial frequency content: ims rotated %d deg', rot_list(rr)));
    xlabel('frequency (cycles/pix)');
    ylabel('average magnitude')
    xlim([min(freq_list),max(freq_list)])
    suptitle(sprintf('all images (%d)',nImsTotal))

end
