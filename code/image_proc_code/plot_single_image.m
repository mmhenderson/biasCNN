%% get the orientation/SF content of an image in the ImageNET database

clear
close all

root = pwd;

if isempty(gcp('nocreate'))
    parpool(12);
end

sets = [175];
ims = [50];
resize_factor = 0.25;

syn_file = fullfile(root, 'ILSVRC2014_devkit/data/meta_det.mat');

load(syn_file)

%%
% im_size_deg = 7;
% ppd = 400/im_size_deg;
% cpd = 4;
% cpp = cpd/ppd;

freq_list = logspace(log10(0.01), log10(.1),3);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = 5:5:180;


for ss=sets

    %% read in names of images for this synset
    im_list_file = fullfile(root,'ILSVRC2014_devkit/data/det_lists', sprintf('train_pos_%d.txt',ss));
    fid = fopen(im_list_file,'r');
    im_list = [];
    line_str = fgetl(fid);
    while ischar(line_str)
        im_list = [im_list, {line_str}];
        line_str = fgetl(fid);
    end
    fclose(fid);

    for ii = ims

        figure;hold all;set(gcf,'Color','w');colormap(gray)

        im_file = fullfile(root, 'ILSVRC2014_DET_train',[im_list{ii} '.JPEG']);

        image = imread(im_file);
        image = im2double(rgb2gray(image));
        image = reshape(zscore(image(:)),size(image));
        
        %% plot the image
        subplot(3,2,1);hold all;axis off
        imagesc(image);
        title('original image');
        
       
        %% plot a downsampled version
        
        image_resize = imresize(image,resize_factor);
        image_resize = reshape(zscore(image_resize(:)),size(image_resize));
        
        subplot(3,2,2);hold all; axis off
        imagesc(image_resize);
        title(sprintf('resized by %.2f',resize_factor));
        
        %% filter the full-size image (slow!)
        tic
        
        mag = zeros(size(image,1),size(image,2),length(wavelength_list),length(ori_list));

        parfor oo=1:length(ori_list)

            gaborbank1 = gabor(wavelength_list,ori_list(oo));
            [this_mag,~] = imgaborfilt(image,gaborbank1);
            mag(:,:,:,oo) = this_mag;

        end
        toc

        subplot(3,2,3);hold all;

        ori_lines = squeeze(mean(mean(mag,2),1));
        plot(ori_list,ori_lines,'Color','k','LineStyle','--')
        ori_hist = squeeze(mean(mean(mean(mag,2),1),3));
        [~,peak] = max(ori_hist);
        plot(ori_list,ori_hist,'Color','k','LineWidth',2)
        line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
        title('orientation content (orig)');
        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])
        set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

        % line([exp_ori_peak,exp_ori_peak],get(gca,'YLim'),'Color','k')
        subplot(3,2,4);hold all;

        wave_hist = squeeze(mean(mean(mean(mag,2),1),4));
        [~,peak] = max(wave_hist);
        plot(freq_list, wave_hist,'Color','k')
        line([freq_list(peak),freq_list(peak)],get(gca,'YLim'),'Color','r')
        title('spatial frequency content (orig)');
        xlabel('frequency (cycles/pix)');
        ylabel('average magnitude')
        xlim([min(freq_list),max(freq_list)])

        suptitle(sprintf('%s, example %d', synsets(ss).name, ii))

        %% filter the downsampled image (faster!)
        tic
        mag = zeros(size(image_resize,1),size(image_resize,2),length(wavelength_list),length(ori_list));

        parfor oo=1:length(ori_list)

            gaborbank1 = gabor(wavelength_list.*resize_factor,ori_list(oo));
            [this_mag,~] = imgaborfilt(image_resize,gaborbank1);
            mag(:,:,:,oo) = this_mag;

        end
        toc
        
        subplot(3,2,5);hold all;

        ori_lines = squeeze(mean(mean(mag,2),1));
        plot(ori_list,ori_lines,'Color','k','LineStyle','--')
        ori_hist = squeeze(mean(mean(mean(mag,2),1),3));
        [~,peak] = max(ori_hist);
        plot(ori_list,ori_hist,'Color','k','LineWidth',2)
        line([ori_list(peak),ori_list(peak)],get(gca,'YLim'),'Color','r')
        title('orientation content (resized)');
        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])
        set(gca,'XTick',[0,45,90,135],'XTickLabels',[0,45,90,135]);

        % line([exp_ori_peak,exp_ori_peak],get(gca,'YLim'),'Color','k')
        subplot(3,2,6);hold all;

        wave_hist = squeeze(mean(mean(mean(mag,2),1),4));
        [~,peak] = max(wave_hist);
        plot(freq_list, wave_hist,'Color','k')
        line([freq_list(peak),freq_list(peak)],get(gca,'YLim'),'Color','r')
        title('spatial frequency content (resized)');
        xlabel('frequency (cycles/pix)');
        ylabel('average magnitude')
        xlim([min(freq_list),max(freq_list)])

        suptitle(sprintf('%s, example %d', synsets(ss).name, ii))

        
    end
end