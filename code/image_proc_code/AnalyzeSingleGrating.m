%% do orientation/SF content analysis of a grating image
% make a plot showing the outcome of filtering for a range of orientations
% and SFs
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root, 'images','gratings','SpatFreqGratings');
save_path = fullfile(root, 'image_stats','gratings','SpatFreqGratings');

if ~isfolder(save_path)
    mkdir(save_path)
end

plotFiltOut = 1;
plotSF = 1;
plotOri = 1;

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);

% which SF and orientation is the grating that I'm analyzing?
truesf2do = [1:6];
trueori2do = [20];

% which SF and orientation do i want to plot? 
sf2filter = [1:6];
ori2filter = [22.5:22.5:180];

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
process_at_size = 224;

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

%% specify the spatial frequencies and orientations to filter at

% freq_list = logspace(log10(0.02), log10(.2),4);
freq_list = true_sflist(sf2filter);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);

ori_list = ori2filter;

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

%% make the filter bank (will be same for all images we look at)

fprintf('making filters...\n')
tic

GaborBank = gabor(wavelength_list.*resize_factor,ori_list);
freq_inds = repelem(1:length(freq_list),numel(ori_list));
sizeLargestKernel = size(GaborBank(end).SpatialKernel);
% Gabor always returns odd length kernels
padding_needed = (sizeLargestKernel-1)/2;

max_pix = process_at_size;  % maximum size of any image dimension
% FIX this so that we can make the filters ahead of time
size_after_pad = max_pix*resize_factor+padding_needed*2;
size_after_pad = size_after_pad + mod(size_after_pad,2);
% making a matrix [nPix x nPix x nFilters]
filters_freq = zeros([size_after_pad,length(GaborBank)]);

for p = 1:length(GaborBank)

    H = makeFrequencyDomainTransferFunction_MMH(GaborBank(p),size_after_pad);
    filters_freq(:,:,p) = ifftshift(H);

end
toc

nIms = zeros(length(true_sflist),1);
total_time = zeros(length(true_sflist),6);

%% loop over true SF
for true_ff = truesf2do
    
    % find the folder based on SF
    image_folder = dir(fullfile(image_path, sprintf('*%.2f_Contrast_0.80',true_sflist(true_ff))));
    image_folder = fullfile(image_folder(1).folder, image_folder(1).name);
    
    %% loop over true orientations
    for true_oo = trueori2do
    
        
        imlist = dir(fullfile(image_folder, sprintf('*_%ddeg.jpeg',true_oo)));

        im_file = fullfile(imlist(1).folder, imlist(1).name);
         
        fprintf('loading image %s\n',im_file);
    
        try
            image = imread(im_file);
        catch err
            fprintf('image %d could not be loaded!\n',filt_ff)
            continue
        end
        image_orig = image;

        % make this params struct to pass into my function
        params.R_MEAN = R_MEAN;
        params.G_MEAN = G_MEAN;
        params.B_MEAN = B_MEAN;

        params.process_at_size = process_at_size;
        params.size_after_pad = size_after_pad;
        params.filters_freq = filters_freq;

        params.ori_list = ori_list;
        params.wavelength_list = wavelength_list;


        %% do the processing in a separate function
        out = process_image_new_plots(image, params);

        mag = reshape(out.mag, size(image,1), size(image,2), length(freq_list), length(ori_list));
        phase = reshape(out.phase, size(image,1), size(image,2), length(freq_list), length(ori_list));
        
        %% make plots
       
        if plotFiltOut
            
        figure;
        set(gcf,'Color','w')
        hold all;kk=0;
        for filt_ff = length(freq_list):-1:1
            for filt_oo = 1:length(ori_list)
                kk=kk+1;
                subplot(length(freq_list),length(ori_list),kk);hold all;
                imagesc(mag(:,:,filt_ff,filt_oo));
                if filt_ff == length(freq_list)
                    title(sprintf('%.0f deg' ,ori_list(filt_oo)));
                end
                if filt_oo==1
                    ylabel(sprintf('%.2f cpp',freq_list(filt_ff)));
                end
                axis square
                set(gca,'XTick',[]);
                set(gca,'YTick',[]);
            end
        end
        suptitle(sprintf('True SF=%.2f\nTrue Ori = %d deg',true_sflist(true_ff), true_oo));
        
        end

        %% plot orientation content - separated by FILTER spatial frequency
        
        if plotOri
            
        figure;set(gcf,'Color','w')
        hold all;
        suptitle(sprintf('True SF=%.2f\nTrue Ori = %d deg',true_sflist(true_ff), true_oo));
        cols = viridis(length(freq_list));
        xx=0;
        ll = [];
        for filt_ff = length(freq_list):-1:1
            xx=xx+1;
            ll{xx} = sprintf('filtered at %.2f cpp',freq_list(filt_ff));
            mag_vals = squeeze(mean(mean(mag(:,:,filt_ff, :),2),1));
            mag_vals = zscore(mag_vals',[],2');
            plot(ori_list, mag_vals, 'Color', cols(xx,:))
        end
        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])
        legend(ll,'Location','EastOutside')
        plot([180-true_oo, 180-true_oo], get(gca,'YLim'),'Color','k')
       
        end
        %% plot SF content - separated by FILTER orientation
        if plotSF
            
        figure;set(gcf,'Color','w')
        hold all;
        suptitle(sprintf('True SF=%.2f\nTrue Ori = %d deg',true_sflist(true_ff), true_oo));
        cols = viridis(length(ori_list));
        xx=0;
        ll = [];
        for filt_oo = 1:length(ori_list)
            xx=xx+1;
            ll{xx} = sprintf('filtered at %.0f deg',ori_list(filt_oo));
            mag_vals = squeeze(mean(mean(mag(:,:,:,filt_oo),2),1));
            mag_vals = zscore(mag_vals',[],2');
            plot(freq_list, mag_vals, 'Color', cols(xx,:))
        end
        xlabel('cpp');
        ylabel('average magnitude')
        xlim([0,max(freq_list)])
        plot([true_sflist(true_ff), true_sflist(true_ff)], get(gca,'YLim'),'Color','k')
        
        legend(ll,'Location','EastOutside')
        
        end
        
    end
end