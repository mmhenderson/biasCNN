%% do orientation/SF content analysis of a single image, after it has been
% rotated by varying increments
% make sure that the rotations did what we think they did!
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

image_path = fullfile(root, 'images','ImageNet','ILSVRC2012');

rots = [0,22,45];

plotFiltOut = 1;
plotSF = 0;
plotOri = 0;
plotOriBySF=1;

% list the ground truth spat freq for gratings
% true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
true_sflist = [0.02, 0.04, 0.09,0.20];

% which SF and orientation do i want to plot? 
sf2filter = [1:4];
% ori2filter = [5:5:180];
ori2filter = [22.5:22.5:180];

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
% process_at_size = 1000;
process_at_size=224;

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

% loop over the rotations of same image
for rr=1:length(rots)

    %% load the image
    % picking one at random here, could use any one
    im_file = fullfile(image_path,sprintf('train_rot_%d',rots(rr)),'n02091032','n02091032_1797.jpeg');

    fprintf('loading image %s\n',im_file);

    image = imread(im_file);

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
    out = process_image(image, params);

    mag = reshape(out.mag, size(image,1), size(image,2), length(freq_list), length(ori_list));
    phase = reshape(out.phase, size(image,1), size(image,2), length(freq_list), length(ori_list));

    %% make plots

    if plotFiltOut

    figure;
    set(gcf,'Color','w')
    hold all;kk=0;
    for filt_ff = length(freq_list):-1:1
        axes = [];
        for filt_oo = 1:length(ori_list)
            kk=kk+1;
            axes = [axes, subplot(length(freq_list),length(ori_list),kk)];hold all;
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
        match_clim(axes);
    end
    suptitle(sprintf('Image rot %d deg',rots(rr)));

    end

    %% plot orientation content - separated by FILTER spatial frequency

    if plotOriBySF

        figure;set(gcf,'Color','w')
        hold all;
        suptitle(sprintf('Image rot %d deg',rots(rr)));

        cols = parula(length(freq_list));
        xx=0;
        ll = [];
        for filt_ff = 1:length(freq_list)
        xx=xx+1;
        ll{xx} = sprintf('filtered at %.2f cpp',freq_list(filt_ff));
        mag_vals = squeeze(mean(mean(mag(:,:,filt_ff, :),2),1));
%         mag_vals = zscore(mag_vals',[],2);
        plot(ori_list, mag_vals, 'Color', cols(xx,:))
        end
        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])
        legend(ll,'Location','EastOutside')
        plot([90+rots(rr), 90+rots(rr)], get(gca,'YLim'),'Color','k')

    end
    
    %% plot ori content averaging over SF
    if plotOri
        
        figure;set(gcf,'Color','w')
        hold all;
        suptitle(sprintf('Image rot %d deg\naverage of all SF',rots(rr)));

        cols = viridis(length(freq_list));

        mag_vals = squeeze(mean(mean(mean(mag(:,:,:,:),3),2),1));
        mag_vals = zscore(mag_vals',[],2');
        plot(ori_list, mag_vals, 'Color', cols(1,:))

        xlabel('degrees');
        ylabel('average magnitude')
        xlim([min(ori_list),max(ori_list)])

        plot([90+rots(rr), 90+rots(rr)], get(gca,'YLim'),'Color','k')

    end
    %% plot SF content - separated by FILTER orientation
    if plotSF

        figure;set(gcf,'Color','w')
        hold all;
        suptitle(sprintf('Image rot %d deg',rots(rr)));

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

        legend(ll,'Location','EastOutside')

    end

end