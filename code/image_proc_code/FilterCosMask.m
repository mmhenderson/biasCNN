%% do orientation/SF content analysis of the gaussian mask
% make a plot showing the outcome of filtering for a range of orientations
% and SFs
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

plotFiltOut = 1;
plotSF = 0;
plotOri = 0;
plotOriBySF = 1;

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
% true_sflist = [0.02, 0.04, 0.09,0.20];

% which SF and orientation do i want to plot? 
sf2filter = [1:6];
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

%% make the mask
% making a circular mask with cosine fading to background
cos_mask = zeros(process_at_size);
values = process_at_size./2*linspace(-1,1,process_at_size);
[gridx,gridy] = meshgrid(values,values);
r = sqrt(gridx.^2+gridy.^2);

% creating three ring sections based on distance from center
outer_range = 100;
inner_range = 50;

% inner values: set to 1
cos_mask(r<inner_range) = 1;

% middle values: create a smooth fade
faded_inds = r>=inner_range & r<outer_range;
cos_mask(faded_inds) = 0.5*cos(pi/(outer_range-inner_range).*(r(faded_inds)-inner_range)) + 0.5;

% outer values: set to 0
cos_mask(r>=outer_range) = 0;

%%
image = repmat(cos_mask,1,1,3);

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
% expand_by = (process_at_size-size(image,1))/2;
% image = [zeros(expand_by,size(image,2));image; zeros(expand_by,size(image,2))];
% image = [zeros(size(image,1),expand_by),image, zeros(size(image,1),expand_by)];
% % image = double(image)./max(double(image(:)));
image = double(image)./2 + mean([R_MEAN,G_MEAN,B_MEAN]);
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
suptitle('Mask image w filters applied');

end

%% plot orientation content - separated by FILTER spatial frequency

if plotOriBySF

    figure;set(gcf,'Color','w')
    hold all;
   suptitle('Mask image w filters applied');

    cols = parula(length(freq_list));
    xx=0;
    ll = [];
    for filt_ff = 1:length(freq_list)
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
    plot([90, 90], get(gca,'YLim'),'Color','k')

end

%% plot orietnation content, averaged over SF

if plotOri

    figure;set(gcf,'Color','w')
    hold all;
    suptitle('Mask image w filters applied');

    mag_vals = squeeze(mean(mean(mean(mag(:,:,:, :),3),2),1));
    mag_vals = zscore(mag_vals',[],2');
    plot(ori_list, mag_vals, 'Color', cols(1,:))

    xlabel('degrees');
    ylabel('average magnitude')
    xlim([min(ori_list),max(ori_list)])

    plot([90, 90], get(gca,'YLim'),'Color','k')

end
%% plot SF content - separated by FILTER orientation
if plotSF

figure;set(gcf,'Color','w')
hold all;
suptitle('Mask image w filters applied');

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
