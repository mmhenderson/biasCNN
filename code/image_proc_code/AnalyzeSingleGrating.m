%% do orientation/SF content analysis of a grating image
% make a plot showing the outcome of filtering for a range of orientations
% and SFs
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

% image_set = 'SmoothedCosGratings';
image_set = 'FiltImsAllSFCos_rand1'

image_path = fullfile(root, 'images','gratings',image_set);
save_path = fullfile(root, 'image_stats','gratings',image_set);

if ~isfolder(save_path)
    mkdir(save_path)
end

plotFiltOut = 1;
plotSF = 1;
plotOri = 1;
plotImageOrig = 1;

% list the ground truth spat freq  and orientations for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
true_orilist = 1:180;

% Going to analyze one grating here - what are its SF and orient index?
truesf2do = [4];
trueori2do = [45];

%% specify the spatial frequencies and orientations to filter at
sf2filter = true_sflist([1:6]);
nSF_filt = length(sf2filter);

ori2filter = true_orilist([22,45,67,90,112,135,157,180]);
nOri_filt = length(ori2filter);

%% more parameters for the images

% how big are the images when we do our analysis? this is the same
% as the size that VGG-16 preprocesing resizes them to, after crop.
process_at_size = 224;

% set an amount of downsampling, for speed of processing
resize_factor = 1;  % if one then using actual size

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

%% make the filter bank (will be same for all images we look at)

fprintf('making filters...\n')
tic

GaborBank = gabor(1./sf2filter.*resize_factor,ori2filter);

% based on the GaborBank - list the orientations and SF of the filters
% note that this might be sorted differently than what we put in, this
% order is the correct one.
orilist_bank = [GaborBank.Orientation];
% this is the orientation of the filter, switched into the same coordinate
% system as the grating images I drew (clockwise from 0, where 0=vertical).
orilist_bank_fliptoCW = 180 - orilist_bank;
sflist_bank = 1./[GaborBank.Wavelength];
wavelist_bank = [GaborBank.Wavelength];

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

%% make this params struct to pass into my function
params.R_MEAN = R_MEAN;
params.G_MEAN = G_MEAN;
params.B_MEAN = B_MEAN;

params.process_at_size = process_at_size;
params.size_after_pad = size_after_pad;
params.filters_freq = filters_freq;
params.ori_list = orilist_bank; % these don't get used for anything
params.wavelength_list = wavelist_bank; % these don't get used for anything

%% loop over true SF
for true_ff = truesf2do
    
    % find the folder based on SF
    image_folder = dir(fullfile(image_path, sprintf('*%.2f_Contrast_0.80',true_sflist(true_ff))));
    image_folder = fullfile(image_folder(1).folder, image_folder(1).name);
    
    %% loop over true orientations
    for true_oo = trueori2do
    
        
        imlist = dir(fullfile(image_folder, sprintf('*_%ddeg.png',true_oo)));

        im_file = fullfile(imlist(1).folder, imlist(1).name);
         
        fprintf('loading image %s\n',im_file);
    
        try
            image = imread(im_file);
        catch err
            fprintf('image %d could not be loaded!\n',filt_ff)
            continue
        end
        image_orig = image;
        
        
        if plotImageOrig
            figure;hold all;
            imagesc(image_orig);
            set(gca,'YDir','reverse');
            axis square off
            title(sprintf('True SF=%.2f\nTrue Ori=%d deg',true_sflist(true_ff), true_orilist(true_oo)));
        end

        %% do the processing in a separate function
        out = process_image(image, params);

        % want output as [nPix x nPix x nSF x nOri]
        mag = reshape(out.mag, size(image,1), size(image,2), nSF_filt, nOri_filt);
        phase = reshape(out.phase, size(image,1), size(image,2), nSF_filt, nOri_filt);
       
        % list of the ori and SF that correspond to each filter
        orilist_out = reshape(orilist_bank_fliptoCW, nSF_filt, nOri_filt);
        sflist_out = reshape(sflist_bank, nSF_filt, nOri_filt);
        
        mag = round(mag);
        phase = round(phase);
        %% make plots
       
        if plotFiltOut
            
            figure;
            set(gcf,'Color','w')
            hold all;kk=0;
            for filt_ff = 1:nSF_filt
                axes = [];
                for filt_oo = 1:nOri_filt
                    kk=kk+1;
                    axes = [axes, subplot(nSF_filt, nOri_filt,kk)];
                    % IMPORTANT - NEED TO REVERSE Y DIRECTION HERE SO THAT
                    % THE PLOTS MAKE SENSE (BECAUSE THEY'RE SUBPLOTS)
                    set(gca,'YDir','reverse');
                    hold all;
                    imagesc(mag(:,:,filt_ff,filt_oo));
                    if filt_ff == 1
                        title(sprintf('%.0f deg' ,orilist_out(filt_ff,filt_oo)));
                    end
                    if filt_oo == 1
                        ylabel(sprintf('%.2f cpp',sflist_out(filt_ff,filt_oo)));
                    end
                    axis square
                    set(gca,'XTick',[]);
                    set(gca,'YTick',[]);
                end
                match_clim(axes);
            end
            sgtitle(sprintf('True SF=%.2f\nTrue Ori=%d deg',true_sflist(true_ff), true_orilist(true_oo)));

        end

        %% plot orientation content - separated by FILTER spatial frequency
        
        if plotOri
            
            figure;set(gcf,'Color','w')
            hold all;
            sgtitle(sprintf('True SF=%.2f\nTrue Ori=%d deg',true_sflist(true_ff), true_orilist(true_oo)));
            cols = viridis(nSF_filt);
            xx=0;
            ll = [];
           	% note that ori_axis is descending order, not ascending.
            ori_axis = orilist_out(1,:);
            for filt_ff = 1:nSF_filt
                xx=xx+1;
                ll{xx} = sprintf('filtered at %.2f cpp',sflist_out(filt_ff,1));
                mag_vals = squeeze(mean(mean(mag(:,:,filt_ff, :),2),1));
                mag_vals = zscore(mag_vals',[],2');
                plot(ori_axis, mag_vals, 'Color', cols(xx,:))
            end
            xlabel('degrees');
            ylabel('average magnitude')
            xlim([min(true_orilist),max(true_orilist)])
            legend(ll,'Location','EastOutside')
            plot([true_oo, true_oo], get(gca,'YLim'),'Color','k')

        end
        %% plot SF content - separated by FILTER orientation
        if plotSF
            
            figure;set(gcf,'Color','w')
            hold all;
            sgtitle(sprintf('True SF=%.2f\nTrue Ori = %d deg',true_sflist(true_ff), true_orilist(true_oo)));
            cols = viridis(nOri_filt);
            xx=0;
            ll = [];
            % note that sf axis is in descending order, not ascending.
            sf_axis = sflist_out(:,1);
            for filt_oo = 1:nOri_filt
                xx=xx+1;
                ll{xx} = sprintf('filtered at %.0f deg',orilist_out(1,filt_oo));
                mag_vals = squeeze(mean(mean(mag(:,:,:,filt_oo),2),1));
                mag_vals = zscore(mag_vals',[],2');
                plot(sf_axis, mag_vals, 'Color', cols(xx,:))
            end
            xlabel('cpp');
            ylabel('average magnitude')
            xlim([0,max(sf_axis)])
            plot([true_sflist(true_ff), true_sflist(true_ff)], get(gca,'YLim'),'Color','k')

            legend(ll,'Location','EastOutside')

        end
        
    end
end