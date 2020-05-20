%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

plotDisc = 1;
plotCorr = 1;
% image_set = 'CosGratings';
% image_set = 'FiltIms9Cos';
% image_set = 'FiltIms9AllSFCos_rand1';
% image_set = 'SquareGratings_Smooth_Big';
% image_set = 'SquareGratings_Smooth_Small';
image_set = 'SpatFreqGratings';
image_set_plot = image_set;
image_set_plot(image_set_plot=='_') = ' ';

image_path = fullfile(root, 'images','gratings',image_set);

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
sf2do = [2:6];
nSF = length(sf2do);

orilist = [0:179];
nOri = length(orilist);

noise_level = 0.01;

nPhase = 1;
nEx = 4;
nExTot=nEx*nPhase;

phaselist=repmat([0],nEx,1);
% phaselist=repmat([0;1],nEx,1);
exlist=repelem((1:nEx)',nPhase);
%% loop over spatial frequencies
for ff = sf2do
    
    % discrim = zeros(nOri,nOri,nEx*nEx);
    discrim_curve = zeros(nOri-1,nExTot*nExTot);
    corr_curve = zeros(nOri-1,nExTot*nExTot);

    % find all the folders in this directory
    image_folder = dir(fullfile(image_path, sprintf('*%.2f*',true_sflist(ff))));
    
    %% loop over images

    for oo = 1:nOri
        
        for ee = 1:nExTot

           fn2load = fullfile(image_folder(1).folder, image_folder(1).name,sprintf('Gaussian_phase%d_ex%d_%ddeg.png',phaselist(ee),exlist(ee),orilist(oo)));
           fprintf('loading %s\n',fn2load);
           im = imread(fn2load);
           if size(im,3)>1
               im = rgb2gray(im);
           end
           if oo==1 && ee==1
               images = zeros(nOri,size(im,1),size(im,2),nExTot);    
           end
           images(oo,:,:,ee) = im;
        end
        
    end
   
       %%
    ax=[];
    pairs = {[0,1],[45,46]};
    for pp=1:length(pairs)
       pair = pairs{pp};
       im1 = squeeze(images(pair(1)+1,:,:,1));
       im2 = squeeze(images(pair(2)+1,:,:,1));
       figure;hold all;
       set(gcf,'Color','w');
       ax=[ax, gca];
       imagesc(im2-im1);colorbar();axis square off
       title(sprintf('%d deg - %d deg\n%s (%.2f cpp)',pair(1),pair(2),image_set_plot,true_sflist(ff)))
    end
    match_clim(ax);

    %%
    for or1 = 1:nOri-1
        or2=or1+1;
        if or2>180
            or2=1;
        end
        % go 1 step clockwise of ori1
        % then do all nEx^2 pairwise comparisons
        fprintf('or1=%d, or2=%d\n',or1,or2);
        ee=0;
        for ee1 = 1:nExTot
            for ee2 = 1:nExTot
                ee=ee+1;

                image1 = images(or1,:,:,ee1);
                image2 = images(or2,:,:,ee2);

                assert(~all(image1(:)==image2(:)))
                dist = sqrt(sum((image1(:)-image2(:)).^2));
                discrim_curve(or1,ee) = dist;

                c = corrcoef(image1(:),image2(:));
                corr_curve(or1,ee) = c(2,1);

            end
        end
    end


    %% 
    if plotDisc

        figure;hold all;
         set(gcf,'Color','w');
        for ee=1:nExTot^2
            plot(orilist(1:size(discrim_curve,1)),discrim_curve(:,ee),'-','Color',[0.8,0.8,0.8]);
        end
        plot(orilist(1:size(discrim_curve,1)),mean(discrim_curve(:,:),2),'-','Color','k');
        set(gca,'XTick',[0:45:180]);
        xlabel('orientation (deg)')
        ylabel('euclidean distance');
        set(gcf,'Color','w')
        title(sprintf('%s (%.2f cpp)\nPixelwise discriminability of 1 degree steps',image_set_plot, true_sflist(ff)));
    end

    %%
    if plotCorr
        figure;hold all;
         set(gcf,'Color','w');
        for ee=1:nExTot^2
            plot(orilist(1:size(corr_curve,1)),corr_curve(:,ee),'-','Color',[0.8,0.8,0.8]);
        end
        plot(orilist(1:size(discrim_curve,1)),mean(corr_curve(:,:),2),'-','Color','k');
        set(gca,'XTick',[0:45:180]);
        xlabel('orientation (deg)')
        ylabel('image correlation');
        set(gcf,'Color','w')
        title(sprintf('%s (%.2f cpp)\nPixelwise correlation between 1 degree steps', image_set_plot,true_sflist(ff)));
    end
end