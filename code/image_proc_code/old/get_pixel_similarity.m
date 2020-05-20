%% get the orientation content of a set of gaussian windowed gratings
% ground truth for this image analysis
%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

plotDisc = 1;
plotCorr = 0;
% image_set = 'CosGratings';
% image_set = 'FiltIms9Cos';
% image_set = 'FiltIms9AllSFCos_rand1';
% image_set = 'SquareGratings_Smooth_Big';
% image_set = 'SquareGratings_Smooth_Small';
image_set = 'SpatFreqGratings';

image_path = fullfile(root, 'images','gratings',image_set);

% list the ground truth spat freq for gratings
true_sflist = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
sf2do = [1];
nSF = length(sf2do);

orilist = [0:179];
nOri = length(orilist);

noise_level = 0.01;

nEx = 1;

discrim = zeros(nSF,nOri,nOri,nEx);
discrim_curve = zeros(nSF,nOri-1,nEx);
corr_curve = zeros(nSF, nOri-1,nEx);

%% loop over spatial frequencies
for ff = sf2do
    
    % find all the folders in this directory
    image_folder = dir(fullfile(image_path, sprintf('*%.2f*',true_sflist(ff))));
    
    %% loop over images
    
    fprintf('processing folder %d of %d\n',ff,length(true_sflist));
        
    
    for oo = 1:nOri
        
        for ee = 1:nEx

           fn2load = fullfile(image_folder(1).folder, image_folder(1).name,sprintf('Gaussian_phase0_ex%d_%ddeg.png',ee,orilist(oo)));
           im = imread(fn2load);
           if size(im,3)>1
               im = rgb2gray(im);
           end
           if oo==1 && ee==1
               images = zeros(nOri,size(im,1),size(im,2),nEx);    
           end
           images(oo,:,:,ee) = im;
        end
        
    end
    
    pairs = {[0,1],[45,46]};
    for pp=1:length(pairs)
       pair = pairs{pp};
       im1 = squeeze(images(pair(1)+1,:,:,1));
       im2 = squeeze(images(pair(2)+1,:,:,1));
       figure;imagesc(im2-im1);colorbar();axis square
       title(sprintf('%d deg - %d deg\n%s - %.2f cpp',pair(1),pair(2),image_set,true_sflist(ff)))
    end
    
    
    for ee = 1:nEx
        for or1 = 1:nOri
            for or2 = or1+1:nOri
                image1 = images(or1,:,:,ee);
                image2 = images(or2,:,:,ee);
                dist = sqrt(sum((image1(:)-image2(:)).^2));
                % get euclidean distance
                discrim(ff,or1,or2,ee) = dist;
                discrim(ff,or2,or1,ee) = dist;

                if abs(or1-or2)==1
                    fprintf('or1=%d, or2=%d, dist = %.2f\n',or1,or2,dist);
                    discrim_curve(ff,min([or1,or2]),ee) = dist;
                    c = corrcoef(image1(:),image2(:));
                    corr_curve(ff,min([or1,or2]),ee) = c(2,1);
                end

            end
        end
    end

    %% 
    if plotDisc
        
        figure;hold all;
        for ee=1:nEx
            plot(orilist(1:size(discrim_curve,2)),discrim_curve(ff,:,ee),'o');
        end
        plot(orilist(1:size(discrim_curve,2)),mean(discrim_curve(ff,:,:),3),'-','Color','k');
        set(gca,'XTick',[0:45:180]);
        xlabel('orientation (deg)')
        ylabel('euclidean distance');
        set(gcf,'Color','w')
        title(sprintf('%s, noise=%.2f\nPixelwise discriminability of 1 degree steps\nSF = %.2f cpp',image_set,noise_level,true_sflist(ff)));
    end
    
    %%
    if plotCorr
        figure;hold all;
        for ee=1:nEx
            plot(orilist(1:size(corr_curve,2)),corr_curve(ff,:,ee),'o');
        end
        plot(orilist(1:size(discrim_curve,2)),mean(corr_curve(ff,:,:),3),'-','Color','k');
        set(gca,'XTick',[0:45:180]);
        xlabel('orientation (deg)')
        ylabel('iamge correlation');
        set(gcf,'Color','w')
        title(sprintf('%s, noise=%.2f\nPixelwise correlation between 1 degree steps\nSF = %.2f cpp',image_set,noise_level,true_sflist(ff)));
    end
end