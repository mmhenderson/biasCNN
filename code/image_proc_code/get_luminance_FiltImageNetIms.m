%% visualize some basic properties of the evaluation image set.
% want to make sure there aren't any big systematic changes in luminance,
% contrast etc across orientations, or any weird spatial patterns that
% could influence measurement of tuning at earliest network layers.

%%

clear
close all

root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-1));

% image_set='FiltIms14AllSFCos';
% image_set = 'FiltIms7AllSFSquare';
image_set='FiltIms11Square_SF_0.01';
image_set_plot =image_set;
image_set_plot(image_set_plot=='_') = ' ';
% image_set='FiltNoiseSquare_SF_0.01';
% image_set= 'FiltIms3AllSFSquare';

nSets=1;

process_at_size=224;

ori_vals_deg = linspace(0,179,180);
nOri = numel(ori_vals_deg);
nImsPerOri=48;
% nImsPerOri=10;

R_MEAN = 124;
G_MEAN = 117;
B_MEAN = 104;

allims = zeros(nSets,nOri,nImsPerOri,process_at_size,process_at_size,3);

meanlum = zeros(nSets,nOri,nImsPerOri,3);
sdlum=zeros(nSets,nOri,nImsPerOri,3);

%% load all the images 


for ss=1:nSets

   
    for ii=1:nImsPerOri

%         for oo=1:15
        for oo=1:nOri

            if contains(image_set,'Noise')
                image_folder = fullfile(root, 'images','gratings',image_set, 'AllIms');
                fn2load = fullfile(image_folder,sprintf('FiltNoise_ex%d_%ddeg.png',ii,ori_vals_deg(oo)));
            else
%                 image_folder = fullfile(root, 'images','gratings',sprintf('%s_rand%d',image_set,ss), 'AllIms');
                image_folder = fullfile(root, 'images','gratings',sprintf('%s_rand%d',image_set,ss), 'AllIms');
                fn2load = fullfile(image_folder,sprintf('FiltImage_ex%d_%ddeg.png',ii,ori_vals_deg(oo)));
            end
            % loop over images and process

            fprintf('loading from %s\n',fn2load)
            try
                image = imread(fn2load);
            catch err
                fprintf('image %d could not be loaded!\n',ii)
                continue
            end
            image_orig = image;

            zero_centered=double(image) - permute(repmat([R_MEAN,G_MEAN,B_MEAN],process_at_size,1,process_at_size), [1,3,2]);
%             zero_centered=double(image_orig) - repmat(mean(mean(image_orig,2),1),process_at_size,process_at_size,1);

            allims(ss,oo,ii,:,:,:) = zero_centered;
           
            for cc=1:3
                vals=zero_centered(:,:,cc);
                meanlum(ss,oo,ii,cc) = mean(vals(:));
                sdlum(ss,oo,ii,cc)=std(vals(:));
            end


        end
    end
end


%% plot average luminance stats, all set

setcols = viridis(nSets);
figure;
subplot(2,2,1);hold all;
lumvals=meanlum(:,:,:,1);
vals=[];
for ss=1:nSets
    vals=[vals, squeeze(lumvals(ss,:,:))'];
end
% vals = reshape(permute(squeeze(meanlum(:,:,:,1)), [3,1,2,4]),nSets*nImsPerOri, nOri,1);
imagesc(squeeze(vals)');colorbar();
xlims=[0.5, (nImsPerOri*nSets)+0.5];
xlim(xlims);
ylims=[0.5, nOri+0.5];
ylim(ylims);
for ii=[0:45:180]
    plot(xlims,[ii,ii],'-','Color','k');
end
title('Mean luminance each image')
xlabel('Random image number')
ylabel('Orientation')

subplot(2,2,2);hold all;
lumvals=sdlum(:,:,:,1);
vals=[];
for ss=1:nSets
    vals=[vals, squeeze(lumvals(ss,:,:))'];
end
imagesc(squeeze(vals)');colorbar();
xlims=[0.5, (nImsPerOri*nSets)+0.5];
xlim(xlims);
ylims=[0.5, nOri+0.5];
ylim(ylims);
for ii=[0:45:180]
    plot(xlims,[ii,ii],'-','Color','k');
end
title('SD lum across pixels each image')
xlabel('Random image number')
ylabel('Orientation')

for ss=1:nSets    

    subplot(2,2,3);hold all;
    meanvals=squeeze(mean(meanlum(ss,:,:,1),3));
    plot(1:nOri,squeeze(meanlum(ss,:,:,1)),'Color',[0.8, 0.8, 0.8])
    plot(1:nOri,meanvals,'-','Color',setcols(ss,:))
    title('Mean luminance, averaged over images')
    xlabel('Orientation');
    set(gcf,'Color','w')

end
plot(1:nOri,squeeze(mean(mean(meanlum(:,:,:,1),3),1)),'-','Color','k')

for ss=1:nSets    

    subplot(2,2,4);hold all;
    
    sdvals=squeeze(std(meanlum(ss,:,:,1),[],3));
    plot(1:nOri,sdvals,'-','Color',setcols(ss,:))
    title('SD of mean image luminance, across ims within orientation')
    xlabel('Orientation');
    set(gcf,'Color','w')

end
plot(1:nOri,squeeze(mean(std(meanlum(:,:,:,1),[],3),1)),'-','Color','k')
suptitle(sprintf('%s\n%d sets',image_set_plot,nSets));

%% Plot average image, binned within orientations, averaged over sets

% plot all orients by 5 deg steps
spat_per_ori = squeeze(mean(mean(allims,3),1));
bin_size=5;
bin_centers = 0:bin_size:179;
npx = ceil(sqrt(length(bin_centers)));
npy=ceil(length(bin_centers)/npx);
c=1;
figure;
set(gcf,'Position',get(0,'ScreenSize'))
pi=0;
ax=[];
for bb=1:length(bin_centers)
    pi=pi+1;
    ax=[ax, subplot(npx,npy,pi)];
    ori2plot = abs(ori_vals_deg-bin_centers(bb))<bin_size/2 | abs(ori_vals_deg-180-bin_centers(bb))<bin_size/2;
    assert(sum(ori2plot)==bin_size);
    spat=squeeze(mean(spat_per_ori(ori2plot,:,:,c),1));
    imagesc(spat);
%     colorbar()
    axis equal off
    title(sprintf('%d deg',bin_centers(bb)));
end
match_clim(ax)
colorbar()
suptitle(sprintf('%s\nAverage of all images within each orient (%d sets)',image_set_plot,nSets));
set(gcf,'Color','w')

%% Plot average image, averaged over sets
% 
% % plot all orients by 5 deg steps
% spat_per_ori = squeeze(mean(mean(allims,3),1));
% ori2plot=1:180;
% npx = ceil(sqrt(length(ori2plot)));
% npy=ceil(length(ori2plot)/npx);
% c=1;
% figure;
% set(gcf,'Position',get(0,'ScreenSize'))
% pi=0;
% ax=[];
% for oo=1:length(ori2plot)
%     pi=pi+1;
%     ax=[ax, subplot(npx,npy,pi)];
%     spat=squeeze(spat_per_ori(ori2plot(oo),:,:,c));
%     imagesc(spat);
% %     colorbar()
%     axis equal off
%     title(sprintf('%d deg',ori2plot(oo)))
% end
% match_clim(ax)
% colorbar();
% suptitle(sprintf('%s\nAverage of all images within each orient (%d sets)',image_set_plot,nSets));
% set(gcf,'Color','w')

%% Plot average image within the mask ring only, averaged over sets
% close all
% 
% outer_range = 100;
% inner_range = 50;
% center=112;
% 
% gridvals=process_at_size/2*linspace(-1,1,process_at_size);
% [xgrid,ygrid]=meshgrid(gridvals,gridvals);
% xgrid=xgrid(:);
% ygrid=ygrid(:);
% 
% rad=sqrt(xgrid.^2+ygrid.^2);
% 
% rad = reshape(rad,process_at_size, process_at_size);
% vals2use = rad<outer_range & rad>inner_range;
% 
% % plot all orients by 5 deg steps
% spat_per_ori = squeeze(mean(mean(allims,3),1));
% ori2plot=1:180;
% npx = ceil(sqrt(length(ori2plot)));
% npy=ceil(length(ori2plot)/npx);
% c=1;
% figure;
% set(gcf,'Position',get(0,'ScreenSize'))
% pi=0;
% ax=[];
% for oo=1:length(ori2plot)
%     pi=pi+1;
%     ax=[ax, subplot(npx,npy,pi)];
%     spat=squeeze(spat_per_ori(ori2plot(oo),:,:,c));
%     imagesc(spat.*vals2use);
% %     colorbar()
%     axis equal off
%     title(sprintf('%d deg',ori2plot(oo)))
% end
% match_clim(ax)
% suptitle('Average of all images within each orient (all sets)');
% set(gcf,'Color','w')

%% plot luminance versus orientation for artifact regions, all sets

figure;hold all

% x coords (W)
vals1=[109:115];
% y coords (H)
vals2=[22:28];

[x,y] = meshgrid(224-vals2, vals1);
coords = [x(:),y(:)];

lum_vals=zeros(nOri,nImsPerOri,size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,:,cc)=squeeze(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),1));
end
subplot(2,2,1);hold all;
plot(ori_vals_deg,mean(lum_vals,3),'Color',[0.8, 0.8, 0.8])
plot(ori_vals_deg,mean(mean(lum_vals,3),2),'Color','k')
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('bottom')


[x,y] = meshgrid(vals2, vals1);
coords = [x(:),y(:)];

lum_vals=zeros(nOri,nImsPerOri,size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,:,cc)=squeeze(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),1));
end
subplot(2,2,2);hold all;
plot(ori_vals_deg,mean(lum_vals,3),'Color',[0.8, 0.8, 0.8])
plot(ori_vals_deg,mean(mean(lum_vals,3),2),'Color','k')
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('top')


[x,y] = meshgrid(vals1, vals2);
coords = [x(:),y(:)];


lum_vals=zeros(nOri,nImsPerOri,size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,:,cc)=squeeze(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),1));
end
subplot(2,2,3);hold all;
plot(ori_vals_deg,mean(lum_vals,3),'Color',[0.8, 0.8, 0.8])
plot(ori_vals_deg,mean(mean(lum_vals,3),2),'Color','k')
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('left')

[x,y] = meshgrid(vals1,224- vals2);
coords = [x(:),y(:)];

lum_vals=zeros(nOri,nImsPerOri,size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,:,cc)=squeeze(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),1));
end
subplot(2,2,4);hold all;
plot(ori_vals_deg,mean(lum_vals,3),'Color',[0.8, 0.8, 0.8])
plot(ori_vals_deg,mean(mean(lum_vals,3),2),'Color','k')
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('right')

suptitle(sprintf('%s\nLuminance of images around mask edge',image_set_plot))

%% plot luminance versus orientation for corner regions, all sets

figure;hold all

% x coords (W)
vals1=[1:6];
% y coords (H)
vals2=[1:6];

[x,y] = meshgrid(224-vals2, vals1);
coords = [x(:),y(:)];

lum_vals=zeros(nOri, size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,cc)=squeeze(mean(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),3),1));
end
subplot(2,2,1);
plot(ori_vals_deg,mean(lum_vals,2))
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('bottom left')


[x,y] = meshgrid(vals2, vals1);
coords = [x(:),y(:)];

lum_vals=zeros(nOri, size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,cc)=squeeze(mean(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),3),1));
end
subplot(2,2,2);
plot(ori_vals_deg,mean(lum_vals,2))
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('top left')


[x,y] = meshgrid(vals2,224-vals1);
coords = [x(:),y(:)];


lum_vals=zeros(nOri, size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,cc)=squeeze(mean(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),3),1));
end
subplot(2,2,3);
plot(ori_vals_deg,mean(lum_vals,2))
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('top right')

[x,y] = meshgrid(224-vals2,224-vals1);
coords = [x(:),y(:)];

lum_vals=zeros(nOri, size(coords,1));
for cc=1:size(coords,1)
    lum_vals(:,cc)=squeeze(mean(mean(allims(:,:,:,coords(cc,1),coords(cc,2)),3),1));
end
subplot(2,2,4);
plot(ori_vals_deg,mean(lum_vals,2))
xlabel('Orientation')
ylabel('mean luminance at desired region')
set(gcf,'Color','w')
title('bottom right')

suptitle(sprintf('%s\nLuminance of images at corners',image_set_plot))

