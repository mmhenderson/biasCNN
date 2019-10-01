%%
clear
close all

root = pwd;

%% now testing with images whose orientation content is known
root_root = '/usr/local/serenceslab/maggie/biasCNN/';

ori2load = 30;
sf2load = 0.22;
exp_ori_peak = 180-ori2load;
exp_sf_peak = sf2load;

test_file = fullfile(root_root,'images','grating_ims_13e',sprintf('SF_%.2f_Contrast_0.95',sf2load*10),['Gaussian_phase0_ex1_' num2str(ori2load) 'deg.png']);

image = imread(test_file);

%%
if isempty(gcp('nocreate'))
    parpool(12);
end
tic 

wavelengthlist = [1/sf2load-0.4, 1/sf2load, 1+1/sf2load];
freqlist = 1./wavelengthlist;
orilist = 5:5:180;
mag = zeros(size(image,1),size(image,2),length(wavelengthlist),length(orilist));

parfor oo=1:length(orilist)
    
    gaborbank1 = gabor(wavelengthlist,orilist(oo));
    [this_mag,phase] = imgaborfilt(image,gaborbank1);
    mag(:,:,:,oo) = this_mag;
    
end

toc
%%
% tic 
% 
% freqlist = sort([0.1:0.05:0.50],'descend'); 
% orilist = 1:180;
% % mag = zeros(size(image,1),size(image,2),length(freqlist),length(orilist));
% 
% % parfor oo=1:length(orilist)
%     
% gaborbank1 = gabor(1./freqlist,orilist);
% [this_mag,phase] = imgaborfilt(image,gaborbank1);
% mag = reshape(this_mag, [size(this_mag,1),size(this_mag,2), length(freqlist),length(orilist)]);
% 
% % end
% 
% toc
%%
close all;

ori_hist = squeeze(mean(mean(mean(mag,2),1),3));
[~,peak] = max(ori_hist);
figure;hold all;
plot(orilist,ori_hist)
line([orilist(peak),orilist(peak)],get(gca,'YLim'),'Color','r')
line([exp_ori_peak,exp_ori_peak],get(gca,'YLim'),'Color','k')

sf_hist = squeeze(mean(mean(mean(mag,2),1),4));
[~,peak] = max(sf_hist);
figure;hold all;
plot(freqlist, sf_hist)
line([freqlist(peak),freqlist(peak)],get(gca,'YLim'),'Color','r')
line([exp_sf_peak,exp_sf_peak],get(gca,'YLim'),'Color','k')
% gure;imagesc(mag1-mag2);colorbar()
% figure;imagesc(image)

