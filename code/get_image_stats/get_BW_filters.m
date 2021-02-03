close all
clear
freq_list = logspace(log10(0.02), log10(.2),4);
[wavelength_list,sorder] = sort(1./freq_list,'ascend');
freq_list = freq_list(sorder);
resize_factor = 1;  
 process_at_size = 224;
ori_list = 5:5:180;
fprintf('making filters...\n')

GaborBank = gabor(wavelength_list.*resize_factor,ori_list);

sizeLargestKernel = size(GaborBank(end).SpatialKernel);
% Gabor always returns odd length kernels
padding_needed = (sizeLargestKernel-1)/2;

max_pix = process_at_size;  % maximum size of any image dimension
% FIX this so that we can make the filters ahead of time
size_after_pad = max_pix*resize_factor+padding_needed*2;
size_after_pad = size_after_pad + mod(size_after_pad,2);
% making a matrix [nPix x nPix x nFilters]
filters_freq = zeros([size_after_pad,length(GaborBank)]);

% for p = 1:length(GaborBank)
%%
% H = makeFrequencyDomainTransferFunction_MMH(GaborBank(p),size_after_pad);
Gabor = GaborBank(141);
imageSize=size_after_pad;

M = imageSize(1);
N = imageSize(2);

if mod(N,2)
    u = linspace(-0.5+1/(2*N),0.5-1/(2*N),N);
else
    u = linspace(-0.5,0.5-1/N,N); 
end

if mod(M,2)
    v = linspace(-0.5+1/(2*M),0.5-1/(2*M),M);
else
    v = linspace(-0.5,0.5-1/M,M); 
end

[U,V] = meshgrid(u,v);

%     orient=15
%     Uprime = U .*cosd(orient) - V .*sind(orient);
%     Vprime = U .*sind(orient) + V .*cosd(orient);

Uprime = U .*cosd(Gabor.Orientation) - V .*sind(Gabor.Orientation);
Vprime = U .*sind(Gabor.Orientation) + V .*cosd(Gabor.Orientation);

% From relationship in "Nonlinear Operator in Oriented Texture", Kruizinga,
% Petkov, 1999.
BW = Gabor.SpatialFrequencyBandwidth;
SigmaX = Gabor.Wavelength/pi*sqrt(log(2)/2)*(2^BW+1)/(2^BW-1);
SigmaY = SigmaX ./ Gabor.SpatialAspectRatio;

sigmau = 1/(2*pi*SigmaX);
sigmav = 1/(2*pi*SigmaY);
freq = 1/Gabor.Wavelength;

A = 2*pi*SigmaX*SigmaY;

H = A.*exp(-0.5*( ((Uprime-freq).^2)./sigmau^2 + Vprime.^2 ./ sigmav^2) );


% filters_freq(:,:,p) = ifftshift(H);

%%
u2plot = u;
v2plot = v;
close all
% for 180 deg, x is ori, y is SF
max_sf = find(max(H,[],1)==max(max(H,[],1)));
max_ori = find(max(H,[],2)==max(max(H,[],2)));
center_ori = v2plot(max_ori);
center_sf = u2plot(max_sf);

cross_sect_sf = H(max_ori,:);
cross_sect_ori = H(:,max_sf);

figure;plot(v2plot, cross_sect_sf)
height = max(cross_sect_sf) - min(cross_sect_sf);
[~,sorder] = sort(abs(cross_sect_sf-height/2),'ascend');
inds = sorder(1:2);
fwhm_sf = abs(v2plot(inds(2)) - v2plot(inds(1)));
title(sprintf('SF\ncenter=%.2f a.u., fwhm=%.2f a.u.',center_sf, fwhm_sf))

figure;plot(u2plot, cross_sect_ori);
height = max(cross_sect_ori) - min(cross_sect_ori);
[~,sorder] = sort(abs(cross_sect_ori-height/2),'ascend');
inds = sorder(1:2);
fwhm_ori = abs(u2plot(inds(2)) - u2plot(inds(1)));
fwhm_ori_deg = 2*asind(fwhm_ori/2/abs(center_sf));
title(sprintf('Ori\ncenter=%.2f a.u., fwhm=%.2f a.u. or %.1f deg',center_ori, fwhm_ori, fwhm_ori_deg))

%%
figure;imagesc(H)
%%
H_new = H;
H_new(max_ori,:) = 1000;
H_new(:,max_sf) = 1000;
figure;imagesc(H_new)
% end
% toc
