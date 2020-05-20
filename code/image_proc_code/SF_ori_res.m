clear
close all
%% first define stuff about orientation/spatial frequency image space

orient_vals_deg = linspace(0,179,180);
nOri = numel(orient_vals_deg);
orient_kappa_deg=1000;
nImsPerOri=48;
nImsTotal=nOri*nImsPerOri;

% this is the height and width of the final images
scale_by=1;
final_size_pix = 224*scale_by;

% first, define sampling rate and frequency axis
samp_rate_pix = 1;   % samples per pixel, always one.
% get frequency axis
nframes = final_size_pix;
nyq = .5*samp_rate_pix;         
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpp = samp_rate_pix/nframes;  
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;
center_pix = find(fax==min(abs(fax)));

% next, we're listing the frequencies corresponding to each point in
% the FFT representation, in grid form
[ x_freq, y_freq ] = meshgrid(fax,fax);
x_freq = x_freq(:);y_freq = y_freq(:);
% converting these into a polar format, where angle is orientation and
% magnitude is spatial frequency
[ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
% adjust the angles so that they go from 0-pi in rads
ang_grid = mod(ang_grid,pi);
ang_grid = reshape(ang_grid,final_size_pix,final_size_pix);
mag_grid = reshape(mag_grid,final_size_pix,final_size_pix);


%% now calculate how orientation resolution should change as a function of SF
% linear relationship by this calculation, which is approximate
% assume that the resolvable number of orientations at each spatial
% frequency in the spectral domain is the circumferene of the cirle locate
% at that distance from the origin, in pixels

sf_list = logspace(log10(0.01), log10(1.2),100);
% sf_list = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25];
pix_per_deg_list = zeros(size(sf_list));
for ii = 1:length(sf_list)
    % C=2*pi*r;
    [~,sfind] = min(abs(fax-sf_list(ii)));
    r_pix = abs(sfind-center_pix);
    C_pix = 2*pi*r_pix;
    pix_per_deg_list(ii) = C_pix/360;
end

figure;hold all;
set(gcf,'Color','w');
plot(sf_list,pix_per_deg_list);
ylabel('Orientation resolution (pix per degree)')
xlabel('Spatial frequency (cpp)')
plot(get(gca,'XLim'),[1,1],'Color','k')
plot(get(gca,'XLim'),[0.2, 0.2],'Color','k')
title(sprintf('Orient resolution for each SF, images %d pix in size',final_size_pix));