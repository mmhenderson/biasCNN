function [images] = get_filtered_noise(params, nIms, randseed)
% filter Gaussian noise to make an image with specified spatial
% frequency and orientation 
 
% INPUT
    % params: a structure describing the image statistics (see function
    % body for default values)
    % nIms: number of images to generate
    % randseed: randomly seed the rng if desired 
    
% OUTPUT
    % images: a cell array {nIms x 1} containing your images (grayscale)
    
%% set default parameter values 
if isempty(params)
    params.freq_mean_cpd = 2;
    params.freq_sd_cpd = 0.1;
    params.orient_mean = 45;
    params.orient_sd = 1;
    params.size_pix = [140,140];
    params.ppd = 10; 
end
if isempty(nIms)
    nIms = 1;
end
if ~isempty(randseed)
    rng(randseed)
end

%% get set up
ppd = params.ppd;
size_pix = params.size_pix;
samp_rate_deg = ppd;   % samples per DEGREE here, instead of sec

% get frequency axis
nframes = size_pix(1);
nyq = .5*samp_rate_deg;         
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpd = samp_rate_deg/nframes;  
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpd:nyq-freq_step_cpd;

orient_mean = params.orient_mean;
orient_sd = params.orient_sd;
% if orient_sd<samp_rate_deg/size_pix(1)
%     warning('your orientation range may be too narrow')
% end
freq_mean_cpd = params.freq_mean_cpd;
freq_sd_cpd = params.freq_sd_cpd;
if freq_mean_cpd>nyq
    error('mean freq to filter needs to be < nyquist freq')
end

assert(orient_mean<180);
% if freq_sd_cpd<samp_rate_deg/size_pix(1);
%     warning('your spatial frequency range might be too narrow')
% end

%% get ready for filtering
% first, we're listing the frequencies corresponding to each point in
% the FFT representation, in grid form
[ x_freq, y_freq ] = meshgrid(fax,fax);
x_freq = x_freq(:);y_freq = y_freq(:);
% converting these into a polar format, where angle is orientation and
% magnitude is spatial frequency
[ang_grid,mag_grid] = cart2pol(x_freq,y_freq);
% adjust the angles so that they go from 0-180 (we don't care about
% distinguishing 45 vs 135 etc here)
ang_grid = mod(ang_grid,pi);
ang_grid = reshape(ang_grid,size_pix);
mag_grid = reshape(mag_grid,size_pix);

%% Make a gaussian SF filter
tar_mag = normpdf(mag_grid, freq_mean_cpd, freq_sd_cpd);

%% Make a gaussian orientation filter
% hack here to make sure it is circular
tar_ang = normpdf(mod(ang_grid-orient_mean*pi/180+pi/2, pi), pi/2, orient_sd*pi/180);

%% make sure the filter will work, fill in a pixel if not
filter = tar_mag.*tar_ang;
if ~any(filter(:)>0)
    error('your spatial frequency and/or orientation range is too narrow!')
%     [xval,yval] = pol2cart([orient_mean*pi/180; orient_mean*pi/180+pi], [freq_mean_cpd; freq_mean_cpd]);
%     [~,xcoord] = min(abs(fax-xval(1)));
%     [~,ycoord] = min(abs(fax-yval(1)));
%     filter(xcoord,ycoord) = 1;
%     [~,xcoord] = min(abs(fax-xval(2)));
%     [~,ycoord] = min(abs(fax-yval(2)));
%     filter(xcoord,ycoord) = 1;
end
% tar_ang = normpdf(ang_grid, orient_mean*pi/180, orient_sd*pi/180);
% tar_ang = reshape(circ_vmpdf(ang_grid*2, orient_mean*pi/180*2, orient_kappa),size_pix);
%% loop over images 
images = [];

for nn=1:nIms

    %% Start with random noise
    
    image = randn(size_pix);
    % % % FFT to convert to frequency domain
    [out] = fft2(image);
    image_fft = fftshift(out);

%     [image_ang, image_mag] = cart2pol(real(image_fft),imag(image_fft));
    %% Filter the image

    image_fft_both_filt = image_fft.*filter;
    image_filtered = ifft2(fftshift(image_fft_both_filt));

    images{nn} = real(image_filtered);

end

