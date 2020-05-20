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
    params.freq_mean_cpp = 0.2;
    params.freq_sd_cpp = 0.1;
    params.orient_mean_deg = 45;
    params.orient_kappa_deg = 20;
    params.size_pix = [224,224];
%     params.ppd = 10; 
end
if isempty(nIms)
    nIms = 1;
end
if ~isempty(randseed)
    rng(randseed)
end

%% define sampling rate and frequency axis
size_pix = params.size_pix;
samp_rate_pix = 1;   % samples per pixel, always one.

% get frequency axis
nframes = size_pix(1);
nyq = .5*samp_rate_pix;         
% step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
freq_step_cpp = samp_rate_pix/nframes;  
% x-axis in freq domain, after fftshift
fax = -nyq:freq_step_cpp:nyq-freq_step_cpp;

orient_mean_deg = params.orient_mean_deg;
orient_kappa_deg = params.orient_kappa_deg;
freq_mean_cpp = params.freq_mean_cpp;
freq_sd_cpp = params.freq_sd_cpp;
if freq_mean_cpp>=nyq
    error('mean frequency to filter needs to be < nyquist freq')
end

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
tar_mag = normpdf(mag_grid, freq_mean_cpp, freq_sd_cpp);

%% Make a gaussian orientation filter

% tar_ang = normpdf(ang_grid, orient_mean*pi/180, orient_sd*pi/180);
tar_ang = reshape(circ_vmpdf(ang_grid*2, orient_mean_deg*pi/180*2, orient_kappa_deg*pi/180*2),size_pix);
%% loop over images 
images = [];

for nn=1:nIms

    %% Start with random noise
    
    image = randn(size_pix);
    % % % FFT to convert to frequency domain
    [out] = fft2(image);
    image_fft = fftshift(out);

    %% Filter the image

    image_fft_both_filt = image_fft.*tar_ang;
%     image_fft_both_filt = image_fft.*tar_ang.*tar_mag;
    image_filtered = ifft2(fftshift(image_fft_both_filt));

    images{nn} = real(image_filtered);

end

