% make some filtered noise images with varying spatial frequency and orientation content

%% Set up parameters here

clear
close all;

nIms = 1;

% freq in cycles per degree 
freq_mean_cpd = 1;
freq_sd_cpd = 2;

orient_mean = 15;
orient_sd = 1;

size_pix = [224,224];

%% more useful parameters here, relating spatial frequency from degrees to pixels

ppd = 10;
samp_rate_deg = ppd;   % samples per DEGREE here, instead of sec
size_deg = size_pix/ppd;
d_deg = 1/samp_rate_deg;

mean_period_pixels = ppd/freq_mean_cpd;
% spat_axis_deg = d_deg:d_deg:size_deg(1);
% spat_axis_pix = spat_axis_deg*ppd;

%% loop over images 

for nn=1:nIms

    image = randn(size_pix);
    % % % FFT to convert to frequency domain
    [out] = fft2(image);
    image_fft = fftshift(out);

    % get frequency axis
    nframes = size_pix(1);
    nyq = .5*samp_rate_deg;         
    if freq_mean_cpd>nyq
        error('mean freq to filter needs to be < nyquist freq')
    end
    % step size (or resolution) in the frequency domain - depends on sampling rate and length of the signal
    freq_step_cpd = samp_rate_deg/nframes;  
    % x-axis in freq domain, after fftshift
    fax = -nyq:freq_step_cpd:nyq-freq_step_cpd;
    
    %% Plot my FFT image in frequency domain
    figure;
    subplot(3,3,1);hold all;
    imagesc(image);axis equal off
    title('Gaussian white noise')
    subplot(3,3,2);hold all;
    imagesc(abs(image_fft)); axis equal off
    title('Noise in frequency domain')

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
    ang_grid = reshape(ang_grid,size(image));
    mag_grid = reshape(mag_grid,size(image));
    %% Make a gaussian SF filter
    tar_mag = normpdf(mag_grid, freq_mean_cpd, freq_sd_cpd);
    subplot(3,3,3);hold all;
    imagesc(tar_mag);
    axis equal off
    title(sprintf('Spatial frequency filter\n(%.2fcpd+/-%.2fcpd)',freq_mean_cpd,freq_sd_cpd));

    %% Filter the image
    image_fft_sf_filt = image_fft.*tar_mag;
  
    %%
    subplot(3,3,4);hold all;
    imagesc(real(image_fft_sf_filt));
    axis equal off
    title('Freq domain, after SF filt')
    
    subplot(3,3,5);hold all;
    imagesc(real(ifft2(fftshift(image_fft_sf_filt))));
    axis equal off
    title('Image, after SF filt')

    %% Make a gaussian orientation filter
    tar_ang = normpdf(ang_grid, orient_mean*pi/180, orient_sd*pi/180);
    subplot(3,3,6);hold all;
    imagesc(tar_ang);
    axis equal off
    title(sprintf('Orientation filter\n(%.2fdeg+/-%.2fdeg)',orient_mean,orient_sd));

    %% Filter the image
    image_fft_ang_filt = image_fft.*tar_ang;
  
    %%
    subplot(3,3,7);hold all;
    imagesc(real(image_fft_ang_filt));
    axis equal off
    title('Freq domain, after orient filt')
    
    subplot(3,3,8);hold all;
    imagesc(real(ifft2(fftshift(image_fft_ang_filt))));
    axis equal off
    title('Image, after orient filt')
    %% Finally, apply both filters
    
    image_fft_both_filt = image_fft.*tar_ang.*tar_mag;
    image_filtered = ifft2(fftshift(image_fft_both_filt));

    subplot(3,3,9);hold all
    imagesc(real(image_filtered));axis equal off
    title('Image, after both filters');

    image = real(image_filtered);
    figure;hold all;
    imshow(image_filtered);
end
