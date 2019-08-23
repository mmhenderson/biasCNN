% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project
clear
close all

rndseed = 678671;
rng(rndseed)

root_save = '/usr/local/serenceslab/maggie/biasCNN/images/grating_ims_10a/';
if ~isdir(root_save)
    mkdir(root_save)
end

ppd = 10;

% what spatial frequencies do you want? these will each be in a separate
% folder.
freq_levels_cpd = logspace(log10(0.2),log10(4),6);

% specify different amounts of noise
noise_levels = [0.01];

% specify different contrast levels
contrast_levels = [0.8];

% how many random instances do you want to make?
numInstances = 8;

%% specify parameters that are the same for all grating images

Smooth_size = round(1*ppd); %size of fspecial smoothing kernel
Smooth_sd = round(.5*ppd); %smoothing kernel sd
   
PatchSize = round(2*7*ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
OuterDonutRadius = (7*ppd)-(Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
InnerDonutRadius = 0;

% make a donut with gaussian blurred edge (except don't cut out the middle,
% it is more like a donut hole)
% start with a meshgrid
X=-0.5*PatchSize+.5:1:.5*PatchSize-.5; Y=-0.5*PatchSize+.5:1:.5*PatchSize-.5;
[x,y] = meshgrid(X,Y);
donut_out = x.^2 + y.^2 <= (OuterDonutRadius)^2;
donut_in = x.^2 + y.^2 >= (InnerDonutRadius)^2;
donut = donut_out.*donut_in;
donut = filter2(fspecial('gaussian', Smooth_size, Smooth_sd), donut);

%% make and save the individual images
nn=1;

for cc=1:length(contrast_levels)

    for ff = 1:length(freq_levels_cpd)

        thisdir = sprintf('%sSF_%.2f_Contrast_%.2f/', root_save, freq_levels_cpd(ff), contrast_levels(cc));
        if ~isdir(thisdir)
            mkdir(thisdir)
        end

        this_freq_cpp = freq_levels_cpd(ff)/ppd;

        orient_vals = linspace(0,179,180);

        for oo=1:length(orient_vals)

            phase_vals = datasample(0:359,1)*ones(numInstances,1)*pi/180;
%             phase_vals = zeros(numInstances,1);

            for pp = 1:numInstances

                %% make the full field grating
                % range is [-1,1] to start
                sine = (sin(this_freq_cpp*2*pi*(y.*sin(orient_vals(oo)*pi/180)+x.*cos(orient_vals(oo)*pi/180))-phase_vals(pp)));

                % make the values range from 1 +/-noise to
                % -1 +/-noise
                sine = sine+ randn(size(sine))*noise_levels(nn);

                % now scale it down (note the noise also gets scaled)
                sine = sine*contrast_levels(cc);
                
                % shouldnt ever go outside the range [-1,1] so values won't
                % get cut off (requires that noise is low if contrast is
                % high)
                assert(max(sine(:))<=1 && min(sine(:))>=-1)

                %% now multiply it by the donut (circle) to get gaussian envelope
                stim = sine.*donut;

                % now reset the range from [-1, 1] to [0, 1].
                % note the actual values don't have to span that whole
                % range, but they will be centered at 0.5.
                stim_scaled = (stim+1)./2;

                assert(stim_scaled(1,1)==0.5)

                fn2save = fullfile(thisdir,sprintf('Gaussian_fixedphase_%d_%ddeg.png',pp,orient_vals(oo)));
                
                imwrite(stim_scaled, fn2save)
                fprintf('saving to %s...\n', fn2save)
            end
        end
    end
end