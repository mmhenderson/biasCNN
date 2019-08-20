% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project

rndseed = 120482;
rng(rndseed)

% root_save = fullfile(pwd,'g/rating_ims_8/');
root_save = '/usr/local/serenceslab/maggie/biasCNN/grating_ims_8/';
if ~isdir(root_save)
    mkdir(root_save)
end

params.size_pix = [140,140];
params.ppd = 10; 
ppd = params.ppd;

MyGrey = 128;
Contrast = 0.5;
% what spatial frequencies do you want? these will each be in a separate
% folder.
my_freqs_cpd = [2.2];
params.freq_sd_cpd = 0.08;

% specify different amounts of noise
% (orientation variance is noise here)
noise_levels = [20];

% how many random images do you want to make? 
numInstances = 12;

%% create a gaussian blurred circular envelope

Smooth_size = round(1*ppd); %size of fspecial smoothing kernel
Smooth_sd = round(.5*ppd); %smoothing kernel sd
   
PatchSize = round(2*7*ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
OuterDonutRadius = (7*ppd)-(Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
InnerDonutRadius = 0;

% start with a meshgrid
X=-0.5*PatchSize+.5:1:.5*PatchSize-.5; Y=-0.5*PatchSize+.5:1:.5*PatchSize-.5;
[x,y] = meshgrid(X,Y);

% make a donut with gaussian blurred edge (except don't cut out the middle,
% it is more like a donut hole)
donut_out = x.^2 + y.^2 <= (OuterDonutRadius)^2;
donut_in = x.^2 + y.^2 >= (InnerDonutRadius)^2;
donut = donut_out.*donut_in;
donut = filter2(fspecial('gaussian', Smooth_size, Smooth_sd), donut);

%% make and save the individual images
for nn = 1:length(noise_levels)
    
    params.orient_sd = noise_levels(nn);
    
    for ff = 1:length(my_freqs_cpd)
        
        thisdir = sprintf('%sSF_%.2f_training/', root_save, my_freqs_cpd(ff));
        if ~isdir(thisdir)
            mkdir(thisdir)
        end
        
        params.freq_mean_cpd = my_freqs_cpd(ff);
        
        orient_vals = linspace(0,179,180);

        for oo=1:length(orient_vals)
           
            params.orient_mean = orient_vals(oo);

            %% make the images here
            images = get_filtered_noise(params,numInstances,rndseed);

            %% loop over and finish them
            
            for ii = 1:numInstances

                %% now multiply it by the donut (circle) to get gaussian envelope
                stim = images{ii}.*donut;
                
                % Scale the image correctly 
                lum_range_half = max(max(stim(:)),-min(stim(:)));
                stim_scaled = stim+lum_range_half;
                stim_scaled = stim_scaled./lum_range_half/2;
%                 stim_scaled = stim_scaled+(MyGrey/255 - stim_scaled(1,1));
%                 stim = max(0,min(255,MyGrey+MyGrey*(Contrast*stim)))/255;
                assert(stim_scaled(1,1)==0.5)
                fn2save = fullfile(thisdir,sprintf('FiltNoiseImage_%d_%ddeg.png',ii,orient_vals(oo)));
                
                imwrite(stim_scaled, fn2save)
                fprintf('saving to %s...\n', fn2save)
            end
        end

    end
end