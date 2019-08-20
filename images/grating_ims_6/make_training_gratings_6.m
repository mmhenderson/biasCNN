% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project

rndseed = 786867;
rng(rndseed)

root_save = '/usr/local/serenceslab/maggie/biasCNN/grating_ims_6/';
if ~isdir(root_save)
    mkdir(root_save)
end

p.ppd = 10;

% what spatial frequencies do you want? these will each be in a separate
% folder.
my_freqs_cpd = [2.2];
my_freqs_cpp = my_freqs_cpd/p.ppd; 
my_periods_ppc = 1./my_freqs_cpp;

% specify different amounts of noise
noise_levels = [0];

% how many random phases do you want to make? 4 
numInstances = 12;
%% specify parameters that are the same for all grating images

CenterX = 0;
CenterY = 0;

black = [0,0,0];
white = [1,1,1];
p.MyGrey = 128;

p.Smooth_size = round(1*p.ppd); %size of fspecial smoothing kernel
p.Smooth_sd = round(.5*p.ppd); %smoothing kernel sd
   
p.PatchSize = round(2*7*p.ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
p.OuterDonutRadius = (7*p.ppd)-(p.Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
% p.InnerDonutRadius = (.4*p.ppd)+(p.Smooth_size/2); %Size of donut insides, automatically defined in pixels.
p.InnerDonutRadius = 0;
% p.OuterFixRadius = .1*p.ppd; %outer dot radius (in pixels)
% p.InnerFixRadius = 0; %set to zero if you a donut-hater
% p.FixColor = black;
% p.ResponseLineWidth = .075*p.ppd; %in pixel
% p.lineSmoothSize = p.ResponseLineWidth*2;
% p.lineSmoothSD = p.ResponseLineWidth;
% p.ResponseLineColor = white;
% p.Freq = 2; %Frequency of the grating in cycles per degree:
p.Contrast = 0.50; %Make contast parameter between 0 and 1. This will be a ratio of (maxlum - meanlum)/meanlum

MyPatch = [CenterX-p.PatchSize/2 CenterY-p.PatchSize/2 CenterX+p.PatchSize/2 CenterY+p.PatchSize/2];

% start with a meshgrid
X=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5; Y=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5;
[x,y] = meshgrid(X,Y);

% make a donut with gaussian blurred edge (except don't cut out the middle,
% it is more like a donut hole)
donut_out = x.^2 + y.^2 <= (p.OuterDonutRadius)^2;
donut_in = x.^2 + y.^2 >= (p.InnerDonutRadius)^2;
donut = donut_out.*donut_in;
donut = filter2(fspecial('gaussian', p.Smooth_size, p.Smooth_sd), donut);

%% make and save the individual images
for nn = 1:length(noise_levels)
    for ff = 1:length(my_freqs_cpd)

        thisdir = sprintf('%sSF_%.2f_training/', root_save, my_freqs_cpd(ff));
        if ~isdir(thisdir)
            mkdir(thisdir)
        end
        p.Freq = my_freqs_cpd(ff);

        p.TargOrient = linspace(0,179,180);

        for tt=1:length(p.TargOrient)
%             p.PhaseJitter = randsample(0:179,numInstances)*(pi/180);
            p.PhaseJitter = zeros(numInstances,1);
            for pp = 1:numInstances

                %% make the full field grating
                sine = (sin(p.Freq/p.ppd*2*pi*(y.*sin(p.TargOrient(tt)*pi/180)+x.*cos(p.TargOrient(tt)*pi/180))-p.PhaseJitter(pp)));

                sine = sine + randn(size(sine))*noise_levels(nn);
                
                sine = sine./max(sine(:));
                
%                 %Give the grating the right contrast level and scale it
%                 image1 = max(0,min(255,p.MyGrey+p.MyGrey*(p.Contrast* sine)))/255;
% 
%                 fn2save = [thisdir, 'Fullfield_randphase' num2str(pp) '_' sprintf('%d', p.TargOrient(tt)) 'deg.png'];
%                 imwrite(image1, fn2save)
%                 fprintf('saving to %s\n', fn2save)
                %% now multiply it by the donut (circle) to get gaussian envelope
                stim = sine.*donut;

                %Give the grating the right contrast level and scale it
                image2 = max(0,min(255,p.MyGrey+p.MyGrey*(p.Contrast* stim)))/255;

                fn2save = [thisdir, 'Gaussian_fixedphase_' num2str(pp) '_' sprintf('%d', p.TargOrient(tt)) 'deg.png'];
                imwrite(image2, fn2save)
                fprintf('saving to %s...\n', fn2save)
            end
        end

    end
end