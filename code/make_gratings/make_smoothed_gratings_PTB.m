% make a bunch of gratings at different orientations, save as images in my
% folder under biasCNN project
% using psychtoolbox to try and anti-alias the images.
% this script will launch a psychtoolbox window, make a bunch of images
% that it flips really rapidly, and save each as an image using
% Screen('GetImage'). Is there a way to do this without launching window,
% maybe?
%%
clear
close all hidden

rndseed = 837987;
rng(rndseed)

% find my main root directory
root = pwd;
filesepinds = find(root==filesep);
root = root(1:filesepinds(end-2));

window = 'cos';
% window = 'square';
% window = 'circ';
image_set = 'CosGratings_Smooth_Big';
% image_set = 'SquareGratings_Smooth_Big';
scale_by = 2;

% define where to save the newly created images
image_save_path = fullfile(root,'biasCNN/images/gratings/',image_set);
if ~isdir(image_save_path)
    mkdir(image_save_path)
end

% this is the height and width of the final images
image_size = 224*scale_by;

%% set up mask for windowing 
if strcmp(window,'circ') 
    % load this circular smoothed mask
    mask_file = fullfile(root,'biasCNN/code/image_proc_code/Smoothed_mask.png');
    % this is a mask of range 0-255 - use this to window the image
    mask_image = imread(mask_file);     
    mask_image = repmat(mask_image,1,1,3);
    mask_image = double(mask_image)./255; % change to 0-1 range
    
elseif strcmp(window,'cos')
    % making a circular mask with cosine fading to background
    cos_mask = zeros(image_size);
    values = image_size./2*linspace(-1,1,image_size);
    [gridx,gridy] = meshgrid(values,values);
    r = sqrt(gridx.^2+gridy.^2);
    % creating three ring sections based on distance from center
    outer_range = 100*scale_by;
    inner_range = 50*scale_by;
    % inner values: set to 1
    cos_mask(r<inner_range) = 1;
    % middle values: create a smooth fade
    faded_inds = r>=inner_range & r<outer_range;
    cos_mask(faded_inds) = 0.5*cos(pi/(outer_range-inner_range).*(r(faded_inds)-inner_range)) + 0.5;
    % outer values: set to 0
    cos_mask(r>=outer_range) = 0;
    
    mask_image = repmat(cos_mask,1,1,3);
end

if ~strcmp(window,'square')
    % also want to change the background color from 0 (black) to a mid gray color 
    % (mean of each color channel). These values match vgg_preprocessing_biasCNN.py, 
    % will be subtracted when the images are centered during preproc.
    R_MEAN = 124;
    G_MEAN = 117;
    B_MEAN = 104;
    mask_to_add = cat(3, R_MEAN*ones(image_size,image_size,1),G_MEAN*ones(image_size,image_size,1),B_MEAN*ones(image_size,image_size,1));
    mask_to_add = mask_to_add.*(1-mask_image);
end

%% enter parameters here

% what spatial frequencies do you want? these will each be in a separate
% folder. Units are cycles per pixel.
freq_levels_cpp_orig = logspace(log10(0.02),log10(0.4),6);
% adjusting these so that they'll be directly comparable with an older
% version of the experiment (in which we had smaller 140x140 images)
freq_levels_cycles_per_image = freq_levels_cpp_orig*140;

% these are the actual cycles-per-pixel that we want, so that we end up
% with the same number of cycles per image as we had in the older version.
freq_levels_cpp = freq_levels_cycles_per_image/image_size;

% specify different amounts of noise
noise_levels = [0.01];

% specify different contrast levels
contrast_levels = [0.8];

% how many random instances do you want to make?
numInstances = 1;

% two opposite phases
phase_levels = [0,180];

% start with a meshgrid
X=-0.5*image_size+.5:1:.5*image_size-.5; Y=-0.5*image_size+.5:1:.5*image_size-.5;
[x,y] = meshgrid(X,Y);

%% psychtoolbox stuff
try

    KbName('UnifyKeyNames')
    %use number pad - change this for scanner 
    p.keys=[KbName('b'),KbName('y')];
    p.escape = KbName('escape');
    % p.space = KbName('space');
    % p.start = KbName('t');
    Screens = Screen('Screens'); %look at available screens
    ScreenNumber = Screens(1); %pick first screen 
%     ScreenSizePixels = Screen('Rect', ScreenNumber);
%     CenterXPix = ScreenSizePixels(3)/2;
%     CenterYPix = ScreenSizePixels(4)/2;
    MyGrey = 128;
    AssertOpenGL;
    PsychJavaTrouble;
    small_rect = [0,0,500,500];
    CenterXPix = small_rect(3)/2;
    CenterYPix = small_rect(4)/2;
    % set multisample to 1 here
    multisample=1;
    numbuffers=2;
    [w] = Screen('OpenWindow',ScreenNumber, MyGrey, small_rect, [],numbuffers,[],multisample);
    white=WhiteIndex(ScreenNumber);
    black=BlackIndex(ScreenNumber);
    HideCursor;
    FlushEvents('keyDown');
    OriginalCLUT = [];
    ListenChar(2)


    PatchSizePix = image_size;
    GratingRect = [CenterXPix-PatchSizePix/2 CenterYPix-PatchSizePix/2 CenterXPix+PatchSizePix/2 CenterYPix+PatchSizePix/2];

    %% make and save the individual images
    nn=1;

    for cc=1:length(contrast_levels)

        for ff = 1:length(freq_levels_cpp)

            thisdir = sprintf('%s/SF_%.2f_Contrast_%.2f/', image_save_path, freq_levels_cpp(ff)*scale_by, contrast_levels(cc));
            if ~isdir(thisdir)
                mkdir(thisdir)
            end

            this_freq_cpp = freq_levels_cpp(ff);

            orient_vals = linspace(0,179,180);

            for oo=1:length(orient_vals)

                for pp=1:length(phase_levels)

                    phase_vals = ones(numInstances,1)*phase_levels(pp)*pi/180;

                    for ii = 1:numInstances

                        %% make the full field grating
                        % range is [-1,1] to start
                        sine = (sin(this_freq_cpp*2*pi*(y.*sin(orient_vals(oo)*pi/180)+x.*cos(orient_vals(oo)*pi/180))-phase_vals(ii)));

                        % make the values range from 1 +/-noise to
                        % -1 +/-noise
                        sine = sine+ randn(size(sine))*noise_levels(nn);

                        % now scale it down (note the noise also gets scaled)
                        sine = sine*contrast_levels(cc);

                        % shouldnt ever go outside the range [-1,1] so values won't
                        % get cut off (requires that noise is low if contrast is
                        % high)
                        assert(max(sine(:))<=1 && min(sine(:))>=-1)

                        % change the scale from [-1, 1] to [0,1]
                        % the center is exactly 0.5 - note the values may not
                        % span the entire range [0,1] but will be centered at
                        % 0.5.
                        stim_scaled = (sine+1)./2;

                        % convert from [0,1] to [0,255]
                        stim_scaled = stim_scaled.*255;


                        if ~strcmp(window,'square')
                            % now multiply it by the donut (circle) to get gaussian envelope
                            stim_masked = stim_scaled.*mask_image(:,:,1);
                            % finally add a mid-gray background color.
                            stim_masked_adj = uint8(stim_masked + mask_to_add(:,:,1));

                            assert(all(squeeze(stim_masked_adj(1,1,1))==[R_MEAN]))
                        else
                            stim_masked_adj = uint8(stim_scaled);
                        end
                        [resp, timeStamp] = checkForResp([p.keys],p.escape);
                        if resp==-1; escaperesponse(OriginalCLUT); end
                        
                        GratingTexture = Screen('MakeTexture', w, stim_masked_adj);

                        Screen('BlendFunction', w, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
                        Screen('DrawTexture',w,GratingTexture,[],GratingRect,0,0);
                        Screen('DrawingFinished', w);
                        Screen('Flip', w);
                        current_time = GetSecs;
                        tic
                        imageArray=Screen('GetImage', w, GratingRect, [], 1, 3);

                        
                        fn2save = fullfile(thisdir,sprintf('Gaussian_phase%d_ex%d_%ddeg.png',phase_levels(pp),ii,orient_vals(oo)));

                        
                        fprintf('saving to %s...\n', fn2save)
                        imwrite(imageArray, fn2save)
                        toc
%                         while GetSecs<current_time+0.05
                        [resp, timeStamp] = checkForResp([p.keys],p.escape);
                        if resp==-1; escaperesponse(OriginalCLUT); end
%                         end
                        
                    end
                end
            end
        end
    end


    Screen('CloseAll');
    clear screen
    ListenChar(1);
    ShowCursor;
    
catch err
     if exist('OriginalCLUT','var') && ~isempty(OriginalCLUT)
        if exist('ScreenNumber','var')
            Screen('LoadCLUT', ScreenNumber, OriginalCLUT);
        else
            Screen('LoadCLUT', 0, OriginalCLUT);
        end
    end
    Screen('CloseAll');                
    ShowCursor;
    if IsWin
        ShowHideWinTaskbarMex;     
    end
    ListenChar(1)
    rethrow(err)
    
end