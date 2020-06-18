function get_orientation_stats_FiltImageNetIms2(image_set, root, numcores)

    %% get the orientation content of a set of gaussian windowed gratings
    % ground truth for this image analysis
     %% set up paths/parameters

    if nargin==0
        % default params
        root = pwd;
        filesepinds = find(root==filesep);
        root = root(1:filesepinds(end-1));       
        numcores = 8;        
        
        image_set = 'FiltIms12AllSFCos_rand1';

    end
    
    fprintf('Filtering set %s\n',image_set);
    fprintf('numcores argument set to %s\n',numcores)

    image_path = fullfile(root, 'images','gratings',image_set);
    save_path = fullfile(root, 'image_stats','gratings',image_set);

    if ~isfolder(save_path)
        mkdir(save_path)
    end

    % list the ground truth spat freq for gratings
%     meas_sf_list = round([0.0125    0.0228    0.0414    0.0754    0.1373    0.2500],2);
    meas_sf_list = round([0.002],3);

    % how big are the images when we do our analysis? this is the same
    % as the size that VGG-16 preprocesing resizes them to, after crop.
    process_at_size = 224;

    % set an amount of downsampling, for speed of processing
    resize_factor = 1;  % if one then using actual size

    ori_vals_deg = linspace(0,179,180);
    nOri = numel(ori_vals_deg);
    nImsPerOri=48;
    %% specify the spatial frequencies and orientations to filter at

    [wavelength_list,sorder] = sort(1./meas_sf_list,'ascend');
    meas_sf_list = meas_sf_list(sorder);

    meas_ori_list = 5:5:180;

    R_MEAN = 124;
    G_MEAN = 117;
    B_MEAN = 104;

    %% make the filter bank (will be same for all images we look at)

    fprintf('making filters...\n')
    tic

    GaborBank = gabor(wavelength_list.*resize_factor,meas_ori_list);

    sizeLargestKernel = size(GaborBank(end).SpatialKernel);
    % Gabor always returns odd length kernels
    padding_needed = (sizeLargestKernel-1)/2;

    max_pix = process_at_size;  % maximum size of any image dimension
    % FIX this so that we can make the filters ahead of time
    size_after_pad = max_pix*resize_factor+padding_needed*2;
    size_after_pad = size_after_pad + mod(size_after_pad,2);
    % making a matrix [nPix x nPix x nFilters]
    filters_freq = zeros([size_after_pad,length(GaborBank)]);

    for p = 1:length(GaborBank)

        H = makeFrequencyDomainTransferFunction_MMH(GaborBank(p),size_after_pad);
        filters_freq(:,:,p) = ifftshift(H);

    end
    toc

    % make this params struct to pass into my filtering function
    params.R_MEAN = R_MEAN;
    params.G_MEAN = G_MEAN;
    params.B_MEAN = B_MEAN;

    params.process_at_size = process_at_size;
    params.size_after_pad = size_after_pad;
    params.filters_freq = filters_freq;

    params.ori_list = meas_ori_list;
    params.wavelength_list = wavelength_list;
    %% set up parallel pool
%     if strcmp(numcores,'max')
%          maxcores = feature('numcores');
%          numcores=maxcores;
%     end
%     fprintf('USING %d CORES FOR PROCESSING IMAGES\n',numcores)
%     if isempty(gcp('nocreate'))
%         parpool(numcores);
%     end
    
    %% define more paths

    image_folder = fullfile(image_path, 'AllIms');

    fn2save = fullfile(save_path, sprintf('AllIms_allstats_highdensity2.mat'));
    clear image_stats
    %% loop over images
   
    for oo=1:nOri
        
%         fn2save = fullfile(save_path, sprintf('AllIms_%ddeg_allstats_highdensity.mat',ori_vals_deg(oo)));
%         clear image_stats
        orient=ori_vals_deg(oo);
        
        for ii = 1:nImsPerOri
  
        
            fn2load = fullfile(image_folder,sprintf('FiltImage_ex%d_%ddeg.png',ii,orient));

            %% loop over images and process

            fprintf('loading from %s\n',fn2load)
            try
                image = imread(fn2load);
            catch err
                fprintf('image %d could not be loaded!\n',ii)
                continue
            end       

            %% do the processing in a separate function
            out = process_image(image, params);
            out.true_ori = orient;
            out.true_sf = NaN;
            out.mag = [];
            out.phase = [];

            image_stats(ii,oo) = out;
        end
        
        
    end
    save(fn2save, 'image_stats');
    fprintf('saving to %s\n',fn2save);
        
    
    
end