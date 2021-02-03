function get_orientation_stats_imagenet_rots(root, numcores)
% Get the orientation content of modified versions of ImageNET database
% Making sure that rotations affected the prior in the
% expected way (peak at cardinals, or cardinals+rotation)
% save the result

% MMH March 2020
    %% set up paths/parameters

    if nargin==0
        % default params
        root = pwd;
        filesepinds = find(root==filesep);
        root = root(1:filesepinds(end-1));       
        numcores = 8;        
    end
    
    fprintf('numcores argument set to %s\n',numcores)
    rot_list = [0,22,45];

    image_path = fullfile(root,'images','ImageNet','ILSVRC2012');
    save_path = fullfile(root,'image_stats','ImageNet','ILSVRC2012');

    % how big are the images when we do our analysis? this is the same
    % as the size that VGG-16 preprocesing resizes them to, after crop.
    process_at_size = 224;

    % set an amount of downsampling, for speed of processing
    resize_factor = 1;  % if one then using actual size

    % which sets should we look at now? start from where we left off(last
    % time this process crashed, probably)
    n_done = zeros(length(rot_list),1);
    for rr=1:length(rot_list)
       n_done(rr) = length(dir(fullfile(save_path,sprintf('*%d*',rot_list(rr)),'*.mat'))); 
    end
    sets2do = [min(n_done)+1:1000];

    % set up parallel pool w 8 cores
    if strcmp(numcores,'max')
         maxcores = feature('numcores');
         numcores=maxcores;
    end
    fprintf('USING %D CORES FOR PROCESSING IMAGES\n',numcores)
    if isempty(gcp('nocreate'))
        parpool(numcores);
    end
    %% specify the spatial frequencies and orientations to filter at

    freq_list = logspace(log10(0.02), log10(.2),4);
    [wavelength_list,sorder] = sort(1./freq_list,'ascend');
    freq_list = freq_list(sorder);

    ori_list = 5:5:180;

    R_MEAN = 124;
    G_MEAN = 117;
    B_MEAN = 104;

    %% make the filter bank (will be same for all images we look at)

    fprintf('making filters...\n')
    tic

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

    for p = 1:length(GaborBank)

        H = makeFrequencyDomainTransferFunction_MMH(GaborBank(p),size_after_pad);
        filters_freq(:,:,p) = ifftshift(H);

    end
    toc

    %% find the names of all image sets (these are identical across rotations)
    % want to make sure we grab the same ones in the same order across all
    % rotations, so make the list now.
    set_folders = dir(fullfile(image_path, sprintf('train_rot_0'),'n*'));
    set_folders = {set_folders.name}; 

    %% loop over image sets first
    for ff=[sets2do]
%     for ff=[]

        fprintf('processing image set %d of %d\n',ff,length(set_folders));

        %% now loop over the three different training sets
        for rr = 1:length(rot_list)

            image_dir_this_rot = fullfile(image_path, sprintf('train_rot_%d',rot_list(rr)));

            folder2save = fullfile(save_path, sprintf('ImageStats_train_rot_%d',rot_list(rr)));
            if ~isfolder(folder2save)
                mkdir(folder2save)
            end

            fn2save = fullfile(folder2save,sprintf('%s_allstats_highdensity.mat',set_folders{ff}));  

            imlist = dir(fullfile(image_dir_this_rot, set_folders{ff}, '*.jpeg'));

            % make this params struct to pass into my function
            params.R_MEAN = R_MEAN;
            params.G_MEAN = G_MEAN;
            params.B_MEAN = B_MEAN;

            params.process_at_size = process_at_size;
            params.size_after_pad = size_after_pad;
            params.filters_freq = filters_freq;

            params.ori_list = ori_list;
            params.wavelength_list = wavelength_list;

            %% loop over images and process
            clear image_stats
            for ii = 1:length(imlist)

                im_file = fullfile(imlist(ii).folder, imlist(ii).name);
                fprintf('loading image %d of %d\n',ii,length(imlist));
                fprintf('    from %s\n',im_file);

                try
                    image = imread(im_file);
                catch err
                    fprintf('image %d could not be loaded!\n',ii)
                    continue
                end

                %% do the processing in a separate function
                out = process_image(image, params);
                % take out these bigger fields here because it will take up too
                % much memory to save, instead just saving average
                out = rmfield(out,'mag');
                out = rmfield(out,'phase');

                image_stats(ii) = out;

            end

            save(fn2save, 'image_stats');
            fprintf('saving to %s\n',fn2save);
        end

    end

end

