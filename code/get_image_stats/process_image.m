function image_stats = process_image(image, params)
% analyze the orientation content of an image, using a predefined set of
% filters and preprocessing parameters  
% MMH 11/19/2019
    %% extract the parameters of interest
     
    R_MEAN = params.R_MEAN;
    G_MEAN = params.G_MEAN;
    B_MEAN = params.B_MEAN;

    process_at_size = params.process_at_size;
    size_after_pad = params.size_after_pad;
    filters_freq = params.filters_freq;

    %% Preprocessing
    
    % subtract the background color (this is the color at a corner
    % pixel), so edges go to zero
    % at this point - all image values are in the range 0-255. When we
    % subtract the mean, some values will go negative. 
    %   IMPORTANT: CONVERT THE IMAGE TO FLOAT FIRST, NOT INTEGER -
    %   OTHERWISE THE NEGATIVE VALUES WILL ALL BE STUCK AT ZERO
    image = double(image) - permute(repmat([R_MEAN;G_MEAN;B_MEAN],1, size(image,1),size(image,2)), [2,3,1]);
    
    % make it grayscale
    if size(image,3)==3
        grey_image = round(mean(image,3),0);
    end

    image = grey_image;

    orig_size = size(image);

    % no need to resize, but make sure it's the size we expect
    assert(orig_size(1)==process_at_size && orig_size(2)==process_at_size)
    if image(1,1)~=0
        warning('corner pixel value is not zero (%d) after background subtraction', image(1,1))
    end
    
    % pad it so we can apply the filters at the correct size
    pad_by = (size_after_pad - size(image))./2;        
    n2pad = [floor(pad_by'), ceil(pad_by')];        

    % Zero-pad the image for filtering
    image_padded = [repmat(zeros(size(image(:,1))), 1, n2pad(2,1)), image, repmat(zeros(size(image(:,end))), 1, n2pad(2,2))];
    image_padded = [repmat(zeros(size(image_padded(1,:))), n2pad(1,1), 1); image_padded; repmat(zeros(size(image_padded(end,:))), n2pad(1,2),1)];

    padded_size = size(image_padded);
    assert(all(padded_size==size_after_pad));

    %% Filtering

    % fft into frequency domain
    image_fft = fft2(image_padded);

    % Apply all my filters all at once
    filtered_freq_domain = image_fft.*filters_freq;

    % get back to the spatial domain
    out_full = ifft2(filtered_freq_domain);

    % un-pad the image (back to its down-sampled size)
    out = out_full(n2pad(1,1)+1:n2pad(1,1)+process_at_size, n2pad(2,1)+1:n2pad(2,1)+process_at_size,:);
    assert(size(out,1)==process_at_size && size(out,2)==process_at_size);

    mag = abs(out);
    phase = angle(out);

    %%  add all this info to my structure
    
    image_stats.phase = phase;
    image_stats.mag = mag;
    image_stats.mean_phase = squeeze(mean(mean(phase,2),1));
    image_stats.mean_mag = squeeze(mean(mean(mag,2),1));
    image_stats.ori_list = params.ori_list;
    image_stats.wavelength_list = params.wavelength_list;
    image_stats.orig_size = orig_size;
    image_stats.padded_size = padded_size;

