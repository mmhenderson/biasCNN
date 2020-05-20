 function [H,freq,sf_axis] = makeFrequencyDomainTransferFunction_MMH2(Gabor,imageSize)
           
% Copied out of:
% /mnt/neurocube/local/MATLAB/R2018b/toolbox/images/images/gabor.m  % gabor method
% by MMH 9/6/2019

    % Directly construct frequency domain transfer function of
    % Gabor filter. (Jain, Farrokhnia, "Unsupervised Texture
    % Segmentation Using Gabor Filters", 1999)
    M = imageSize(1);
    N = imageSize(2);
    
    if mod(N,2)
        u = linspace(-0.5+1/(2*N),0.5-1/(2*N),N);
    else
        u = linspace(-0.5,0.5-1/N,N); 
    end
    
    if mod(M,2)
        v = linspace(-0.5+1/(2*M),0.5-1/(2*M),M);
    else
        v = linspace(-0.5,0.5-1/M,M); 
    end
    sf_axis=v;
    
    [U,V] = meshgrid(u,v);

%     orient=15
%     Uprime = U .*cosd(orient) - V .*sind(orient);
%     Vprime = U .*sind(orient) + V .*cosd(orient);

    Uprime = U .*cosd(Gabor.Orientation) - V .*sind(Gabor.Orientation);
    Vprime = U .*sind(Gabor.Orientation) + V .*cosd(Gabor.Orientation);

    % From relationship in "Nonlinear Operator in Oriented Texture", Kruizinga,
    % Petkov, 1999.
    BW = Gabor.SpatialFrequencyBandwidth;
    SigmaX = Gabor.Wavelength/pi*sqrt(log(2)/2)*(2^BW+1)/(2^BW-1);
    SigmaY = SigmaX ./ Gabor.SpatialAspectRatio;
    
    sigmau = 1/(2*pi*SigmaX);
    sigmav = 1/(2*pi*SigmaY);
    freq = 1/Gabor.Wavelength;

    A = 2*pi*SigmaX*SigmaY;

    H = A.*exp(-0.5*( ((Uprime-freq).^2)./sigmau^2 + Vprime.^2 ./ sigmav^2) );
        
end