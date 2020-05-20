
%% set parameters 

clear
close all

freq_vals_cpp = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25];
size_pix = [224,224];

orient_vals_deg = linspace(0,179,5);
rndseed = 123248;

figure;hold all;
nSF = length(freq_vals);
nOri = length(orient_vals_deg);
ii=0;
for ff=1:length(freq_vals)
    for oo=1:length(orient_vals_deg)

        params.freq_mean_cpp = freq_vals(ff);

        params.freq_sd_cpd = 10/140;

        params.orient_mean = 180-orient_vals_deg(oo);

        params.orient_kappa = 500;
        
        params.size_pix = [140,140];
        params.ppd = 10; 

        nIms = 1;

        images = get_filtered_noise(params,nIms,rndseed);

        diffs = [];

        ii=ii+1;
        subplot(nSF,nOri,ii);hold all;
        imagesc(images{1});

        axis equal off

        title(sprintf('frequency=%.2f+/-%.2f cpd\norient=%0.f,kappa=%0.f',freq_vals(ff),params.freq_sd_cpd,orient_vals_deg(oo),params.orient_kappa))

    
    end
end