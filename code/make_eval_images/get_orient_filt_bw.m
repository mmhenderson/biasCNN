kappa_vals = logspace(log10(100),log10(1000),4);
kappa_vals = kappa_vals(1:4)';
nBW = length(kappa_vals);
orient_vals_deg = linspace(0,179,180);
nOri = numel(orient_vals_deg);

oo=85;
ang_grid = linspace(0,pi,5000);
ang_grid_deg = ang_grid/pi*180;

fwhm_vals = zeros(size(kappa_vals));

for bw =1:nBW
    
    orient_kappa_deg = kappa_vals(bw);
    tf = circ_vmpdf(ang_grid*2, orient_vals_deg(oo)*pi/180*2, orient_kappa_deg*pi/180*2);

    height = max(tf) - min(tf);
    [~,sorder] = sort(abs(tf-height/2),'ascend');
    half_inds = sorder(1:2);

    fwhm_deg = abs(ang_grid_deg(half_inds(2)) - ang_grid_deg(half_inds(1)));
   
    fwhm_vals(bw)  = fwhm_deg;
end
 
round(kappa_vals,0)
round([fwhm_vals],1)