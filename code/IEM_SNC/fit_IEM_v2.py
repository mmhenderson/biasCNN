import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool


n_chan = 9
ang_s = 180
xx = np.linspace(0,179,180)
cosd = lambda x : np.cos( np.deg2rad(x) )
make_basis_function = lambda xx,mu : np.power( (cosd(xx-mu) ), ( n_chan - (n_chan % 2) ) )

def anova1(dat,grps):
    n_roi = np.shape(dat)[0]
    f_stats = np.zeros(n_roi)
    for r in range(n_roi):
        group_data = []
        for i in np.unique(grps):
            group_data.append(dat[r,grps==i])
        f_stats[r],_ = stats.f_oneway(*group_data)
    return f_stats

def gen_basis_set(shift=1):
    ang = np.arange(ang_s)
    chan_center = np.linspace(ang_s/n_chan, ang_s, n_chan,dtype=int) - shift
    basis_set = np.zeros((ang_s,n_chan))
    for c in np.arange(n_chan):
        basis_set[:,c] = make_basis_function(xx,chan_center[c])
    chan_center_mask = np.zeros(ang_s)==1
    chan_center_mask[chan_center]=1
    return basis_set, chan_center_mask

def gen_design_matrix(ang):
    assert all(ang<ang_s),'Invalid Angles!'
    n_trials = len(ang)
    design_matrix  = np.zeros((len(ang),ang_s))
    for i in range(n_trials):
        this_angle = ang[i]
        design_matrix[i,this_angle] = 1
    return design_matrix

def pred_IEM(X,X0,Y):
    w = np.linalg.solve(Y.T@Y,Y.T)@X.T
    Y_ = (np.linalg.solve(w@w.T,w)@X0).T
    return Y_
def fit_IEM(X,Y):
    w = np.linalg.solve(Y.T@Y,Y.T)@X.T
    w_ = np.linalg.solve(w@w.T,w)
    return w_

def fit_IEM_slide(X,ang):
    ## ##
    design_matrix = gen_design_matrix(ang)
    n_vox = np.shape(X)[0]
    w=np.zeros((180,n_vox))
    for shift in range(1,21):
        basis_set, chan_center_mask = gen_basis_set(shift=shift)
        Y = design_matrix@basis_set
        this_w = fit_IEM(X,Y)
        w[chan_center_mask,:] = this_w
    return w

## voxel selection
def get_donut_mask(this_dat,EV,donut_thresh):
    dat1 = this_dat[:,EV.d1==1] # donut
    dat2 = this_dat[:,EV.d1==2] # hole
    t_scores,_ = stats.ttest_ind(dat1.T,dat2.T)
    cuttoff = np.percentile(t_scores,donut_thresh)
    roi_use_donut = t_scores>cuttoff
    return roi_use_donut
def get_anova_mask(this_dat,angles_use,anova_thresh):
    angles_anova = np.round((angles_use-10)/2,-1)
    f_stats= anova1(this_dat,angles_anova)
    cuttoff = np.percentile(f_stats,anova_thresh)
    roi_use_anova = f_stats>cuttoff
    return roi_use_anova

def center_resp(chan_resp,center_ang):
    centered_chan_resp = np.zeros(np.shape(chan_resp))
    n_trials_use,n_ang = np.shape(chan_resp)
    if (np.isscalar(center_ang)) or (len(center_ang) == 1):
        center_ang = np.ones(n_trials_use,dtype=int)*center_ang
    cent = round(n_ang/2)
    for i in range(n_trials_use):
        centered_chan_resp[i,:] = np.roll(chan_resp[i,:],cent-center_ang[i])
    return centered_chan_resp


# cross val localizer data
def fit_by_roi(DAT,EV=None,donut_thresh=0,anova_thresh=0,center=True,G=None):
    # DAT - voxel data by ROI
    
    ang = EV.d0.values
    if G is None:
        G = EV.block.values
    ang[ang==180]=0
    recon_all = dict()
    for this_reg in DAT.index:
        print(this_reg,end=' ')
        this_dat = DAT[this_reg]
        recon_all[this_reg] = fit_by_run(this_dat,ang,G=G,center=center,EV=EV,
                                         donut_thresh=donut_thresh,anova_thresh=anova_thresh)
    OUT = pd.Series(recon_all)
    return OUT

def fit_by_run(X,ang,G,center=True,donut_thresh=0,anova_thresh=0,EV=None): #

    if donut_thresh:
        roi_use_donut = get_donut_mask(X,EV,donut_thresh)
        X = X[roi_use_donut,:]
        
    ind_donut = EV.d1==1  
    EVU = EV[ind_donut]
    X = X[:,ind_donut]
    G=G[ind_donut]
    ang = ang[ind_donut]
    
    grps = np.unique(G)
    n_trials = len(ang)
    chan_resp = np.zeros((n_trials,ang_s))
    design_matrix = gen_design_matrix(ang)
    OUT = np.zeros((n_trials,180))
    #- do donut threshold outside... not angle selective anyhow
    
    if anova_thresh:
        roi_use_anova = get_anova_mask(X,ang,anova_thresh)
        X = X[roi_use_anova,:]
    
    for grp in grps:
        trn_index = np.ones(n_trials)==1
        trn_index[G==grp] = 0
        tst_index = ~trn_index
        
        this_X_train = X[:,trn_index]
        this_X_test = X[:,tst_index]
        angles_use=ang[trn_index]

#         if anova_thresh:
#             roi_use_anova = get_anova_mask(this_X_train,angles_use,anova_thresh)
#             this_X_train = this_X_train[roi_use_anova,:]
#             this_X_test = this_X_test[roi_use_anova,:] 
        
        w=fit_IEM_slide(this_X_train,angles_use)
        OUT[tst_index,:] = (w@this_X_test).T
    if center:
        return(center_resp(OUT,ang))
    else:
        return OUT    
        

# fitting from localizer data
def build_model(EV,DBR,donut_thresh=0,anova_thresh=0,pre_cleaned=0): # for export, no CV
    rois = DBR.index
    EVU = EV[EV.d1==1]
    ang = EVU.d0.values.copy()
    ang[ang==180] = 0
    M_all =dict()
    for roi in rois:
        M = dict() # model
        this_dat = DBR[roi] # n_roi x n_trials
        if not pre_cleaned:
            dat1 = this_dat[:,EV.d1==1]
        else:
            dat1 = this_dat.T
        n_roi = np.shape(this_dat)[0]
        ind_use = np.arange(n_roi)
        if donut_thresh:
            roi_use_donut = get_donut_mask(this_dat,EV,donut_thresh)
            this_dat = dat1[roi_use_donut,:]
            ind_use = ind_use[roi_use_donut]
        else:
            this_dat=dat1
        if anova_thresh:
            angles_use = EVU.d0.values-1
            roi_use_anova = get_anova_mask(this_dat,angles_use,anova_thresh)
            this_dat = this_dat[roi_use_anova,:]
            ind_use = ind_use[roi_use_anova]
        # now lets fit model...
        w = fit_IEM_slide(this_dat,ang)
        M['W'] = w
        M['ind'] = ind_use
        M_all[roi] = M
    return M_all

def fit_model(M_all,DBR_Task,pre_cleaned=1):
    rois = DBR_Task.index
    OUT = dict()
    for roi in rois:
        this_M = M_all[roi]
        if pre_cleaned:
            this_dat = DBR_Task[roi].T
        else:
            this_dat = DBR_Task[roi].T
            this_dat = this_dat[this_M['ind'],:]
        out = (this_M['W']@this_dat).T
        OUT[roi] = out
    OUT = pd.Series(OUT)
    return OUT