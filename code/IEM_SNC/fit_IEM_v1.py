import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool



n_chan = 9
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

def gen_basis_set(ang_s=180, n_chan=9, shift=1):
    ang = np.arange(ang_s)
#     chan_center = np.linspace(ang_s/n_ori_chans, ang_s, n_ori_chans,dtype=int) - shift
    chan_center = np.linspace(ang_s/n_chan, ang_s, n_chan,dtype=int) - shift
    basis_set = np.zeros((ang_s,n_chan))
    for c in np.arange(n_chan):
        basis_set[:,c] = make_basis_function(xx,chan_center[c])
    chan_center_mask = np.zeros(ang_s)==1
    chan_center_mask[chan_center]=1
    return basis_set, chan_center_mask

def gen_design_matrix(ang,ang_s=180):
    ang[ang==180]=0
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

def center_resp(chan_resp,center_ang):
    centered_chan_resp = np.zeros(np.shape(chan_resp))
    n_trials_use,n_ang = np.shape(chan_resp)
    if (np.isscalar(center_ang)) or (len(center_ang) == 1):
        center_ang = np.ones(n_trials_use,dtype=int)*center_ang
    cent = round(n_ang/2)
    for i in range(n_trials_use):
        centered_chan_resp[i,:] = np.roll(chan_resp[i,:],cent-center_ang[i])
    return centered_chan_resp

def get_fit_crossval(grp,X,Y,EV,donut_thresh=0,anova_thresh=0):
    G = EV.block.values
    trn_index = np.ones(n_trials)==1
    trn_index[G==grp] = 0
    tst_index = ~trn_index

    EVU = EV[trn_index]
    this_X_train = X[:,trn_index]
    this_X_test = X[:,tst_index]
    this_label_train = Y[trn_index]

    # select voxels! 
    if donut_thresh:
        roi_use_donut = get_donut_mask(this_X_train,EVU,donut_thresh)
        this_X_train = this_X_train[roi_use_donut,:]
        this_X_test = this_X_test[roi_use_donut,:]

    if anova_thresh:
        angles_use = EVU.d0.values-1
        roi_use_anova = get_anova_mask(this_X_train,angles_use,anova_thresh)
        this_X_train = this_X_train[roi_use_anova,:]
        this_X_test = this_X_test[roi_use_anova,:]

    out = pred_IEM(this_X_train,this_X_test,this_label_train)
    return (out,tst_index)

def fit_by_run_EV(EV,X,donut_thresh=0,anova_thresh=0,G=True,center=True,ang_s=180,n_chan=9): #
    # X - voxel data
    # EV- events with fields d0, d1 (if doing donut thresh), and block
    # ang_s - range of angles presented
    # center - whether to center output
    ang = EV.d0.values
    ang[ang==180] = 0
    n_trials=len(ang)
    if G:
        G = EV.block.values
    else:
        G = np.arange(n_trials)
    grps = np.unique(G)
    n_grp = len(grps)
  
    chan_resp = np.zeros((n_trials,ang_s))
    design_matrix = gen_design_matrix(ang)
    for shift in range(1,21): # no choice for now...
        basis_set, chan_center_mask = gen_basis_set(ang_s=ang_s,n_chan=n_chan,shift=shift)
        Y = design_matrix@basis_set
        
        print(shift,end=' ')
        
        
       
          
        with Pool(10) as p:
            OUT = p.map(get_fit_crossval,grps)
            
         
        for O in OUT:
            chan_resp[np.ix_(O[1],chan_center_mask)] = O[0]
#             trn_index = np.ones(n_trials)==1
#             trn_index[G==grp] = 0
#             tst_index = ~trn_index
            
#             EVU = EV[trn_index]
#             this_X_train = X[:,trn_index]
#             this_X_test = X[:,tst_index]
#             this_label_train = Y[trn_index]
            
#             # select voxels! 
#             if donut_thresh:
#                 roi_use_donut = get_donut_mask(this_X_train,EVU,donut_thresh)
#                 this_X_train = this_X_train[roi_use_donut,:]
#                 this_X_test = this_X_test[roi_use_donut,:]

#             if anova_thresh:
#                 angles_use = EVU.d0.values-1
#                 roi_use_anova = get_anova_mask(this_X_train,angles_use,anova_thresh)
#                 this_X_train = this_X_train[roi_use_anova,:]
#                 this_X_test = this_X_test[roi_use_anova,:]

#             chan_resp[np.ix_(tst_index,chan_center_mask)] = pred_IEM(this_X_train,this_X_test,this_label_train)
    if center:
        return(center_resp(chan_resp,ang))
    else:
        return chan_resp
    
def fit_by_run(X,ang,G=None,center=True,ang_s=180,n_chan=9): #
    # X - voxel data
    # Y - presented angle
    # G - groups, should be by run
    # ang_s - range of angles presented
    # center - whether to center output
    
    if G is not None:
        grps = np.unique(G)
    else:
        G = np.arange(len(ang))
        grps = np.unique(G)
    n_trials = len(ang)
    chan_resp = np.zeros((n_trials,ang_s))
    design_matrix = gen_design_matrix(ang)
    for shift in range(1,21): # no choice for now...
        basis_set, chan_center_mask = gen_basis_set(ang_s=ang_s,n_chan=n_chan,shift=shift)
        Y = design_matrix@basis_set
        
        for grp in grps:
            trn_index = np.ones(n_trials)==1
            trn_index[G==grp] = 0
            tst_index = ~trn_index
            
            chan_resp[np.ix_(tst_index,chan_center_mask)] = pred_IEM(X[:,trn_index],X[:,tst_index],Y[trn_index])
    if center:
        return(center_resp(chan_resp,ang))
    else:
        return chan_resp

def fit_by_roi(DAT,ang,G=None,ind_use=None,center=True):
    # DAT - voxel data by ROI
    recon_all = dict()
    for this_reg in DAT.index:
        this_dat = DAT[this_reg]
        if ind_use is not None:
            this_dat = this_dat[:,ind_use]
        recon_all[this_reg] = fit_by_run(this_dat,ang,G=G,center=center)
    return recon_all

def plot_by_ROI_recon(recon_all):
    if type(recon_all) is dict:
        for this_reg in recon_all.keys():
            plt.plot(np.mean(recon_all[this_reg],0))
    ax = plt.legend(DBR.index,bbox_to_anchor=(1.1, 1.05))
    plt.title('Localizer Orientation Recon (LORO)')
    plt.xlabel('$\Theta$')
    plt.show()

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
    
def build_model(EV,DBR,donut_thresh=0,anova_thresh=0): # for export, no CV
    rois = DBR.index
    EVU = EV[EV.d1==1]
    ang = EVU.d0.values.copy()
    ang[ang==180] = 0
    M_all =dict()
    for roi in rois:
        M = dict() # model
        this_dat = DBR[roi] # n_roi x n_trials
        dat1 = this_dat[:,EV.d1==1]
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

def fit_model(M_all,DBR_Task):
    rois = DBR_Task.index
    OUT = dict()
    for roi in rois:
        this_dat = DBR_Task[roi].T
        this_M = M_all[roi]
        this_dat = this_dat[this_M['ind'],:]
        out = (this_M['W']@this_dat).T
        OUT[roi] = out
    return OUT