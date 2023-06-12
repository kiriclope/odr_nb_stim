import numpy as np
from scipy.special import i0
import time


def von_mises(theta, kappa):
    return np.exp(kappa * np.cos(theta)) / (2.0 * np.pi * i0(kappa))


# def decode_rE(rE, a_ini=0, a_fin=360, N=512):
#     #Population vector for a given rE
#     # return ( angle in radians, absolut angle in radians, abs angle in degrees )
#     N=len(rE)
#     Angles = np.linspace(a_ini, a_fin, N) 
#     angles = np.radians(Angles)
#     rE = np.reshape(rE, (1,N))
#     R = np.sum(np.dot(rE, np.exp(1j*angles)))/np.sum(rE)
    
#     angle_decoded = np.degrees(np.angle(R))
#     if angle_decoded<0:
#         angle_decoded = 360 + angle_decoded
    
#     return angle_decoded
#     #Mat.append(  [angle(R), abs(angle(R)) , degrees(abs(angle(R)))]  )
#     #return round( np.degrees(abs(np.angle(R))), 2)


# def circ_dist(a1,a2):
#     ## Returns the minimal distance in angles between to angles 
#     op1=abs(a2-a1)
#     angs=[a1,a2]
#     op2=min(angs)+(360-max(angs))
#     options=[op1,op2]
#     return min(options)



# def err_deg(a1,ref):
#     ### Calculate the error ref-a1 in an efficient way in the circular space
#     ### it uses complex numbers!
#     ### Input in degrees (0-360)
#     a1 = np.radians(a1)
#     ref = np.radians(ref)
#     err = np.angle(np.exp(1j*ref)/np.exp(1j*(a1) ), deg=True) 
#     err = np.round(err, 2)
#     return err


def model(totalTime, targ_onset_1, targ_onset_2, presentation_period, angle_target_i, angle_separation, tauE=9, 
          tauI=4, n_stims=2, I0E=0.1, I0I=0.5, GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.5, sigI=1.6,
          kappa_E=100, kappa_I=1.75, kappa_stim=100, N=512, 
          stim_strengthE=1., stim_strengthI=1., 
          phantom_st = 0.2, phantom_onset=500, phantom_on='off', phantom_duration=200):
    
    st_sim =time.time()
    
    dt=.1
    nsteps = int(np.floor(totalTime/dt)) 

    # angles
    origin = np.radians(angle_target_i) % (2.0 * np.pi)
    separation =  np.radians(angle_separation) % (2.0 * np.pi)
    angle_target = angle_target_i
    angle_distractor = angle_target_i - angle_separation
    
    if n_stims==1:
        separation=0

    # connectivity
    theta = np.linspace(0, 2.0 * np.pi, N)
    vm_E = von_mises(theta, kappa_E) 
    vm_I = von_mises(theta, kappa_I) 

    WE = np.zeros((N,N))
    WI = np.zeros((N,N))
    
    for i in range(N):
        WE[:,i] = np.roll(vm_E, i) 
        WI[:,i] = np.roll(vm_I, i) 
    
    # stimuli
    stimulus_1 = von_mises(theta - origin, kappa_stim)
    stimulus_2 = von_mises(theta - (origin + separation), kappa_stim)

    # events
    stimon1 = np.floor(targ_onset_1/dt)
    stimoff1 = np.floor(targ_onset_1/dt) + np.floor(presentation_period/dt) 
    stimon2 = np.floor(targ_onset_2/dt)
    stimoff2 = np.floor(targ_onset_2/dt) + np.floor(presentation_period/dt) 

    # rates
    rE = np.zeros(N)    
    rI = np.zeros(N)
    
    RE = np.zeros((N, nsteps))
    RI = np.zeros((N, nsteps))
    
    # TF
    # f = lambda x : x*x*(x>0)*(x<1) + np.reshape(array([np.sqrt(4*x[i]-3) for i in range(0, len(x))]).real, (N,1)) * (x>=1)
    f = lambda x: np.where(x >= 1.0, np.sqrt(np.abs(4.0 * x - 3.0)), x * x * (x > 0))
    # f = lambda x: x * x * (x > 0) * (x < 1) * (x >= 1.0) * np.sqrt(np.abs(4.0 * x - 3.0))
    
    background_silent = I0E
    background_on = (I0E + phantom_st) 
    # background_on = I0E  
    
    background_E = I0E 
    background_I = I0I 
    
    for i in range(nsteps):
        # background conditions 
        # if phantom_on == 'on':
        #     print('phantom')
        #     if i < float(phantom_onset/dt):
        #         background_E = background_silent                
        #     elif i > float(phantom_onset/dt) and i < float(phantom_onset/dt) + float(phantom_duration/dt) :
        #         background_E = background_on
        #     else:
        #         background_E = I0E
            
        ## stim condition
        if i < stimon1:
            background_E = background_silent  # baseline is a forced dead network            
        elif i > stimon1 and i < stimoff1:
            background_E = stim_strengthE * stimulus_1 + background_on
            background_I = stim_strengthI * stimulus_1 + I0I
        elif i > stimon2 and i < stimoff2:
            if n_stims==2:
                background_E = stim_strengthE * stimulus_2 + background_on
                background_I = stim_strengthI * stimulus_2 + I0I
        else:
            background_E = I0E
            background_I = I0I
        
        # noise
        noiseE = sigE * np.random.standard_normal(N)
        noiseI = sigI * np.random.standard_normal(N)
        
        # inputs
        IE = GEE * np.dot(WE, rE) - GIE * np.dot(WI, rI) + background_E
        II = GEI * np.dot(WE, rE) - GII * np.dot(WI, rI) + background_I
        
        # integration
        rE = rE + (f(IE) - rE + noiseE) * dt / tauE
        rI = rI + (f(II) - rI + noiseI) * dt / tauI

        # rE = rE + (f(IE + noiseE) - rE) * dt / tauE
        # rI = rI + (f(II + noiseI) - rI) * dt / tauI
        
        RE[:, i] = rE
        RI[:, i] = rI
        
    end_sim =time.time()
    total_time= end_sim - st_sim 
    total_time = np.round(total_time, 1)
    #print('Simulation time: ' + str(total_time) + 's')
        
    return RE

if __name__ == "__main__":

    rE = model(totalTime=1000,
        targ_onset_1=250,
        targ_onset_2=500,
        angle_target_i=0, # 170 far, 90 close
        presentation_period=50,
        angle_separation=180,
        tauE=20, tauI=10,
        n_stims=2,
        I0E=0.05, I0I=0.5, # I0E=-3.5 for off, 0.05 on
        GEE=0.068,
        GII=0.13,
        GEI=0.13,
        GIE=0.042,
        # sigE=0, sigI=0,
        sigE=5., sigI=5.,
        kappa_E=10, kappa_I=0,
        kappa_stim=10., N=512,
        stim_strengthE=50, stim_strengthI=0.,
        phantom_st=1.2, phantom_onset=50000, phantom_duration=100)
