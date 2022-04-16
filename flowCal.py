# import required packages
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import ctypes
from numpy.ctypeslib import ndpointer
import pymc3 as pm
import arviz as az
import pred

def main():
    # Load data for VHB 4910
    flow_data = sio.loadmat('flow_rate_data.mat')
    # cohesion = flow_data['data'][0,:]
    # radius = flow_data['data'][1,:]
    # flow_rate = flow_data['data'][2,:]
    x = flow_data['data'][:-1,:]
    y = flow_data['data'][-1,:]

    # Now normalize data
    upper_c, lower_c = 100., 0.
    upper_r, lower_r = 0.5, 0.
    upper_rho, lower_rho = 10., 0.
    upper_y, lower_y = 1e5, 0.
    c = (x[0,:] - lower_c) / (upper_c - lower_c)
    r = (x[1,:] - lower_r) / (upper_r - lower_r)
    rho = (x[2,:] - lower_rho) / (upper_rho - lower_rho)
    #y = (y - np.min(y)) / (np.max(y) - np.min(y))
    #y = np.log(y)
    y = pred.normalize(y, upper=upper_y, lower=lower_y)

    pred_r = np.array([0.07, 0.125,0.3])
    pred_rho = np.array([0.5, 3.25,7.])
    pred_r = (pred_r - lower_r) / (upper_r - lower_r)
    pred_rho = (pred_rho - lower_rho) / (upper_rho - lower_rho)

    print(f"Running on PyMC3 v{pm.__version__}") 

    basic_model = pm.Model()
    nsteps = 50000
    with basic_model:
        #m1 = pm.Uniform('m1', lower=-10, upper=10)
        #p1 = pm.Uniform('p1', lower=-3, upper=3)
        #m2 = pm.Uniform('m2', lower=-10, upper=10)
        #p2 = pm.Uniform('p2', lower=-3, upper=3)
        #b = pm.Uniform('b', lower=-100, upper=100)
        
        C = pm.Uniform('C', lower=0., upper=1000.)
        k = pm.Uniform('k', lower=-10., upper=10.)
        #p = pm.Uniform('p', lower=.5, upper=4.)
        D = pm.Uniform('D', lower=0., upper=50.)
        p = 5./2.
        #D = 4.5
        sigma = pm.HalfNormal("sigma", sigma=1)

        flow_rate = C * rho * (D - k*r)**p
        #flow_rate = np.log(C) + np.log(rho) + p*np.log(D - k*r)
        y_obs = pm.Normal("y_obs", mu=flow_rate, sigma=sigma, observed=y)

    # map_est = pm.find_MAP(model=basic_model)
    # print(map_est)

    with basic_model:
        step = pm.Metropolis(vars=[C, k, D, sigma])
        #step = pm.Metropolis(vars=[C,k, p,sigma])
        trace = pm.sample(nsteps, tune=nsteps//2, step=step, discard_tuned_samples=True, return_inferencedata=False,
                #start={'C':0.2, 'k':1., 'p':2.5, 'sigma':0.1})
                start={'C':0.2, 'k':1.,'D':3.,'sigma':0.1})
        fig1 = az.plot_trace(trace)
        fig2 = az.plot_autocorr(trace, var_names=['C', 'k', 'D'], max_lag=nsteps, combined=True)
        fig1.ravel()[0].figure.savefig("fig1.png")
        fig2.ravel()[0].figure.savefig("fig2.png")

        print(az.summary(trace, round_to=2))


if __name__ == "__main__":
    main()

