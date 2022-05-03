
# import required packages
import numpy as np
import scipy.io as sio
import ctypes
from numpy.ctypeslib import ndpointer
#import pymc3 as pm
#import arviz as az

def main():
    # Load data for VHB 4910
    flow_data = sio.loadmat('flow_rate_data.mat')
    # cohesion = flow_data['data'][0,:]
    # radius = flow_data['data'][1,:]
    # flow_rate = flow_data['data'][2,:]
    x = flow_data['data'][:-1,:]
    y = flow_data['data'][-1,:]
    #y = np.log(y)
    # Now normalize data
    sqrt_g = np.sqrt(981.)    
    """
    upper_c, lower_c = 100., 0.
    upper_r, lower_r = 0.5, 0.
    upper_rho, lower_rho = 10., 0.
    upper_y, lower_y = 1e5, 0.
    """

    C = 0.16
    k = 2.07
    p = 2.59 
    pred_d = 2.*np.array([0.04])
    pred_rho = np.array([1.])
    pred_D = 2.*np.array([2.2])
    #pred_r = (pred_r - lower_r) / (upper_r - lower_r)
    #pred_rho = (pred_rho - lower_rho) / (upper_rho - lower_rho)

    pred_y = []
    out_d = []
    out_D = []
    out_rho = []
    for a_d in pred_d:
        for a_rho in pred_rho:
            for a_D in pred_D:
                out_d.append(a_d)
                out_rho.append(a_rho)
                out_D.append(a_D)
                pred_y.append(C * a_rho * sqrt_g * (a_D - k*a_d)**p)
    """
    print_arr(denormalize(out_r, lower=lower_r, upper=upper_r), "radius:")
    print_arr(denormalize(out_rho, lower=lower_rho, upper=upper_rho),'density:')
    print_arr(denormalize(pred_y,lower=lower_y, upper=upper_y),"flow rate pred:")
    """
    print_arr(out_d, "Particle diameter:")
    print_arr(out_rho, "Density:")
    print_arr(out_D, "Orifice diameter:")
    print_arr(pred_y, "Predicted flowrate:")

def print_arr(arr, arr_name="Array:"):
    print(arr_name)
    print(arr)

def normalize(arr, lower=0., upper=100., self_range=False):
    a = np.array(arr)
    if self_range:
        a = (arr - np.min(a)) / (np.max(a) - np.min(a))
    else:
        a = (a - lower) / (upper - lower)
    return a

def denormalize(arr, lower=0., upper=100.):
    a = np.array(arr)
    a = (upper-lower)*a + lower
    return a

if __name__ == "__main__":
    main()

#radius:
#[0.07  0.07  0.07  0.125 0.125 0.125 0.3   0.3   0.3  ]
#density:
#[0.5  3.25 7.   0.5  3.25 7.   0.5  3.25 7.  ]
#flow rate pred:
#[ 153.73735731  999.29282248 2152.32300227  136.6896836   888.48294342
#         1913.65557045   90.20711768  586.34626494 1262.89964757]

