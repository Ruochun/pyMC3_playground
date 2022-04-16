# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Calibration of the SCM parameters using Bayesian Optimization
# =============================================================================

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import ctypes
from numpy.ctypeslib import ndpointer
import pymc3 as pm
import arviz as az
import pred
import theano
import theano.tensor as tt

import pychrono.core as chrono
import pychrono.vehicle as veh
import time
import math
import random

obsData = np.array(
    [[1.0, 0.1105264874554570],
    [2.0,  0.1284066605714614],
    [3.0,  0.1446152832459853],
    [4.0,  0.1571563079528167],
    [5.0,  0.1679413064413113],
    [6.0,  0.1768981111025230],
    [7.0,  0.1857720722754904],
    [8.0,  0.1933190641386293],
    [9.0,  0.2011194970278973],
    [10.0, 0.2075504746135038],
    [11.0, 0.2148167411938940],
    [12.0, 0.2208218156542296],
    [13.0, 0.2272613543988050],
    [14.0, 0.2335657076907068],
    [15.0, 0.2395278102591918],
    [16.0, 0.2461011598342170],
    [17.0, 0.2507693709223819],
    [18.0, 0.2575280584746413],
    [19.0, 0.2625686831909034],
    [20.0, 0.2673013686359279]]
)

nSims = obsData.shape[0]
print(f"There are {nSims:d} data points")

def F(x, nSims):
    # func_out = [] 
    sample_no = random.randint(0, nSims-1)
    # t = time.time()
    
    # for i in range(nSims):
    nsim = ( sample_no + 1 ) * 1.0
    # Create the mechanical system
    mysystem = chrono.ChSystemSMC()

    # The path to the Chrono data directory containing various assets
    chrono.SetChronoDataPath('/home/ruochunz/chrono_data/')
    # Parameters for tire
    tire_rad = 0.8
    tire_vel_z0 = -3
    tire_center = chrono.ChVectorD(0, 0.02 + tire_rad, -1.5)
    tire_w0 = tire_vel_z0 / tire_rad

    # Create the ground
    ground = chrono.ChBody()
    ground.SetBodyFixed(True)
    mysystem.Add(ground)

    # Create the rigid body with contact mesh
    body = chrono.ChBody()
    mysystem.Add(body)
    body.SetMass(25.0 * nsim)
    body.SetInertiaXX(chrono.ChVectorD(2 * nsim, 2 * nsim, 2 * nsim))
    body.SetPos(tire_center + chrono.ChVectorD(0, 0.3, 0))

    # Load mesh
    mesh = chrono.ChTriangleMeshConnected()
    mesh.LoadWavefrontMesh(chrono.GetChronoDataFile('models/tractor_wheel/tractor_wheel.obj'))

    # Set visualization assets
    vis_shape = chrono.ChTriangleMeshShape()
    vis_shape.SetMesh(mesh)
    body.AddAsset(vis_shape)
    body.AddAsset(chrono.ChColorAsset(0.3, 0.3, 0.3))

    # Set collision shape
    material = chrono.ChMaterialSurfaceSMC()

    body.GetCollisionModel().ClearModel()
    body.GetCollisionModel().AddTriangleMesh(material,                # contact material
                                            mesh,                    # the mesh 
                                            False,                   # is it static?
                                            False,                   # is it convex?
                                            chrono.ChVectorD(0,0,0), # position on body
                                            chrono.ChMatrix33D(1),   # orientation on body 
                                            0.01)                    # "thickness" for increased robustness
    body.GetCollisionModel().BuildModel()
    body.SetCollide(True)

    # Create motor
    motor = chrono.ChLinkMotorRotationAngle()
    motor.SetSpindleConstraint(chrono.ChLinkMotorRotation.SpindleConstraint_OLDHAM)
    motor.SetAngleFunction(chrono.ChFunction_Ramp(0, math.pi / 4))
    motor.Initialize(body, ground, chrono.ChFrameD(tire_center, chrono.Q_from_AngY(math.pi/2)))
    mysystem.Add(motor)

    # Create SCM terrain patch
    # Note that SCMDeformableTerrain uses a default ISO reference frame (Z up). 
    # Since the mechanism is modeled here in a Y-up global frame, we rotate
    # the terrain plane by -90 degrees about the X axis.
    terrain = veh.SCMDeformableTerrain(mysystem)
    terrain.SetPlane(chrono.ChCoordsysD(chrono.ChVectorD(0, 0.2, 0), chrono.Q_from_AngX(-math.pi/2)))
    terrain.Initialize(2.0, 6.0, 0.04)

    # Constant soil properties
    terrain.SetSoilParameters( x[0],  # Bekker Kphi 0.2e6
                            x[1],      # Bekker Kc 0
                            x[2],    # Bekker n exponent 1.1
                            0,      # Mohr cohesive limit (Pa)
                            30,     # Mohr friction limit (degrees)
                            0.01,   # Janosi shear coefficient (m)
                            4e7,    # Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
                            3e4     # Damping (Pa s/m), proportional to negative vertical speed (optional)
    )

    # Run the simulation
    while(mysystem.GetChTime() < 3.0):
        mysystem.DoStepDynamics(0.02)
    # sinkage_obj = obsData[i][1]
    sinkage_sim = 1.0 - body.GetPos().y
    # func_out.append(sinkage_sim)

    del mysystem

    # elapsed = time.time() - t

    # print(f"One sim time: {elapsed:f}")
    # if value_return < 0.0001:
    # print("x=%.2f %.2f %.2f f(x)=%.12f" % (x[0], x[1], x[2], value_return))

    #return np.array(func_out), sample_no
    return sinkage_sim, sample_no
    """
    res = np.array([1.,1.,1.,1.,1.])
    elapsed = time.time() - t
    print(f"One sim time: {elapsed:f}")
    return res
    """

def likelihood_func(cali_param, data, sigma):
    model_output, sample_no = F(cali_param, nSims)
    #return -(0.5/sigma**2)*np.sum((model_output - data[:nSims])**2)
    return -(0.5/sigma**2)*(model_output - data[sample_no])**2

def main():
    # Load data for VHB 4910
    m = obsData[:,0] * 25. # this is in fact the wheel mass
    sinkage = obsData[:,1].flatten() # this is the corresponding sinkage
    sigma = 1.

    print(f"Running on PyMC3 v{pm.__version__}") 

    logl = LogLike(likelihood_func, sinkage, sigma)

    basic_model = pm.Model()
    nsteps = 5000
    with basic_model:
        x = pm.Uniform('x', lower=[0., -10., -10.], upper=[1e7, 10., 10.],  shape=3) # x is calibration params
        #x1 = pm.Uniform('x1', lower=0., upper=1e7)
        #x2 = pm.Uniform('x2', lower=-10., upper=10.)
        #x3 = pm.Uniform('x3', lower=-10., upper=10.)

        #sigma = pm.HalfNormal("sigma", sigma=1) 
        
        theta = tt.as_tensor_variable(x)
        #theta = tt.as_tensor_variable([x1,x2,x3])
        
        #pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        pm.Potential("likelihood",logl(theta))

    # map_est = pm.find_MAP(model=basic_model)
    # print(map_est)

    with basic_model:

        step = pm.Metropolis(vars=x)
        #step = pm.Metropolis(vars=[x1,x2,x3])
        tic = time.perf_counter()

        trace = pm.sample(nsteps, tune=nsteps//2, chains=2, step=step, discard_tuned_samples=True, return_inferencedata=False,
                #start={'x1':1e5, 'x2':0.5, 'x3':0.5})
                start={'x':[100000,0.5,0.5]})
        toc = time.perf_counter()
        print("===========================================================")
        print(f"Simulation finished in {toc - tic:0.4f} seconds")
        print("===========================================================")
        
        print(pm.summary(trace).to_string())
        #fig1 = az.plot_trace(trace)
        #fig2 = az.plot_autocorr(trace, var_names=['x'], max_lag=nsteps, combined=True)
        fig1 = pm.traceplot(trace)
        fig1.ravel()[0].figure.savefig("fig1.png")
        
        #fig2 = pm.plot_autocorr(trace, var_names=['x1','x2','x3'], max_lag=nsteps, combined=True)
        fig2 = pm.plot_autocorr(trace, var_names=['x'], max_lag=nsteps, combined=True)
        fig2.ravel()[0].figure.savefig("fig2.png")

        # print(az.summary(trace, round_to=2))

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    
    def __init__(self, loglike, data, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood


if __name__ == "__main__":
    main()

