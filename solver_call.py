
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from functools import partial
import diffrax
from diffrax import diffeqsolve, ODETerm, Kvaerno3, Kvaerno4, Kvaerno5, SaveAt, PIDController, ImplicitEuler

@partial(jax.jit, static_argnums=[0,4,5])
def computeTrajectoryWithSolverTs(odes, y0,p0, ts, solver, max_steps=4096):
    t1=ts[-1]
    function = ODETerm(lambda t,y,args: odes(y, args))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-9, pcoeff=0.1, icoeff=0.2, dcoeff=0)
    saveat = SaveAt(t0=True, t1=True, ts=jnp.array(ts))
    sol = diffeqsolve(function, solver, t0=0, t1=t1, dt0=None, #dt0=0.1, 
                      y0=y0, 
                      saveat=saveat,
                      stepsize_controller=stepsize_controller,
                      args=p0,
                      throw=False,
                      adjoint=diffrax.ForwardMode(),
                      max_steps=max_steps)

    return sol.ys
@partial(jax.jit, static_argnums=[0,4,5])
def computeTrajectoryWithSolverSteps(odes, y0,p0, ts, solver, max_steps=4096):
    t1=ts
    function = ODETerm(lambda t,y,args: odes(y, args))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-9, pcoeff=0.1, icoeff=0.2, dcoeff=0)
    saveat = SaveAt(t0=True, t1=True, steps=True)
    sol = diffeqsolve(function, solver, t0=0, t1=ts, dt0=None,
                      y0=y0, 
                      saveat=saveat,
                      stepsize_controller=stepsize_controller,
                      args=p0,
                      throw=False,
                      adjoint=diffrax.ForwardMode(),
                      max_steps=max_steps)

    return sol
def computeTrajectoryWithSolver(odes,y0,p0,ts,solver,max_steps=4096):
    if isinstance(ts, list):
        return computeTrajectoryWithSolverTs(odes,y0,p0,ts,solver,max_steps)
    else:
        nfeats = len(y0)
        sol = computeTrajectoryWithSolverSteps(odes,y0,p0,ts,solver,max_steps)

        times = sol.ts[jnp.isfinite(sol.ts)]
        
        ret = sol.ys[jnp.isfinite(sol.ys)]
        ret = ret.reshape(-1,nfeats) 
        return times, ret




