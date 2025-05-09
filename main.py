import torch

from CRN import *
from solver_call import *
import libsbml

import matplotlib.pyplot as plt

from diffrax import Kvaerno5

import os
from tqdm import tqdm
import numpy as np

solver = Kvaerno5()





def compareModels(i):
    fname = f"biomodels/BIOMD000000000{i}.xml"
    truth = torch.load(f"datas/{i}.pt").transpose(0,1)

    print("simulating model ", str(i))
    crn = CRNGraph(libsbml.readSBML(fname).getModel())
    odes, y0, p0, _ = crn.generateODEs()

    ts, ys = computeTrajectoryUntilSteadyState (odes, y0, p0, 10000, solver, track_gradient=False)
    ys = ys.transpose()
    nspecies = len(ys)

    
    for s in range(nspecies):
        plt.plot(truth[0], truth[s+1], label="tellurium")
        plt.plot(ts, ys[s], label="me")
        plt.legend()
        plt.savefig(f"figs/merged/{i}_{s}.png")

        plt.clf()
        plt.cla()
        plt.close()

        plt.plot(truth[0], truth[s+1], label="tellurium")
        plt.savefig(f"figs/{i}_{s}_tellurium.png")

        plt.clf()
        plt.cla()
        plt.close()


        plt.plot(ts, ys[s], label="me")
        plt.savefig(f"figs/{i}_{s}_me.png")

        plt.clf()
        plt.cla()
        plt.close()



def run(fpath,fname):
    try:
        tel_ans = np.load("../tellurium/results/" + fpath + ".npy").transpose()
    except Exception as e:
                    print(e)
                    return
    te_ts, te_vals = tel_ans[0], tel_ans[1:]

    try:
        crn = CRNGraph(fpath)
        odes, y0, p0, names , assignment= crn.generateODEs()
    except ValueError as e:
        print(e)
        return
    ret = cvode(odes,y0,p0,100)
    me_ts, me_vals = ret.t, ret.y
    #me_ts, me_vals = computeTrajectoryUntilSteadyState(odes,y0,p0,100,Kvaerno5())
    me_vals = me_vals.transpose()
    
    for i,(te, me) in enumerate (zip (te_vals, me_vals)):
        plt.plot(me_ts,me, label="me")
        plt.plot(te_ts, te, label="tellurium")
        plt.legend()
        plt.savefig(f"figures/{fname}_{i}.png")
        plt.clf()
    



directory="biomodels/"

def test():
    for filename in tqdm(list(sorted(os.listdir(directory)))):
        print(filename)
        if filename.startswith("BIO") and os.path.isfile(os.path.join(directory, filename)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                run (filepath, filename)
                

    






