

import libsbml

import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=30' # Use multiples CPU devices
#os.environ["XLA_FLAGS"]="--xla_cpu_multi_thread_eigen=trueintra_op_parallelism_threads=1 --xla_force_host_platform_device_count=8"

import jax
jax.config.update("jax_enable_x64", True)
from jax.flatten_util  import ravel_pytree
from functools import partial
from enum import Enum
from astread import TIME_SYMBOL
from Compartment import *
from Specie import *
from Reaction import *
from Parameter import *
from Rule import *

from diffrax import Kvaerno3, Kvaerno4, Kvaerno5, SaveAt, PIDController, ImplicitEuler
from Function import *
from solver_call import *


import networkx as nx
import jax.numpy as jnp
from astread import parse_expr

import optimistix as optx
import lineax
import scipy
import scipy.integrate as si

import sys
jnp.set_printoptions(threshold=sys.maxsize)



def cast_to_float64(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), tree)
def cast_module_to_float64(module):
    for attr in dir(module):
        value = getattr(module, attr)
        if isinstance(value, jnp.ndarray):
            setattr(module, attr, jnp.asarray(value, dtype=jnp.float64))
    return module



def lsoda (odes, y0, p0, ts):
    function = lambda t,y: odes(y,p0) 
    ret = si.solve_ivp(function,(0,ts[-1]), y0, method="LSODA", t_eval=ts)
    return ret




def showParam (params, pdict):
    ndict = {pdict[p[0]]:p[1] for p in params.items()}
    return ndict

class VertexType (Enum):
    SPECIE=0
    REACTION=1

class CRNGraph (libsbml.Model):

    def __init__(self, sbml_mdl):
        sbml_mdl = libsbml.readSBML(sbml_mdl).getModel()
        super().__init__(sbml_mdl)

        if self.getNumEvents() > 0:
            raise ValueError ("Events found.")


        self.sr = nx.DiGraph() #specie-reaction graph

        self.compartments = {} # map compartmentid -> Compartment
        self.reactions = {} # map reactionid -> Reaction
        self.species = {}   # map specieid -> Specie : for the variables species
        self.parametersspecies = {} #map specieid -> Specie : for the constant species that acts like parameters
        self.parameters = {} #map identifier -> Parameter (global parameters)
        self.functions = {} # map functionID -> Function (for the functions defined in the sbml)
        self.rules = {"assignment":{}, "rate":{}, "algebraic":{}}

        self.initialassignments = {}
        for initialassignment in sbml_mdl.getListOfInitialAssignments():
            print ("found initial assignment")
            self.initialassignments[initialassignment.getSymbol()] = initialassignment
 
        for compartment in sbml_mdl.getListOfCompartments():
            self.compartments[compartment.getId()] = Compartment(compartment)

        for function in sbml_mdl.getListOfFunctionDefinitions():
            self.functions[function.getId()] = Function(function)

        for param in sbml_mdl.getListOfParameters():
            self.parameters[param.getId()] = Parameter(param)

        for specie in sbml_mdl.getListOfSpecies():
            s = Specie(specie, self.compartments)
            specieid = s.getId()
            self.sr.add_node(specieid, bipartite=VertexType.SPECIE)
            if specie.getBoundaryCondition() or specie.getConstant():
                self.parametersspecies[specieid] = s
            else:
                self.species[specieid] = s

        for rule in sbml_mdl.getListOfRules():
            if rule.isAssignment():
                self.rules["assignment"][rule.getVariable()] = Rule(rule)
            elif rule.isAlgebraic():
                self.rules["algebraic"][rule.getIdAttribute()]=Rule(rule)
                print ("[Warning] algebraic rule found.")
                raise ("Algebraic rule found")
            elif rule.isRate():
                self.rules["rate"][rule.getIdAttribute()]=Rule(rule)
            else:
                raise ValueError ("Unimplemented rule managment")
        print ("Rules found ",len(self.rules["assignment"]), " assigment, ", len(self.rules["algebraic"])," algebraic, ", len(self.rules["rate"]), " rate.")
        #symbol map to rewrite expressions using unique symbols
        pdict = {sp.Function(sym.getId())():sym.symbol for sym in self.parameters.values()}
        cdict = {sp.Function(sym.getId())():sym.symbol for sym in self.compartments.values()}
        sdict = {sp.Function(sym.getId())():sym.symbol for sym in self.species.values()}
        spdict = {sp.Function(sym.getId())():sym.symbol for sym in self.parametersspecies.values()}
        ruledict = {sp.Function(name)():rule.formula for name, rule in self.rules["assignment"].items()}
        symdict = pdict | cdict | sdict | spdict

        self.symdict = symdict



        for reaction in sbml_mdl.getListOfReactions():
            reactionid=reaction.getId()
            self.sr.add_node(reactionid, bipartite=VertexType.REACTION)
            self.reactions[reactionid]=Reaction(reaction, self.functions, symdict, self.rules["assignment"])

            for reactant in reaction.getListOfReactants():
                sid = reactant.getSpecies()
                self.sr.add_edge(sid, reactionid, type="reactant", stoichiometry=reactant.getStoichiometry())
            for product in reaction.getListOfProducts():
                sid = product.getSpecies()
                self.sr.add_edge(reactionid, sid, type="product", stoichiometry=product.getStoichiometry())
            for modifier in reaction.getListOfModifiers():
                sid = modifier.getSpecies()
                self.sr.add_edge(sid,reactionid, type="modifier")

        self.identifyConstantSpecies()
        print (f"Found {len(self.species)} species, {len(self.parameters)} global parameters, {len(self.compartments)} compartments, {len(self.reactions)} reactions and {len(self.parametersspecies)} constant species.")

        
    def identifyConstantSpecies(self):
        species = {}
        for specieid, specie in self.species.items():
            expr = specie.buildDifferentialEquation(self.sr, 
                                                    self.species|self.parametersspecies,
                                                    self.reactions, 
                                                    self.compartments)
            if expr == 0:
                self.parametersspecies[specieid] = specie
            else:
                species[specieid] = specie

        self.species = species


            
    def listParameters(self):
        params = []
        params += list(self.parameters.values())
        for r in self.reactions.values():
            params+= list(r.localparameters.values())
        params += list (self.compartments.values())
        params += list (self.parametersspecies.values())
        return params

    def rewrite(self, rename=False):
        """
            Displays the differential equations:
                if rename = True, relabels the parameters p1...pn and the species x1...xp
        """
        odes = [specie.buildDifferentialEquation(self.sr,
                                                 self.species | self.parametersspecies,
                                                 self.reactions,
                                                 self.compartments) for specieid, specie in self.species.items()]
        params = self.listParameters()
        symbs = {}
        i = 1
        for p in params:
            if rename:
                symbs[p.symbol] = sp.symbols("k"+str(i))
            else:
                symbs[p.symbol] = sp.symbols(p.getId())
            i = i+1
        i=1
        for s in self.species.values():
            if rename:
                symbs[s.symbol] = sp.symbols("x"+str(i))
            else:
                symbs[s.symbol] = sp.symbols(s.getId())
            i = i+1

        exprs = [ode.subs(symbs) for ode in odes]



        for specieid,expr in zip(self.species.keys(),exprs):
            print (f"{specieid}:", expr)

    def generateODEs(self):
        exprs = []
        for specie in self.species.values():
            expr = specie.buildDifferentialEquation(self.sr,
                                                    self.species|self.parametersspecies,
                                                    self.reactions,
                                                    self.compartments)
            exprs.append(expr)
        entities = {entitie.getId():entitie for entitie in list(self.species.values())+self.listParameters()}
        self.initializesEntities(entities.values())

        if len(self.species) > 0:
            ssymb,sval = zip (*[(str(specie.symbol)[1:], specie.val) for specie in self.species.values()])
            ssymb, sval = list(ssymb), list(sval)
        else: #no species, only variable parameters
            ssymb, sval = [],[]
        variables = {}


        for rule in self.rules["rate"].values():
            var = entities[rule.getVariable()]
            variables[var.symbol] = var.getId()

            expr = parse_expr(rule.getMath())
            print ("Found rateOf rule concerning ", rule.getVariable(), "=", expr)

            # application of functions and assignment rules
            expr = self._expandFunctions(self.functions, expr)
            expr = self._expandFunctions(self.rules["assignment"],expr)
            expr = expr.subs(self.symdict)
            exprs.append(expr)

        snames = [specie.getId() for specie in self.species.values()]
        psymb, pval, pnames = [], [], []
        vsymb, vval, vnames = [], [], []


        #total assignment (species + params)
        assignment = {specie.symbol:specie.val for specie in self.species.values()}

        for param in self.listParameters():
            assignment[param.symbol] = param.val
            if not param.symbol in variables:
                psymb.append(str(param.symbol)[1:])
                pval.append (param.val)
                pnames.append(param.getId())
            else:
                vsymb.append(str(param.symbol)[1:])
                vval.append(param.val)
                vnames.append(param.getId())
                

        odes = cast_module_to_float64(sympy2jax.SymbolicModule(exprs))
        y0 = jnp.array(sval+vval, dtype=jnp.float64) #TODO
        y0dict = dict (zip(ssymb+vsymb, y0))
        y0names = {specie.getId():specie.val for specie in self.species.values()} | {vsymb:vval for vsymb, vval in zip (vnames, vval)}
        #y0names = {specie.getId():specie.val for specie in self.species.values()}
        #y0dict = {str(specie.symbol)[1:]:specie.val for specie in self.species.values()}
        y0, mky0 = ravel_pytree(y0dict)
        print(y0names)


        dict_function = lambda y, params: mky0(y) | params
        self.odes = lambda y, params: cast_to_float64(jnp.stack(odes(**dict_function(y,params))))
        self.symbolic_odes = exprs

    


        return self.odes, y0,  dict(zip (psymb, jnp.array(pval, dtype=jnp.float64))), dict(zip(psymb, pnames)), assignment
    
    @partial(jax.jit, static_argnames=["self","ode_solver", "n_step"])
    def adjust(self, y0, p0, ts, target, n_step=10, ode_solver=Kvaerno5(), ode_max_steps=4096):
          
        loss_fn = lambda params,_: jnp.mean((computeTrajectoryWithSolver(self.odes, y0, params, ts, ode_solver, max_steps=ode_max_steps) - target)**2)
        solver = optx.LevenbergMarquardt(
                    rtol=1e-8, atol=1e-8,
                    #verbose=frozenset({"step", "accepted","loss", "step_size"})
                )

        res = optx.least_squares(loss_fn, solver, p0, max_steps=n_step, options={"autodiff_mode":"fwd"}, throw=False)
        return res.value


    def simulate(self,ts, solver=Kvaerno5(), max_steps=4096, return_times = False):
        f, y0, p0, names = self.generateODEs()
        times, traj = computeTrajectoryWithSolver(f, y0, p0,ts,solver, max_steps=max_steps)

        if return_times:
            return times, traj
        else:
            return traj

    def initializesEntities (self, entities):
        """
            Identifies entities for which the initial value are functions of the initial values of other entities
        """
        initialized, tocompute = {}, {}
        g = nx.DiGraph()

        for e in entities:
            if e.getId() in self.initialassignments:
                expr = parse_expr(self.initialassignments[e.getId()].getMath())
                expr = self._expandFunctions(self.functions, expr)
                expr = self._expandFunctions(self.rules["assignment"], expr)
                tocompute[e.symbol] = (e,expr.subs(self.symdict))
                g.add_node(e.symbol, mark=False)
            else:
                initialized[e.symbol] = e.val
                g.add_node(e.symbol, mark=True)

        for symb, (_,expr) in tocompute.items():
            for atom in expr.atoms(sp.Symbol):
                g.add_edge(atom, symb)

        if tocompute != {}:
            print ("Applying initialAssignments:")
        while True:
            finished=True
            newtocompute = {}
            for symb, (e,expr) in tocompute.items():
                ancestors = g.predecessors(symb)
                if all ([g.nodes[node]["mark"] for node in ancestors]):
                    initialized[symb] = expr.subs(initialized)
                    e.val = initialized[symb]
                    print ("assigning: ",symb, "=", initialized[symb])
                    g.nodes[symb]["mark"]=True
                    finished = False
                else:
                    newtocompute[symb]=expr
            tocompute = newtocompute
            if finished == True:
                break
        if tocompute != {}:
            raise ValueError ("Unable to compute initialAssignments for ", tocompute)


    def _expandFunctions(self, macros, expression):
        """
            expands the macros in the current expression.
            macros should be a map symbol -> object implementing the method apply [TODO should be a callable]
        """
        while True: #because there is no infinite recursions in python (nor do-until syntax) -_-
            recursive_call = False
            for func in expression.atoms(sp.Function):
                name = func.func.__name__
                if name in macros:
                    expression = macros[name].apply(expression)
                    recursive_call = True
            if recursive_call == False:
                break
        return expression


def test(solver=Kvaerno5(root_finder=optx.Chord(rtol=1e-8,
                                                atol=1e-8,
                                                linear_solver=lineax.AutoLinearSolver(well_posed=False))),
         ts=[1,2,3]):
    x =  CRNGraph(libsbml.readSBML("biomodels/BIOMD0000000005.xml").getModel())
    f, y0, p0, names = x.generateODEs()
    traj = computeTrajectoryWithSolver(x.odes, y0, p0, ts, solver)
    pinit = {p[0]:jnp.array(0.01, dtype=jnp.float64) for p in p0.items()}
    #pinit = {p[0]:p[1]+100 for p in p0.items()}
    return x,y0,p0,pinit,traj,solver,names


def generate_data(filename, t1):
    x =  CRNGraph(libsbml.readSBML(filename).getModel())
    return x.simulate(t1)


import time

def timeit (f):
    start = time.time()
    f()
    end = time.time()
    print ("elapsed: ", end-start)



#print (jax.devices())
#ts=[1,2,5,10,15,50,100,200,300]
#x,y0,p0,pinit,traj,solver,names=test(ts=ts)
#print (traj)
#f= lambda :x.adjust(y0,pinit,ts,traj,n_step=10000)
#timeit(f)
