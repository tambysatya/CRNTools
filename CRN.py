

import libsbml

from enum import Enum
from astread import TIME_SYMBOL
from Compartment import *
from Specie import *
from Reaction import *
from Parameter import *
from Rule import *

from Function import *


import networkx as nx
import jax.numpy as jnp
import jax
from astread import parse_expr

#from diffrax import diffeqsolve, ODETerm, Kvaerno3, Kvaerno4, Kvaerno5, SaveAt, PIDController, ImplicitEuler
#import jaxopt
#import optax
import scipy
from pybamm import jax_bdf_integrate

jax.config.update("jax_enable_x64", True)

def cast_to_float64(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), tree)
def cast_module_to_float64(module):
    for attr in dir(module):
        value = getattr(module, attr)
        if isinstance(value, jnp.ndarray):
            setattr(module, attr, jnp.asarray(value, dtype=jnp.float64))
    return module

class VertexType (Enum):
    SPECIE=0
    REACTION=1

class CRNGraph:

    def __init__(self, sbml_mdl):


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
            self.initialassignments[initialassignment.getSymbol()] = initialassignment
 
        for compartment in sbml_mdl.getListOfCompartments():
            self.compartments[compartment.getId()] = Compartment(compartment)

        for function in sbml_mdl.getListOfFunctionDefinitions():
            self.functions[function.getId()] = Function(function)

        for param in sbml_mdl.getListOfParameters():
            self.parameters[param.getId()] = Parameter(param)

        for specie in sbml_mdl.getListOfSpecies():
            s = Specie(specie)
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

        self.getY0()
        
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

    def rewrite(self):
        odes = [specie.buildDifferentialEquation(self.sr,
                                                 self.species | self.parametersspecies,
                                                 self.reactions,
                                                 self.compartments,
                                                 symbolic=True) for specieid, specie in self.species.items()]
        params = self.listParameters()
        symbs = {}
        i = 1
        for p in params:
            symbs[p.symbol] = sp.symbols("k"+str(i))
            i = i+1
        i=1
        for s in self.species.values():
            symbs[s.symbol] = sp.symbols("x"+str(i))
            i = i+1

        exprs = [ode.subs(symbs) for ode in odes]
        for expr in exprs:
            print (expr)

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
        ssymb,sval = zip (*[(str(specie.symbol)[1:], specie.val) for specie in self.species.values()])
        ssymb, sval = list(ssymb), list(sval)
        variables = {}
        for rule in self.rules["rate"]:
            print ("Found rateOf rule concerning ", rule.getVariable())
            var = entities[rule.getVariable()]
            variables[var.symbol] = var.getId()

            expr = parse_expr(rule.getMath())
            expr = self._expandFunctions(self.functions, expr)
            expr = self._expandFunctions(self.rules["assignment"],expr)
            expr = expr.subs(self.symdict)
            exprs.append(expr)

        snames = [specie.getId() for specie in self.species.values()]
        psymb, pval, pnames = [], [], []
        vsymb, vval, vnames = [], [], []
        for param in self.listParameters():
            if not param.symbol in variables:
                psymb.append(str(param.symbol)[1:])
                pval.append (param.val)
                pnames.append(param.getId())
            else:
                vsymb.append(str(param.symbol)[1:])
                vval.append(param.val)
                vnames.append(param.getId())
                

        odes = cast_module_to_float64(sympy2jax.SymbolicModule(exprs))
        #dict_function = lambda y: dict(zip(ssymb+vsymb+psymb, y))
        #self.odes = lambda y: jnp.stack(odes(**dict_function(y)))
        dict_function = lambda y, params: dict(zip(ssymb+vsymb, y)) | dict(zip(psymb,params))
        self.odes = lambda y, params: cast_to_float64(jnp.stack(odes(**dict_function(y,params))))

        #return self.odes, jnp.array(sval+vval+pval, dtype=jnp.float64), (snames+vnames+pnames)
        return self.odes, jnp.array(sval+vval, dtype=jnp.float64), jnp.array(pval, dtype=jnp.float64), (snames+vnames, pnames)
    def computeTrajectory(self, y0,p0, ts):
        odes = self.odes
        #saveat = SaveAt(ts=ts)
        t1=ts[-1]
        function = lambda y,t,args: odes(y, args)
        #function = lambda t,y,args: odes(y, args)
        #function = ODETerm(lambda t,y,args: odes(y,args))
        #if solver==None:
        #    solver = ImplicitEuler()
        #    stepsize_controller = PIDController(rtol=1e-8,atol=1e-8)
        #sol = diffeqsolve(function, solver, t0=t0, t1=t1, dt0=dt0, saveat=saveat, y0=y0, args=params,
        #                  stepsize_controller=stepsize_controller, max_steps=5000,
        #                  throw=True)
        #sol = scipy.integrate.solve_ivp(function,(min(ts), max(ts)),y0, t_eval=ts,method="LSODA", args=p0) 
        sol = jax_bdf_integrate(function, y0, jnp.array(ts, dtype=jnp.float64), p0)
        print(sol)
        return cast_to_float64(sol)

    def adjust(self, y0, p0, ts, target):


        def loss_fn(params):
            pred = self.computeTrajectory(y0,params,ts)
            print (pred.shape, target.shape)
            return pred.mean()
            #pred = jnp.where(jnp.isnan(pred) | jnp.isinf(pred), 1e6,pred)
            
            #return jnp.mean((pred-target)**2)
        
        def step (params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss


        #solver = jaxopt.BFGS(loss_fn, implicit_diff=True,
                             #maxiter=1000,
                             #maxls=50,
        #                     verbose=True, linesearch="zoom")
        #ret = solver.run(p0)
        #return ret.params
        print ("initial_loss=",loss_fn(p0))
        grad = jax.grad(loss_fn)
        print ("grad=", grad(p0))
        loss, grads = jax.value_and_grad(loss_fn)(p0)
        print ("initial loss=",loss_fn(p0), " grad=", grads)
        crash
        optimizer = optax.adabelief(1e-4)
        opt_state = optimizer.init(p0)
        params = p0
        for i in range(1000):
            params, opt_state, loss = step(params, opt_state)
            if i % 10 == 0:
                print (f"step={i}: {loss}")
            if loss < 1e-3:
                print (loss, ": ", params)
        return params
 
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

    def getY0(self):
        #y0 = jnp.array([specie.val for specie in self.species.values()], dtype=float)
        #p0 = jnp.array([param.val for param in self.listParameters()], dtype=float)
        #return y0,p0

        species = {specie.symbol:specie.getId() for specie in self.species.values()}
        params = self.listParameters()
        paramsym = {param.symbol:param.getId() for param in params}

        self.initializesEntities(list(self.species.values()) + params)

        y0 = jnp.array([specie.val for specie in self.species.values()], dtype=float)
        p0 = jnp.array([param.val for param in self.listParameters()], dtype=float)
        return y0,p0



