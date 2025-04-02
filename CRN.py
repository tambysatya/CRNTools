

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
                                                    self.compartments,
                                                    symbolic=True)
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
                                                    self.compartments,
                                                    symbolic=False)
            exprs.append(expr)
        idx = jnp.arange(len(exprs)) 
        vmap_functions = jax.vmap (lambda i,x: jax.lax.switch(i, exprs, x))
        dict_function = lambda y, params: dict(zip(self.species.keys(), y)) | params

        self.odes = lambda y, params: vmap_functions(idx, dict_function(y,params))
        return self.odes
            
