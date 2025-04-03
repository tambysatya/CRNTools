

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
i           self.initialassignments[initialassignment.getSymbol()] = initialassignment
 
        for compartment in sbml_mdl.getListOfCompartments():
            self.compartments[compartment.getId()] = Compartment(compartment, initialassignments)

        for function in sbml_mdl.getListOfFunctionDefinitions():
            self.functions[function.getId()] = Function(function)

        for param in sbml_mdl.getListOfParameters():
            self.parameters[param.getId()] = Parameter(param, initialassignments)

        for specie in sbml_mdl.getListOfSpecies():
            s = Specie(specie, initialassignments)
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

        odes = sympy2jax.SymbolicModule(exprs)
        ssymb = [str(specie.symbol)[1:] for specie in self.species.values()]
        psymb = [str(param.symbol)[1:] for param in self.listParameters()]
        dict_function = lambda y, params: dict(zip(ssymb, y)) | dict(zip(psymb,params))

        self.odes = lambda y, params: jnp.stack(odes(**dict_function(y,params)))
        return self.odes
    
    def splitInitializedSet(self, entities):
        """
            Identifies entities for which the initial value are functions of the initial values of other entities
        """
        initialized, tocompute = [], {}
        g = nx.DiGraph()

        for e in entities:
            if e.getId() in self.initialassignments:
                tocompute[e.symbol] = parse_expr(self.initialassignments[e.getId()].getMath())
            else:
                initialized.append(e.symbol)

        for symb, expr in tocompute.items():
            for atom in expressions.atoms
        return initialized, tocompute


    #def _expandExpression(self, expression, macros):
    #    for function in macros.values():
    def getY0(self):
        y0 = jnp.array([specie.val for specie in self.species.values()], dtype=float)
        p0 = jnp.array([param.val for param in self.listParameters()], dtype=float)
        return y0,p0
