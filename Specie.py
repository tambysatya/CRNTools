

import libsbml
from astread import parse_expr
import sympy
import math

import jax
import jax.numpy as jnp



class Specie (libsbml.Species):
    def __init__ (self, specie, compartments):
        super().__init__(specie)
        self.symbol = sympy.Dummy()

        self.isConcentration = False
        
        if specie.isSetInitialConcentration():
            self.val = specie.getInitialConcentration()
            self.isConcentration = True
            print (f"{self.getId()}: initial concentration is {self.val}")
        elif specie.isSetInitialAmount():
            vi = compartments[self.getCompartment()]
            self.val = specie.getInitialAmount()/vi.getVolume()
            print (f"{self.getId()}: initial amount is {self.val}")
        else:
            print ("WARNING: ", self.getId(), " has no initial amount nor concentration")
            self.val=specie.getInitialConcentration()

    def buildDifferentialEquation (self, sr, species, reactions, compartments):
        expr = 0
        vi = compartments[self.getCompartment()]
        me = self.getId()

        for reactionid in sr.predecessors(me): #reactions for which I am a product
            if sr[reactionid][me]["type"] == "product":
                reaction = reactions[reactionid]
                coef = sr[reactionid][me]["stoichiometry"]#
                term = coef*reaction.symbolic_formula
                if reaction.getNumReactants() > 0 and len(compartments) > 1:
                    reactantid = reaction.getReactant(0).getSpecies()
                    reactant = species[reactantid]
                    compartment = compartments[reactant.getCompartment()]
                    print ("multiple compartments => multiplying by vj")
                    expr += term*compartment.symbol
                else:
                    expr += term

        for reactionid in sr.successors(me):
            if sr[me][reactionid]["type"] == "reactant":
                reaction = reactions[reactionid]
                coef = sr[me][reactionid]["stoichiometry"]
                expr -= coef*reaction.symbolic_formula

        if self.isConcentration:
            return expr
        else:
           return expr  / vi.symbol
    
