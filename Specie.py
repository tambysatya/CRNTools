

import libsbml
from astread import parse_expr
import sympy
import math

import jax
import jax.numpy as jnp



class Specie (libsbml.Species):
    def __init__ (self, specie):
        super().__init__(specie)
        self.symbol = sympy.Dummy()

        
        if specie.isSetInitialConcentration():
            self.val = specie.getInitialConcentration()
        elif specie.isSetInitialAmount():
            self.val = specie.getInitialAmount()
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
                coef = sr[reactionid][me]["stoichiometry"]
                term = coef*reaction.symbolic_formula
                if reaction.getNumReactants() > 0:
                    reactantid = reaction.getReactant(0).getSpecies()
                    reactant = species[reactantid]
                    compartment = compartments[reactant.getCompartment()]
                    expr += term*compartment.symbol

        for reactionid in sr.successors(me):
            if sr[me][reactionid]["type"] == "reactant":
                reaction = reactions[reactionid]
                coef = sr[me][reactionid]["stoichiometry"]
                expr -= coef*reaction.symbolic_formula

        return expr / vi.symbol
    
