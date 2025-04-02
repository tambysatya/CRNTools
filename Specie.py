

import libsbml
from astread import parse_expr
import sympy
import math



class Specie (libsbml.Species):
    def __init__ (self, specie):
        super().__init__(specie)
        self.symbol = sympy.Dummy()
        
        value = specie.getInitialConcentration()
        if math.isnan(value):
            self.initialvalue = specie.getInitialAmount()
        else:
            self.initialvalue = value

    def buildDifferentialEquation (self, sr, species, reactions, compartments, symbolic=False):
        expr = 0
        vi = compartments[self.getCompartment()].symbol
        me = self.getId()

        for reactionid in sr.predecessors(me): #reactions for which I am a product
            reaction = reactions[reactionid]
            coef = sr[reactionid][me]["stoichiometry"]
            if symbolic == True:
                term = coef*reaction.symbolic_formula
            else:
                term = coef*reaction.kineticformula
            if reaction.getNumReactants() > 0:
                reactantid = reaction.getReactant(0).getSpecies()
                reactant = species[reactantid]
                compartment = compartments[reactant.getCompartment()]
                if symbolic == True:
                    term /= compartment.symbol
                else:
                    term /= compartment.val
            expr += term
        for reactionid in sr.successors(me):
            reaction = reactions[reactionid]
            coef = sr[me][reactionid]["stoichiometry"]
            if symbolic==True:
                expr -= coef*reaction.symbolic_formula
            else:
                expr -= coef*reaction.kineticformula

        return expr / vi
    
