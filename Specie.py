

import libsbml
from astread import parse_expr
import sympy
import math



class Specie (libsbml.Species):
    def __init__ (self, specie):
        super().__init__(specie)
        self.reactions = set() #the reaction IDs where the considered specie is either a product or reactant
        self.symbol = sympy.Dummy()
        
        value = specie.getInitialConcentration()
        if math.isnan(value):
            self.initialvalue = specie.getInitialAmount()
        else:
            self.initialvalue = value

    def addReaction (self, reactionid):
        self.reactions.add(reactionid)
    def buildDifferentialEquation (self, reactions_dict, species, compartments):
        expr = 0
        vi = compartments[self.getCompartment()].symbol
        for reactionid in self.reactions:
            reaction = reactions_dict[reactionid]
            kinetic = reaction.kineticformula
            coef = reaction.getStoichiometryMatrixCoeficient(self)
            if coef == 1:
                expr += kinetic
            elif coef == -1:
                expr -= kinetic
            else:
                expr += coef*kinetic
            
            #TODO
            #r0 = reaction.getReactant(0).getSpecies()
            #vj = species[r0].getCompartment()
            #vj = compartments[vj].symbol
            expr = expr
        return expr / vi
    
