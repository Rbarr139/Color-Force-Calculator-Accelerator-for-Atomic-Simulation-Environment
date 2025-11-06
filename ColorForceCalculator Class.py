#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:29:13 2025

@author: booros
"""

from ase.calculators.emt import EMT
from ase.calculators.calculator import all_changes
import numpy as np

direction = [1,1,1]

class ColorForceCalculator(EMT):
    def __init__(self, element, color_force=1, direction=[1,0,0], **kwargs):
        EMT.__init__(self, **kwargs)
        self.direction = np.array(direction)
        self.element = element
        self.color_force = color_force
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes) # Run wrapped calculator first
        color_charge = np.zeros_like(self.forces) # Create an array of zeroes same size as forces array
        for i, atom in enumerate(atoms):
            if atom.symbol == self.element:
                color_charge[i] = self.direction # If the symbol matches the proper element, substitute the direction for the zeroes for said atom
        self.forces = self.forces + self.color_force*color_charge # Add the color force array to the original forces array
        self.results['forces'] = self.forces # Save new forces
        
"""
Need to add in color force itself, as well as a designation of what atom/atoms should be used
Color force matrix to be added should be a matrix of 1s, -1s, and 0s along diagonal multiplied 
by the magnitude of the color force
0s dictated by not being an atom of the specified type
1s and -1s evenly split between the designated atoms for color force, specific assignment should
be allowed, maybe enforced?
"""

