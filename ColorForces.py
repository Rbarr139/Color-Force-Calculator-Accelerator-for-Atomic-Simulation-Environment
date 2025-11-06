#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:25:39 2025

@author: booros
"""

from ase.md.md import MolecularDynamics as MD
import numpy as np
from ase.units import kB
from mace.calculators import mace_mp


class ColorThermometer(MD):
    def __init__(self, atoms, timestep, temperature, exclude_elements=None, **kwargs):
        super().__init__(atoms, timestep)
        self.T_target = temperature
        #print("Excluded", exclude_elements)
        if isinstance(exclude_elements, str):
            exclude_elements = {exclude_elements}
        else:
            exclude_elements = set(exclude_elements or [])
        #print("Atoms", atoms)
        symbols = atoms.get_chemical_symbols()
        self.thermometer_indices = np.array([i for i, s in enumerate(symbols) if s not in exclude_elements])
        self.forces = self.atoms.get_forces()
        self.old_forces = self.forces.copy()
        
        #print("Initial forces norm:", np.linalg.norm(self.forces))
        #print("Atoms affected by thermostat:", self.thermometer_indices)

        if not hasattr(self, 'velocities') or self.velocities is None:
            self.velocities = np.zeros_like(self.atoms.positions)
        
    def step(self):
        #print("step() called")
        m = self.atoms.get_masses()[:,np.newaxis]
        self.atoms.positions += self.velocities * self.dt + 0.5 * self.old_forces / m * self.dt**2
        self.forces = self.atoms.get_forces()
        self.velocities += 0.5 * (self.forces + self.old_forces) / m * self.dt
        self.rescale()
        self.old_forces = self.forces.copy()
        
    def rescale(self, tol = 1e-5, max_iter = 100):
        for i in range(max_iter):
            v = self.velocities
            m = self.atoms.get_masses()
            dof = 3 * len(m)
            
            #KE_before = 0.5 * np.sum(m[:, np.newaxis] * v**2)
            #T_before = (2 * KE_before) / (dof * kB)
            #print(f"Iter {i}: Temperature before scaling: {T_before:.4f} K")
            
            KE = 0.5 * np.sum(m[:, np.newaxis] * v**2)
            T_current = (2 * KE) / (dof * kB)
            
            if abs(T_current - self.T_target) < tol:
                print(f"Converged at iteration {i}")
                break
            scale = np.sqrt(self.T_target / T_current)
            #print("Scale", scale)
            #print("Current", T_current)
            #print("Target", self.T_target)
            v[self.thermometer_indices] *= scale
            
            #KE_after_scale = 0.5 * np.sum(m[:, np.newaxis] * v**2)
            #T_after_scale = (2 * KE_after_scale) / (dof * kB)
            #print(f"Iter {i}: Temperature after scaling: {T_after_scale:.4f} K")            
            
            total_momentum = np.sum(m[:, np.newaxis] * v, axis=0)
            avg_momentum = total_momentum/ len(self.thermometer_indices)
            v[self.thermometer_indices] -= avg_momentum / m[self.thermometer_indices, np.newaxis]
            
            #KE_after_corr = 0.5 * np.sum(m[:, np.newaxis] * v**2)
            #T_after_corr = (2 * KE_after_corr) / (dof * kB)
            #print(f"Iter {i}: Temperature after momentum correction: {T_after_corr:.4f} K")
            if i == 99:
                print(f"Converged at iteration {i}", "\nCurrent Temperature: ", T_current)
            
            self.velocities = v



from ase.calculators.emt import EMT
from fairchem.core import OCPCalculator
from ase.calculators.calculator import all_changes
import numpy as np

parent = OCPCalculator(checkpoint_path="/Users/booros/REU Summer Stuff/Research/esen_30m_oam.pt")
#parent = EMT
#parent = mace_mp(model="small", device='cpu') # Path to downloaded checkpoint

class LayeredColorForceCalculator(parent):
    def __init__(self, element, color_force=0.01, direction=[1,0,0], layer_length=2.04, layer_orientation = 'x', **kwargs):
        parent.__init__(self, **kwargs)
        self.direction = np.array(direction)
        self.element = element
        self.color_force = color_force
        self.layer_length = layer_length
        self.layer_floor = layer_length//1
        self.layer = layer_orientation
        self.layer_direction()
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes) # Run wrapped calculator first
        self.forces = self.results['forces']
        color_charge = np.zeros_like(self.forces) # Create an array of zeroes same size as forces array
        for i, atom in enumerate(atoms):
            if atom.symbol == self.element:
                if ((atom.position[self.layer]+(self.layer_length/2))//self.layer_floor)%2==0:
                    flip = +1
                else:
                    flip = -1
                color_charge[i] = flip * self.direction # If the symbol matches the proper element, substitute the direction for the zeroes for said atom
        self.forces = self.forces + self.color_force * color_charge # Add the color force array to the original forces array
        self.results['forces'] = self.forces # Save new forces
    def layer_direction(self):
        if self.layer in ('x','a'):
            self.layer = 0
        elif self.layer in ('y','b'):
            self.layer = 1
        elif self.layer in ('z','c'):
            self.layer = 2