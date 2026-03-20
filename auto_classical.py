# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:23:39 2024

@author: dcsal
"""

from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply
import numpy as np
import sys
import os
import time



class PowerSupplies:
    from qlab.devices.KeithleyPowerSupply import KeithleyPowerSupply

    def __init__(self, list_of_addresses = [], mode = 'CV'):
        """
        This class controls 3 Keithley 2231A power supplies at once. The 
        address of the devices have to be known beforehand.
        
        The class KeithleyPowerSupply sets voltage 0 0 0 and current MAX MAX MAX
        to each object by default.
        
        These devices have 3 channels, and this code uses only CH1 and CH2 of each,
        
        Ex: 
        supply = PowerSupplies('ASRL3::INSTR', 'ASRL5::INSTR', 'ASRL7::INSTR')
        """
        
        self._list_of_addresses = list_of_addresses
        self._num_supplies = len(self._list_of_addresses)
        self._Keithleys = []
        
        for index, key in enumerate(self._list_of_addresses):
            self._Keithleys.append(KeithleyPowerSupply(self._list_of_addresses[index], mode = mode))
        
    def __del__(self):
        for supply in self._Keithleys:
            supply._link.close()
        
    @property
    def output(self):
        return [supply.output for supply in self._Keithleys]
    
    @output.setter
    def output(self, status):
        for supply in self._Keithleys:
            supply.output = status
            
    @property
    def voltages(self):
        volts = np.concatenate([supply.voltages[:2] for supply in self._Keithleys])
        return volts
    
    @voltages.setter
    def voltages(self, input_volts):
        # Ex: supply.voltages = [0.0, 0.5, 2.0, 4.0, 3.0]
        input_volts = np.array(input_volts)
        assert input_volts.size == 2*self._num_supplies, "Too many or too few voltages provided. Provide {} values in a list (unit = Volts).".format(2*self._num_supplies)
        
        if np.any(input_volts> 8.0):
            raise ValueError("Maximum allowed of 6 Volts")
           
        for index, supply in enumerate(self._Keithleys):
             volts = np.zeros(3)
             volts[:2] = np.copy(input_volts[2*index: (2*index+2)])
             supply.voltages = volts
        
    @property
    def currents(self):
        currs = np.concatenate([supply.currents[:2] for supply in self._Keithleys])
        return currs
    
    @currents.setter
    def currents(self, input_currs):
        
        input_currs = np.array(input_currs)
        assert input_currs.size == 2*self._num_supplies, "Too many or too few voltages provided. Provide {} values in a list (unit = Amperes).".format(2*self._num_supplies)
        
        # if np.any(input_currs> 8.0):
        #     raise ValueError("Maximum allowed of 6 currs")
           
        for index, supply in enumerate(self._Keithleys):
             currs = np.zeros(3)
             currs[:2] = np.copy(input_currs[2*index, (2*index+1)])
             supply.currents = currs  
        
    @property
    def voltages_measure(self):
        return np.concatenate([supply.voltages_measure[:2] for supply in self._Keithleys])

    @property
    def currents_measure(self):
        return np.concatenate([supply.currents_measure[:2] for supply in self._Keithleys])
        
    

