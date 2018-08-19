# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:13:23 2017

@author: Borrero Lab
"""

import os, sys; 

sys.path.append (os.path.abspath (".")); 
import ring_optimize as ro; 

# No need to import Bokeh here ... 
# Import pyplot instead. 

import matplotlib.pyplot as pp; 

import numpy; 



program = ro.SimpleSetup (); 
program.set_up (); 


# Get the minimum and maximum magnet ring radius allowed: 
m_range = (program.cell_params.m_constrain_min, \
					program.cell_params.m_constrain_max); 
# Define the directory where we will be writing to: 
write_dir = "Optimized-Plots"; 

# Number of points, x values, y values, etc.: 
N = 128; 
xs = ys = numpy.linspace (-program.cell_params.m_constrain_max, \
				+program.cell_params.m_constrain_max, N); 
x = xs.reshape (N, 1); 
y = ys.reshape (1, N); 

dx = dy = xs[1] - xs[0]; # How much xs or ys go up by. 

# Make the mesh coordinates have a little bit more points inside them: 
meshX = meshY = numpy.concatenate ((xs - dx / 2, [xs[-1] + dx / 2])); 
state = program.optimize.last_result.x; 

# Find the predicted magnetic field as a 2D grid: 
B_actual = program.optimize.b_field_calc.net_field_z (state, x, y); 
r = numpy.sqrt (x ** 2 + y ** 2); # Just the "r" position here. 


pp.figure (figsize = (12, 12)); 
pp.title ("Magnetic Field"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.pcolormesh (meshX, meshY, B_actual); 
pp.savefig (write_dir + "/optimized-by-hand.png", dpi = 90); 


rs = r.ravel (); 
bs = B_actual.ravel (); 
mask = (rs > program.cell_params.r_min) * (rs < program.cell_params.r_max); 
some_rs = numpy.linspace (program.cell_params.r_min, program.cell_params.r_max, N); 
A = ro.calc_power_coef (rs[mask], bs[mask], 1/2, True); 

pp.figure (); 
pp.title ("Theoretical Dipole Magnetic Field Profile"); 
pp.xlabel ("Radial Position (mm)"); 
pp.ylabel ("Vertical Magnetic Field (gauss)"); 
pp.plot (rs, bs, 'o', markersize = 1, label = "Bz Samples"); 
pp.plot (some_rs, A * some_rs ** (1/2), label = "Power Fit"); 
pp.legend (loc = "best"); 
pp.savefig (write_dir + "/optimized-slice.png", dpi = 300); 


