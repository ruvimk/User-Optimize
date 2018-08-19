# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:34:17 2017

@author: Borrero Lab
"""

import matplotlib.pyplot as pp; 
import os, sys; 

sys.path.append (os.path.abspath (".")); 
import ring_optimize as ro; 

import numpy; 


program = ro.SimpleSetup (); 
program.set_up (); 

Ap = numpy.array ([627, 2]); 
water_level = 5; 

power = 0.5; 


m_range = (program.cell_params.m_constrain_min, \
					program.cell_params.m_constrain_max); 
water_range = (program.cell_params.r_min, program.cell_params.r_max); 

write_dir = "User-Optimize"; 
plot_dir = "Final-Plots"; 

if os.path.exists ("./measurements.txt"): 
	meas = numpy.loadtxt ("./measurements.txt"); 
else: 
	meas = numpy.zeros ((0, 2)); 

if os.path.exists ("./meas-offs.txt"): 
	r_offs = numpy.loadtxt ("./meas-offs.txt") [()]; 
else: 
	r_offs = 18; # mm 
meas[:, 0] += r_offs; 

if os.path.exists (write_dir + "/water-z.txt"):
	wz = numpy.loadtxt (write_dir + "/water-z.txt") [()]; 
else: 
	wz = 0; 
if os.path.exists (write_dir + "/M-scale.txt"): 
	Ms = numpy.loadtxt (write_dir + "/M-scale.txt") [()]; 
else: 
	Ms = 1; 
program.optimize.setM (Ms * ro.M); 

Slice_Average = 32; 
N_actual = 64; 
r_actual = numpy.linspace (m_range[0], m_range[1], N_actual); 
B_actual = program.Bz_slice (r_actual, Slice_Average, probe_z = wz); 

N_template = 128; 
r_template = numpy.linspace (m_range[0], m_range[1], N_template); 
tmp_mask = (r_actual >= 50) * (r_actual <= 70); 
B_template = program.desired_slice (r_template, power, B_actual.mean (axis = 1) [tmp_mask], r_actual [tmp_mask]); 

actual_mean = B_actual.mean (axis = 1); 
actual_std = B_actual.std (axis = 1); 



if meas.shape[0] > 0: 
	use_r_min = numpy.floor (meas[:, 0].min () / 5) * 5; 
	use_r_max = numpy.ceil (meas[:, 0].max () / 5) * 5; 
else: 
	use_r_min = m_range[0]; 
	use_r_max = m_range[1]; 


if not os.path.exists (plot_dir): 
	os.makedirs (plot_dir); 


pp.figure (figsize = (6, 4)); 
pp.xlabel ("Radial Position (mm)"); 
pp.ylabel ("Vertical Magnetic Field (gauss)"); 
pp.xlim (use_r_min, use_r_max); 
pp.xticks (numpy.arange (20, 110, 10) [1::2]); 
pp.plot (r_template, B_template, color = "#dd8800", label = "$\propto$ sqrt (r)"); 
pp.errorbar (r_actual, B_actual.mean (axis = 1), B_actual.std (axis = 1), color = 'k', label = "Predicted"); 
if meas.shape[0] > 0: 
	pp.plot (meas[:, 0], meas[:, 1], 'go', label = "Measured"); 
#pp.plot (r_actual, B_actual.mean (axis = 1), label = "$B_z$"); 
pp.legend (loc = "best"); 
pp.savefig (plot_dir + "/vertical-magnetic-field-slice.png", dpi = 300); 



r_mask = (meas[:, 0] >= 50) * (meas[:, 0] <= 70); 
Ak_data = Ap[0] * meas[:, 1][r_mask] / (2 * numpy.pi * water_level * meas[:, 0][r_mask] ** power); 
Ak = numpy.array ([Ak_data.mean (), Ak_data.std ()]); 
print ("Ak/I = {} ± {}".format (Ak[0], Ak[1])); 



