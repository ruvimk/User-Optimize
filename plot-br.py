# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:41:17 2017

@author: Borrero Lab
"""

import matplotlib.pyplot as pp; 
import os, sys; 

sys.path.append (os.path.abspath (".")); 
import ring_optimize as ro; 

import numpy; 


# Load our program: 
program = ro.SimpleSetup (); 
program.set_up (); 



write_dir = "User-Optimize"; 
plot_dir = "Final-Plots"; 


# Load the number parameter by which to scale the magnetic field: 
if os.path.exists (write_dir + "/M-scale.txt"): 
	Ms = numpy.loadtxt (write_dir + "/M-scale.txt") [()]; 
else: 
	Ms = 1; 
program.optimize.setM (Ms * ro.M); 


# Let R be the radius we want to capture, N be the number of points: 
R = 130; 
N = 2 * R + 1; 
# Then s is an array with some points in one dimension: 
s = numpy.linspace (-R, R, N); 
# We convert that to two dimensions using reshape (): 
x = s.reshape ((N, 1)); 
y = s.reshape ((1, N)); 
r = numpy.sqrt (x ** 2 + y ** 2); # Radial position = sqrt (x^2 + y^2); 

# Take the last-optimized state vector into its own variable for easy access: 
state = program.optimize.last_result.x; 

# Find the X, Y, Z components of the magnetic field at each point (x, y): 
[Bx, By, Bz] = program.optimize.b_field_calc.net_field_xyz (state, x, y); 

# Let (xunit, yunit) be the normalized form of (x, y): 
xunit = x / r; 
yunit = y / r; 

# Get rid of NANs, which result from dividing by r=0 (one of the "r"s may = 0): 
xunit[numpy.isnan (xunit)] = 0; 
yunit[numpy.isnan (yunit)] = 0; 

# Take the dot product of the magnetic field vector with the cylindrical r^hat: 
Br = Bx * xunit + By * yunit; 
Bt = Bx * (-yunit) + By * xunit; 




if not os.path.exists (plot_dir): 
	os.makedirs (plot_dir); 




# Show some plots: 
pp.figure (figsize = (6, 4.5)); 
#pp.title ("Bx"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.xlim (s.min (), s.max ()); 
pp.ylim (s.min (), s.max ()); 
pp.pcolormesh (s, s, Bx); 
pp.colorbar (); 
pp.savefig (plot_dir + "/Bx.png", dpi = 300); 

pp.figure (figsize = (6, 4.5)); 
#pp.title ("By"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.xlim (s.min (), s.max ()); 
pp.ylim (s.min (), s.max ()); 
pp.pcolormesh (s, s, By); 
pp.colorbar (); 
pp.savefig (plot_dir + "/By.png", dpi = 300); 

pp.figure (figsize = (6, 4.5)); 
#pp.title ("Br"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.xlim (s.min (), s.max ()); 
pp.ylim (s.min (), s.max ()); 
pp.pcolormesh (s, s, Br); 
pp.colorbar (); 
pp.savefig (plot_dir + "/Br.png", dpi = 300); 

pp.figure (figsize = (6, 4.5)); 
#pp.title ("Bt"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.xlim (s.min (), s.max ()); 
pp.ylim (s.min (), s.max ()); 
pp.pcolormesh (s, s, Bt); 
pp.colorbar (); 
pp.savefig (plot_dir + "/Bt.png", dpi = 300); 

pp.figure (figsize = (6, 4.5)); 
#pp.title ("Bz"); 
pp.xlabel ("Position X (mm)"); 
pp.ylabel ("Position Y (mm)"); 
pp.xlim (s.min (), s.max ()); 
pp.ylim (s.min (), s.max ()); 
pp.pcolormesh (s, s, Bz); 
pp.colorbar (); 
pp.savefig (plot_dir + "/Bz.png", dpi = 300); 

# Print the min and max of the magnetic field (units: gauss): 
print ("Min Br: " + str (Br.min ())); 
print ("Max Br: " + str (Br.max ())); 


