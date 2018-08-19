# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:59:12 2017

@author: Borrero Lab
"""

import os, sys; 

sys.path.append (os.path.abspath (".")); 
import ring_optimize as ro; 

import bokeh.colors; 
from bokeh.layouts import column; 
from bokeh.models import Button; 
from bokeh.palettes import RdYlBu3; 
from bokeh.layouts import row, widgetbox; 
from bokeh.models import ColumnDataSource; 
from bokeh.models.widgets import Slider, TextInput; 
from bokeh.plotting import figure, curdoc; 
from bokeh.core.property.descriptors import UnitsSpecPropertyDescriptor; 
from bokeh.models.glyphs import Text; 

import numpy; 


def to_slider (h): 
	return 1e4 / h ** 2; 
def from_slider (B): 
	return numpy.sqrt (1e4 / B); 


program = ro.SimpleSetup (); 
program.set_up (); 


if os.path.exists ("./measurements.txt"): 
	meas = numpy.loadtxt ("./measurements.txt"); 
else: 
	meas = numpy.zeros ((0, 2)); 

if os.path.exists ("./meas-offs.txt"): 
	r_offs = numpy.loadtxt ("./meas-offs.txt") [()]; 
else: 
	r_offs = 18; # mm 
meas[:, 0] += r_offs; 

m_mask = (meas[:, 0] >= 40) * (meas[:, 0] <= 80); 


r_min = program.cell_params.r_min; 
r_max = program.cell_params.r_max; 
water_level = 5; # in mm 


# Set up a data source: 

m_range = (program.cell_params.m_constrain_min, \
					program.cell_params.m_constrain_max); 
water_range = (program.cell_params.r_min, program.cell_params.r_max); 

write_dir = "User-Optimize"; 

if not os.path.exists (write_dir): 
	os.makedirs (write_dir); 

if os.path.exists (write_dir + "/power.txt"): # Template curve's power (.5 for sqrt curve). 
	P = numpy.loadtxt (write_dir + "/power.txt"); 
else: 
	P = numpy.array ([1/2, 1]); 
if os.path.exists (write_dir + "/B-plot-max.txt"): # Plot's maximum range (600 for 600 gauss, e.g.). 
	B_plot_max = numpy.loadtxt (write_dir + "/B-plot-max.txt") [()]; 
else: 
	B_plot_max = 400; 
	numpy.savetxt (write_dir + "/B-plot-max.txt", numpy.array ([B_plot_max])); # Save, for easy edit. 
if os.path.exists (write_dir + "/water-z.txt"): # Water level vertical displacement from the XY plane. 
	wz = numpy.loadtxt (write_dir + "/water-z.txt") [()]; 
else: 
	wz = 0; 
if os.path.exists (write_dir + "/M-scale.txt"): # Scalar by which to scale the magnetic field. 
	Ms = numpy.loadtxt (write_dir + "/M-scale.txt") [()]; 
else: 
	Ms = 1; 
Slice_Average = 5; 
N_template = 128; N_actual = 64; 
r_template = numpy.linspace (water_range[0], water_range[1], N_template); 
r_actual = numpy.linspace (m_range[0], m_range[1], N_actual); 
B_actual = program.Bz_slice (r_actual, Slice_Average, probe_z = wz); 

actual_mean = B_actual.mean (axis = 1); 
actual_std = B_actual.std (axis = 1); 

source_template = []; 
source_template2 = []; 
for power in P: 
	B_template = program.desired_slice (r_template, power, B_actual.mean (axis = 1), r_actual); 
	source_template.append (ColumnDataSource (data = dict (x = r_template, y = B_template))); 
	if meas.shape[0] > 0: 
		B_template2 = program.desired_slice (meas[:, 0], power, meas[:, 1][m_mask], meas[:, 0][m_mask]); 
		source_template2.append (ColumnDataSource (data = dict (x = meas[:, 0], y = B_template2))); 

source_measured = ColumnDataSource (data = dict (x = meas[:, 0], y = meas[:, 1])); 

source_actual = ColumnDataSource (data = dict (x = r_actual, y = actual_mean)); 

source_a_min = ColumnDataSource (data = dict (x = r_actual, y = B_actual.min (axis = 1))); 
source_a_max = ColumnDataSource (data = dict (x = r_actual, y = B_actual.max (axis = 1))); 
source_b_lower = ColumnDataSource (data = dict (x = r_actual, y = actual_mean - actual_std)); 
source_b_upper = ColumnDataSource (data = dict (x = r_actual, y = actual_mean + actual_std)); 

hs = program.optimize.get_hs (program.optimize.last_result.x); 
rs = program.optimize.get_rs (program.optimize.last_result.x); 
da = program.optimize.get_da (program.optimize.last_result.x); 
h0 = numpy.zeros_like (hs); 
source_magnets = ColumnDataSource (data = dict (x = rs, y = h0)); 
source_below_water = ColumnDataSource (data = dict (x = rs, y = -h0)); 

# A source of 2 points defining a line segment to draw to "show" where the water is: 
source_water = ColumnDataSource (data = dict (x = numpy.array ([program.cell_params.r_min, \
							program.cell_params.r_max]), y = numpy.array ([wz, wz]))); 
source_floor = ColumnDataSource (data = dict (x = numpy.array ([r_min, r_max]), y = numpy.array ([-1, -1]) * water_level + wz)); 

ring_width = 2 * ro.magnet_radius; 
ring_height = ro.magnet_height; 

desc_width = ring_width; 
desc_height = ring_height; 


def create_figure (): 
	# Create a plot and style: 
	p = figure (x_range = m_range, 
				y_range = (-max (50, program.optimize.cell_params.m_height_max), B_plot_max), 
				toolbar_location = None); 
	p.xaxis.axis_label = "Radial Position (mm)"; 
	p.yaxis.axis_label = "Vertical Field Strength (G)"; 
	
	p.border_fill_color = "white"; 
	p.background_fill_color = "white"; 
	p.grid.grid_line_color = None; 
	
	for template in source_template: 
		p.line ('x', 'y', source = template, line_width = 3); 
	for template in source_template2: 
		p.line ('x', 'y', source = template, line_width = 3); 
	p.line ('x', 'y', source = source_a_min, line_width = 1, color = "yellow"); 
	p.line ('x', 'y', source = source_a_max, line_width = 1, color = "yellow"); 
	p.line ('x', 'y', source = source_b_lower, line_width = 1, color = "orange"); 
	p.line ('x', 'y', source = source_b_upper, line_width = 1, color = "orange"); 
	p.line ('x', 'y', source = source_actual, line_width = 2, color = "#6AB652"); 
	p.line ('x', 'y', source = source_water, line_width = 1, color = "blue"); # Water-line. 
	p.line ('x', 'y', source = source_floor, line_width = 1, color = "black"); 
	p.circle ('x', 'y', source = source_measured, size = 5, color = 'navy', alpha = 0.5); 
	# "#CAB2D6"
	return p; 

def fig_add_magnets (p): 
	p.rect ('x', 'y', source = source_below_water, width = desc_width, height = desc_height, color = "#CAB2D6", alpha = 0.5); 

p = create_figure (); 
tr = fig_add_magnets (p); 


# Set up widgets: 

class MySliders: 
	height = radius = []; 
	def __init__ (our): 
		our.height = []; 
		our.radius = []; 
	def apply_all (our, add_before = None, add_after = None): 
		before = add_before if (add_before != None) else tuple (); 
		heights = tuple (our.height[i] for i in range (0, len (our.height))); 
		radii = tuple (our.radius[i] for i in range (0, len (our.radius))); 
		after = add_after if (add_after != None) else tuple (); 
		inputs1 = widgetbox (*(heights + before)); 
		inputs2 = widgetbox (*(radii + after)); 
		curdoc ().add_root (row (inputs1, inputs2, p, width = 1400)); 

sliders = MySliders (); 

p_input = tuple (TextInput (title = "Power", value = str (power)) for power in P); 

s_platform_z = Slider (title = "Water Z", value = wz, start = -3, end = 12); 
s_magnet_M = Slider (title = "Magnet M", value = Ms, start = 0, end = 1, step = 0.01); 

def update_plot_data (attr_name, old, new): 
	# Get ready to modify data by first loading the data: 
	state = program.optimize.last_result.x; 
	hs = program.optimize.get_hs (state); 
	rs = program.optimize.get_rs (state); 
	da = program.optimize.get_da (state); 
	# Modify the data using values from all the sliders: 
	for i in range (0, min (len (sliders.height), len (sliders.radius))): 
		s_h = sliders.height[i]; 
		s_r = sliders.radius[i]; 
		hs[i] = from_slider (s_h.value); 
		rs[i] = s_r.value; 
	state = program.optimize.make_state (hs, rs, da); 
	program.optimize.last_result.x = state; 
	Ms = s_magnet_M.value; 
	if Ms == 0: 
		Ms = s_magnet_M.value = 0.01; 
	numpy.savetxt (write_dir + "/M-scale.txt", numpy.array ([Ms])); 
	program.optimize.setM (s_magnet_M.value * ro.M); 
	wz = s_platform_z.value; 
	numpy.savetxt (write_dir + "/water-z.txt", numpy.array ([wz])); 
	try: 
		tmp_P = [float (inp.value) for inp in p_input]; 
		P = numpy.array (tmp_P); 
		if not os.path.exists (write_dir): 
			os.makedirs (write_dir); 
		numpy.savetxt (write_dir + "/power.txt", P); 
	except ValueError: 
		tmp_P = None; 
	# Generate new curves: 
	B_actual = program.Bz_slice (r_actual, Slice_Average, probe_z = wz); 
	for i in range (0, len (P)): 
		B_template = program.desired_slice (r_template, P[i], B_actual.mean (axis = 1), r_actual); 
		source_template[i].data = dict (x = r_template, y = B_template); 
	if meas.shape[0] > 0: 
		for i in range (0, len (P)): 
			B_template2 = program.desired_slice (meas[:, 0], P[i], meas[:, 1][m_mask], meas[:, 0][m_mask]); 
			source_template2[i].data = dict (x = meas[:, 0], y = B_template2); 
	# Send new curve data: 
	actual_mean = B_actual.mean (axis = 1); 
	actual_std = B_actual.std (axis = 1); 
	source_b_lower.data = dict (x = r_actual, y = actual_mean - actual_std); 
	source_b_upper.data = dict (x = r_actual, y = actual_mean + actual_std); 
	source_a_min.data = dict (x = r_actual, y = B_actual.min (axis = 1)); 
	source_a_max.data = dict (x = r_actual, y = B_actual.max (axis = 1)); 
	source_actual.data = dict (x = r_actual, y = actual_mean); 
	source_magnets.data = dict (x = rs, y = hs); 
	source_below_water.data = dict (x = rs, y = -hs); 
	source_water.data = dict (x = numpy.array ([program.cell_params.r_min, \
							program.cell_params.r_max]), y = numpy.array ([wz, wz])); 
	source_floor.data = dict (x = numpy.array ([r_min, r_max]), y = numpy.array ([-1, -1]) * water_level + wz); 
	# Save to file: 
	program.save (); 

bounds = program.optimize.get_all_bounds (); 
hs = program.optimize.get_hs (program.optimize.last_result.x); 
rs = program.optimize.get_rs (program.optimize.last_result.x); 

s_platform_z.on_change ("value", update_plot_data); 
s_magnet_M.on_change ("value", update_plot_data); 

for i in range (0, int (len (program.cell_params.S_up))): 
	b_h = bounds.heights[i]; 
	b_r = bounds.radii[i]; 
	h = min (hs[i], b_h[1]); 
	r = min (rs[i], b_r[1]); 
	h = max (h, b_h[0]); 
	r = max (r, b_r[0]); 
	s_h = Slider (title = "Strength {}".format (i + 1), value = to_slider (h), end = to_slider (b_h[0]), start = to_slider (b_h[1])); 
	s_r = Slider (title = "Radius {}".format (i + 1), value = r, start = b_r[0], end = b_r[1]); 
	sliders.height.append (s_h); 
	sliders.radius.append (s_r); 
	s_h.on_change ('value', update_plot_data); 
	s_r.on_change ('value', update_plot_data); 

for inp in p_input: 
	inp.on_change ('value', update_plot_data); 

def load_document (): 
	sliders.apply_all (add_before = p_input, add_after = (s_platform_z, s_magnet_M)); 
	update_plot_data ("", 0, 0); 

load_document (); 

curdoc ().title = "Optimize Magnet Positions"; 


