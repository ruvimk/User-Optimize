# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:41:03 2017

@author: Ruvim
"""

import os; 
import numpy; 
import scipy.optimize; 
import scipy.stats; 
import matplotlib.pyplot as pp; 
import matplotlib.patches as patches; # See http://matthiaseisen.com/pp/patterns/p0203/ 



# Fit from KJ Magnetics N42-14-14 B Strength.py, from folder Field-Strength: 
# M = 217209.80135205924; # for mm, valid from r = 0.4 in 
M = 220152.80788049; 

# Typical magnet radius and height for our N42 .25-by-.25-inch magnets: 
magnet_radius = 3.3; # mm 
magnet_height = 6.3; # mm 

n = scipy.stats.norm (); 


def parse_profile_method (power, r, actual, optimize_context, method): 
	if method == optimize_context.METHOD.B_MAX: 
		B_max = optimize_context.cell_params.B_max; 
		at_minR = optimize_context.cell_params.r_min ** (-3/2); 
		B_factor = B_max / at_minR; 
	elif method == optimize_context.METHOD.PROFILE: 
		B_factor = calc_power_coef (r, actual, power); 
	else: 
		B_factor = 1; 
	return B_factor; 

def profile_Keplerian (r, actual, optimize_context, method): 
	B_factor = parse_profile_method (-3/2, r, actual, optimize_context, method); 
	result = B_factor * r ** (-3/2); 
	result[numpy.isnan (result) + numpy.isinf (result)] = 0; 
	return result; 

def profile_sqrt (r, actual, optimize_context, method): 
	B_factor = parse_profile_method (1/2, r, actual, optimize_context, method); 
	result = B_factor * numpy.sqrt (r); 
	result[numpy.isnan (result) + numpy.isinf (result)] = 0; 
	return result; 


def do_nothing (): 
	return; 


# calc_power_coef (r, y_r, power): 
# 
# A function to calculate the coefficient A of a power 
# profile, y_r = A * r^power, given the data samples (r, y_r); 
# 

def calc_power_coef (r, y_r, power = 1, use_median = False): 
	possible = y_r / r ** power; 
	if use_median is True: 
		return numpy.median (possible); 
	return possible.mean (); 


def chord_length (R = 1, r = 1, d = 1): 
	return (1 / d) * numpy.sqrt (4 * d * d * R * R - (d * d - r * r + R * R) ** 2); 




class OptimizeWhat: 
	HS = 1; 
	RS = 2; 
	DA = 4; 
class FieldTargetMethod: 
	B_MAX = 1; 
	PROFILE = 2; 

OPTIMIZE = OptimizeWhat (); 
METHOD = FieldTargetMethod (); 




# SimpleSetup 
# 
# A class to make the process easier for writing a program 
# that uses ring_optimize to optimize a magnet arrangement. 
# 
# Minimal example program: 
# 
# program = ro.SimpleSetup ("optimize-params.txt"); 
#
# program.set_up (); 
# program.run (); 
#
# program.plot (); 
# 
# /End example program; 
# 
# 

class SimpleSetup: 
	class Params: 
		VER = 0; 
		RING_COUNT = 1; 
		B_MAX = 2; 
		H_MAX = 3; 
	# Param keys: 
	PARAM = Params (); 
	# Variables: 
	write_dir = "Optimized-Params"; 
	result_file = ""; 
	cell_params = {}; 
	optimize_params = {}; 
	optimize = {}; 
	plot_params = {}; 
	program_data = []; # Ring counts, etc. 
	# Program data: 2D grid of data in the 'magnet-arrangement.csv' file. 
	def __init__ (our, magnet_arrangement_filename = "{write}/magnet-arrangement.csv"): 
		our.result_file = magnet_arrangement_filename; 
		our.set_up (); 
	def Bz_slice (our, rs, slice_samples = 4, probe_z = 0): 
		state = our.optimize.last_result.x; 
		s_rs = our.optimize.get_rs (state); 
		s_hs = our.optimize.get_hs (state); 
		s_da = our.optimize.get_da (state); 
		state = our.optimize.make_state (s_hs + probe_z, s_rs, s_da); 
		r = rs.reshape ((len (rs), 1)); 
		if len (our.optimize.cell_params.M_num) > 0: 
			t_max = 2 * numpy.pi / numpy.min (our.optimize.cell_params.M_num); 
		else: 
			t_max = numpy.pi; 
		t = numpy.linspace (0, t_max, slice_samples, endpoint = False); 
		t = t.reshape ((1, len (t))); 
		x = r * numpy.cos (t); 
		y = r * numpy.sin (t); 
		Bz = our.optimize.b_field_calc.net_field_z (state, x, y); 
		return Bz; 
	def desired_slice (our, rs, power, actual_Bz = None, actual_rs = None): 
		if actual_rs is None: 
			actual_rs = rs; 
		if actual_Bz is None: 
			actual_Bz = our.Bz_slice (actual_rs); 
		mask = (actual_rs >= rs.min ()) * (actual_rs <= rs.max ()); 
		A = calc_power_coef (actual_rs[mask], actual_Bz[mask], power, True); 
		return A * rs ** power; 
	def load (our): 
		# Load the last result, if exists: 
		path = our.result_file.replace ("{write}", "{}").format (our.write_dir); 
		szResult = 3 * len (our.optimize.cell_params.S_up); 
		if os.path.exists (path): 
			# Try the NumPy default tab-separated read format first: 
			our.program_data = numpy.array (numpy.genfromtxt (path)); 
			if len (our.program_data.shape) < 1 or \
					numpy.isnan (our.program_data).sum () == \
					numpy.array (our.program_data).ravel ().shape[0]: # No data. Maybe it's a comma-separated file?: 
				our.program_data = numpy.genfromtxt (path, delimiter = ","); 
			if len (our.program_data.shape) < 1 or \
					numpy.isnan (our.program_data).sum () == \
					numpy.array (our.program_data).ravel ().shape[0]: # Still no data - empty file, probably. 
				our.cell_params.S_up = numpy.zeros ((0,)); 
				our.cell_params.M_num = numpy.zeros ((0,)); 
				if len (our.optimize.last_result.x) != szResult: 
					our.optimize.last_result.x = numpy.zeros ((szResult,)); 
			else: 
				# Hopefully, we have the data now that we can extract! 
				if len (our.program_data.shape) < 2: 
					our.program_data = our.program_data.reshape ((1, -1)); 
				our.cell_params.S_up = our.program_data[:, 1]; 
				our.cell_params.M_num = our.program_data[:, 2].astype (int); 
				rs = our.program_data[:, 3]; 
				hs = our.program_data[:, 4]; 
				da = our.program_data[:, 5]; 
				# Replace undefined values with our defaults: 
				our.cell_params.S_up[numpy.isnan (our.cell_params.S_up)] = 1; # Default UP. 
				our.cell_params.M_num[our.cell_params.M_num == 
						numpy.array ([numpy.nan]).astype (int) [0]] = 5; # Default magnets/ring. 
				rs[numpy.isnan (rs)] = our.cell_params.m_constrain_min; # Default use minimum R. 
				hs[numpy.isnan (hs)] = our.cell_params.minDZ; # Default make closest to the water. 
				da[numpy.isnan (da)] = 0; # Default don't rotate about Z axis. 
				# Load the state into our last_result: 
				our.optimize.last_result.x = our.optimize.make_state (hs, rs, da); 
			#our.optimize.last_result.x = numpy.loadtxt (our.write_dir + "/state.txt"); 
		else: 
			our.cell_params.S_up = numpy.zeros ((0,)); 
			our.cell_params.M_num = numpy.zeros ((0,)); 
			if len (our.optimize.last_result.x) != szResult: 
				our.optimize.last_result.x = numpy.zeros ((szResult,)); 
		our.optimize.update (); 
	def save (our): 
		path = our.result_file.replace ("{write}", "{}").format (our.write_dir); 
		# Make sure the output directory exists: 
		dir_name = os.path.dirname (path); 
		if len (dir_name) > 0 and not os.path.exists (dir_name): 
			os.makedirs (dir_name); 
		# Save the result: 
		state = our.optimize.last_result.x; 
		hs = our.optimize.get_hs (state); 
		rs = our.optimize.get_rs (state); 
		da = our.optimize.get_da (state); 
		Nd = len (our.optimize.cell_params.S_up); 
		for_csv = numpy.concatenate (( 
			numpy.linspace (1, Nd, Nd, endpoint = True).reshape (Nd, 1), 
			our.optimize.cell_params.S_up.reshape (Nd, 1), 
			our.optimize.cell_params.M_num.reshape (Nd, 1), 
			rs[0:Nd].reshape (Nd, 1), 
			hs[0:Nd].reshape (Nd, 1), 
			da[0:Nd].reshape (Nd, 1) 
		), axis = 1); 
		numpy.savetxt (path, for_csv, delimiter = "\t", \
			header = "Ring\tOrientation\tMagnets in Ring\tRadius (mm)\tHeight (mm)\tZ-Axis Rotation (rad)"); 
		#numpy.savetxt (our.write_dir + "/state.txt", our.optimize.last_result.x); 
		return; 
	def set_up (our): 
		# Prepare the parameters for ring optimization: 
		our.cell_params = CellParameters (); # Water cell parameters. 
		our.optimize_params = OptimizationParameters (); 
		# Create a ring optimize context: 
		our.optimize = RingOptimizeContext (our.cell_params, our.optimize_params); 
		# Set up plot parameters: 
		our.plot_params = PlotParameters (); 
		# Load the previous result, if available: 
		our.load (); 
		return; 
	def run (our, profile = profile_Keplerian, optimize_what = "all", method = METHOD.B_MAX): 
		# Optimize!: 
		if optimize_what == "all": 
			our.optimize.find_profile (profile, our.optimize.last_result.x, method = method); 
		else: 
			our.optimize.find_profile (profile, our.optimize.last_result.x, \
									optimize_what = optimize_what, method = method); 
		# Save the result: 
		our.save (); 
		return; 
	def plot (our, state = None): 
		if state is None: 
			state = our.optimize.last_result.x; 
		# Load plot parameters: 
		our.plot_params.readCellParameters (our.cell_params); 
		plotter = RingResultPlotter (our.optimize, our.plot_params); 
		# Plot: 
		plotter.plot3 (state); 
		plotter.plot4 (state); 
	def user (our): 
		# Ask the user to edit the parameters. 
		os.system ("bokeh serve --show user-optimize.py"); 






# CellParameters 
# 
# A class to hold the water cell parameters and limitations. 
# Parameters include the inner and outer radii of 
# the donut containing the water area, 
# the magnetic field strength factor (B_factor), 
# the minimum/maximum radii of placement of a magnet ring, 
# the minimum/maximum heights the rings can be lowered/raised to, 
# and the number of magnets in each magnet ring. 
# 

class CellParameters: 
	M_num = []; 
	S_up = []; # Magnet orientations (1 = up, 0 = no magnet, -1 = down). 
	r_min = 30; 
	r_max = 100; 
	B_max = 200; 
	# Minimum distance between magnets: 
	minDR = 11; # mm 
	minDZ = 12; # mm 
	# Magnet grid construction parameters: 
	# 1. Maximum and minimum magnet heights (below the water) we can allow: 
	m_height_min = 10; 
	m_height_max = 240; 
	# 2. Maximum and minimum magnet ring radii we can allow: 
	m_constrain_min = 10; # From how much we can fit ... 
	m_constrain_max = 120; # Up to what we can build using the 3D printer. 
	# 3. The range of radii to initialize the ring arrangement with (has 
	#     little to do with the final arrangement, really): 
	m_min = 30; 
	m_max = 200; 
	m_min_sep = 3; 
	# ... 
	def __init__ (our, B_max = 200, r_min = 30, r_max = 90): 
		our.B_max = B_max; 
		our.r_min = r_min; 
		our.r_max = r_max; 


# ProbeParameters 
# 
# The base-class for holding parameters on where 
# and how to probe the magnetic field. 
# 

class ProbeParameters: 
	p_cnt = 15; 
	p_min = -1; 
	p_max = +1; 
	probe_xs = []; 
	probe_ys = []; 
	probe_dx = 0; 
	probe_dy = 0; 
	probe_mesh_coord_x = []; 
	probe_mesh_coord_y = []; 
	probe_lenX = 0; 
	probe_lenY = 0; 
	donut_minR = 0; 
	donut_maxR = 0; 
	def readCellParameters (our, cell_parameters): 
		our.setBounds (cell_parameters.r_min, cell_parameters.r_max); 
	def setBounds (our, r_min, r_max): 
		our.p_min = -r_max; 
		our.p_max = +r_max; 
		our.donut_minR = r_min; 
		our.donut_maxR = r_max; 
		our.calcProbes (); 
	def calcProbes (our): 
		# Construct the B-field probes: 
		our.probe_xs = numpy.linspace (our.p_min, our.p_max, our.p_cnt); 
		our.probe_ys = numpy.linspace (our.p_min, our.p_max, our.p_cnt); 
		our.probe_dx = our.probe_xs[1] - our.probe_xs[0]; 
		our.probe_dy = our.probe_ys[1] - our.probe_ys[0]; 
		# mesh_coord is for using the pcolormesh () from pyplot ... 
		our.probe_mesh_coord_x = numpy.append (our.probe_xs - our.probe_dx / 2, our.probe_xs[-1] + our.probe_dx / 2); 
		our.probe_mesh_coord_y = numpy.append (our.probe_ys - our.probe_dy / 2, our.probe_ys[-1] + our.probe_dy / 2); 
		our.probe_lenX = len (our.probe_xs); 
		our.probe_lenY = len (our.probe_ys); 


# OptimizationParameters 
# 
# A class to hold parameters such as where to 
# start or stop probing the magnetic field for 
# optimization. 
# 

class OptimizationParameters (ProbeParameters): 
	def __init__ (our): 
		our.p_cnt = 15; 


# PlotParameters 
# 
# A class to describe what and how to plot. 
# 

class PlotParameters (ProbeParameters): 
	r_cnt = 131; 
	slice_rs = []; 
	avl_colors = numpy.array (['b', 'g', 'r', 'c', 'm', 'y', 'k', \
		'b', 'g', 'r', 'c', 'm', 'y', 'k']); 
	def __init__ (our): 
		our.p_cnt = 81; # More points for showing. 
	def setBounds (our, r_min, r_max): 
		super (PlotParameters, our).setBounds (r_min, r_max); 
		our.slice_rs = numpy.linspace (r_min, r_max, our.r_cnt); 


# RingContext 
# 
# A class to describe a cell system's setup with rings. 
# The only things this context does not describe is 
# the actual water cell's limitations or dimensions, 
# optimization parameters, and optimization results 
# (i.e., the "state" vector). 
# 

class RingContext: 
	class ShapeIndexSelector: 
		R = 1; 
		M = 2; 
		X = 4; 
		Y = 8; 
	SHAPE = ShapeIndexSelector (); 
	# State initial values: 
	magnet_ring_radii = []; 
	zero_delta_angles = []; 
	h0 = []; 
	# Rows and columns, for the grid size: 
	M_rows = 0; 
	M_cols = 0; 
	# Multidimensional grids for use everywhere in the code: 
	#I_use = []; 
	#R_use = []; 
	#M_use = []; 
	#A_use = []; 
	I_grid = []; 
	R_grid = []; 
	M_grid = []; 
	A_grid = []; 
	# Methods: 
	def make_h0 (our, height = 1e1): 
		return height * numpy.ones_like (our.magnet_ring_radii); 
	def init_radii (our, m_min, m_max, count = 0): 
		m_count = count; 
		if (m_count == 0): 
			m_count = len (our.magnet_ring_radii); 
		# Linearly initialize the magnet ring radii: 
		our.magnet_ring_radii = \
			numpy.linspace (m_min, m_max, m_count); 
	def init_grids (our, cell_params): 
		# Get counts and orientations: 
		S_up = cell_params.S_up; 
		M_num = cell_params.M_num; 
		# Initialize magnet ring radii: 
		our.init_radii (0, 1, len (M_num)); 
		# Make an array of 0s: 
		our.zero_delta_angles = numpy.zeros_like (our.magnet_ring_radii); 
		# Fill out some heights: 
		our.h0 = our.make_h0 (); 
		# Find the maximum number per ring, and use that as a grid dimension: 
		if len (M_num) > 0: 
			n_magnets_max_per_ring = M_num.max (); 
		else: 
			n_magnets_max_per_ring = 0; 
		# Initialize some 2D grids: 
		our.M_rows = M_rows = len (M_num); 
		our.M_cols = M_cols = n_magnets_max_per_ring; 
		I_grid = numpy.zeros ((M_rows, M_cols)); # Just a grid of 1s and 0s for whether there is a magnet or not. 
		R_grid = numpy.zeros ((M_rows, M_cols), dtype = int); # A "row number" grid of row numbers. 
		A_grid = numpy.zeros ((M_rows, M_cols)); # Rotation angles. 
		M_grid = numpy.zeros ((M_rows, M_cols)); # Magnetization grid. 
		for ring in range (0, len (M_num)): 
			I_grid[ring, 0 : M_num[ring]] = 1; 
			R_grid[ring, 0 : M_num[ring]] = ring; 
			M_grid[ring, 0 : M_num[ring]] = S_up[ring] * M; # Orientation times magnetization. 
			A_grid[ring, 0 : M_num[ring]] = numpy.linspace (0, 2 * numpy.pi, M_num[ring], endpoint = False); 
		# Use this 'for' loop only once, and then all the rest is 
		# numpy numerical processing using grids (no more loops!). 
		our.I_grid = I_grid; 
		our.R_grid = R_grid; 
		our.A_grid = A_grid; 
		our.M_grid = M_grid; 
		# Axes: (probe X, probe Y, which ring, which magnet) 
	def setM (our, M): 
		for ring in range (0, our.M_rows): 
			our.M_grid[ring, our.M_grid[ring] != 0] = our.cell_params.S_up[ring] * M; 
	def getM (our, shape_X, shape_Y): 
		# In order to be able to use the powerful numerical processing 
		# of numpy, we have to reshape things into multiple dimensions: 
		return our.M_grid.reshape (our.keep_shape_comp (\
				our.find_shape_0_by_shape (shape_X, shape_Y), \
			    our.SHAPE.M | our.SHAPE.R)); 
	def getA (our, shape_X, shape_Y): 
		return our.A_grid.reshape (our.keep_shape_comp (\
				our.find_shape_0_by_shape (shape_X, shape_Y), \
			    our.SHAPE.M | our.SHAPE.R)); 
	def getR (our, shape_X, shape_Y): 
		return our.R_grid.reshape (our.keep_shape_comp (\
				our.find_shape_0_by_shape (shape_X, shape_Y), \
			    our.SHAPE.M | our.SHAPE.R)); 
	def getI (our, shape_X, shape_Y): 
		return our.I_grid.reshape (our.keep_shape_comp (\
				our.find_shape_0_by_shape (shape_X, shape_Y), \
			    our.SHAPE.M | our.SHAPE.R)); 
	# State: 
	def get_hs (our, state): 
		n = len (state) / 3; 
		N = int (n); 
		if n != N: 
			throw ("Invalid state vector length!"); 
		return state[0: N]; 
	def get_rs (our, state): 
		n = len (state) / 3; 
		N = int (n); 
		if n != N: 
			throw ("Invalid state vector length!"); 
		return state[N : 2 * N]; 
	def get_da (our, state): 
		n = len (state) / 3; 
		N = int (n); 
		if n != N: 
			throw ("Invalid state vector length!"); 
		return state[2 * N : 3 * N]; 
	def make_state (our, hs, rs, dA): 
		N = len (hs); 
		if len (rs) != N or len (dA) != N: 
			throw ("BAD EXCEPTION!"); 
		state = numpy.zeros ((3 * N,)); 
		state[0 : N] = hs[0 : N]; 
		state[N : 2 * N] = rs[0 : N]; 
		state[2 * N : 3 * N] = dA[0 : N]; 
		return state; 
	def find_shape_0 (our, xs, ys): 
		return our.find_shape_0_by_shape (xs.shape, ys.shape); 
	def find_shape_0_by_shape (our, shape_X, shape_Y): 
		shape_R = (len (our.h0),); 
		shape_M = (our.M_cols,); 
		return shape_R + shape_M + shape_X + shape_Y; 
	def keep_shape_comp (our, shape_0 = None, filter_what = SHAPE.M): 
		xy_len = len (shape_0) - 2; 
		half_len = int (xy_len / 2); 
		x_range = y_range = (2, 2 + half_len); # Assume overlap between X, Y shape. 
		return tuple (max (shape_0[i], shape_0[i + half_len]) if (i in range (*x_range) and \
							filter_what & our.SHAPE.X and filter_what & our.SHAPE.Y) else \
						shape_0[i] if (i in range (*x_range) and filter_what & our.SHAPE.X) or \
						(i in range (1, 2) and filter_what & our.SHAPE.M) or \
						(i in range (0, 1) and filter_what & our.SHAPE.R) \
					else (shape_0[i + half_len] if (i in range (*y_range) \
						and filter_what & our.SHAPE.Y) else 1) \
					for i in range (0, 2 + half_len)); 
	def find_x_y_r_cos_sin (our, state, A, xs, ys): 
		shape_0 = our.find_shape_0 (xs, ys); 
		rs = our.get_rs (state); 
		dA_linear = our.get_da (state); 
		dA = dA_linear.reshape (our.keep_shape_comp (shape_0, our.SHAPE.R)); 
		angle = -(A + dA); 
		r = rs.reshape (our.keep_shape_comp (shape_0, our.SHAPE.R)); 
		x = xs.reshape (our.keep_shape_comp (shape_0, our.SHAPE.X)); 
		y = ys.reshape (our.keep_shape_comp (shape_0, our.SHAPE.Y)); 
		cosA = numpy.cos (angle); 
		sinA = numpy.sin (angle); 
		return x, y, r, cosA, sinA; 
	# Generation of the x, y, z, and r parameters for the dipole B-field equations: 
	def get_xy_intermediate (our, state, A, xs, ys): 
		x, y, r, cosA, sinA = our.find_x_y_r_cos_sin (state, A, xs, ys); 
		xp = cosA * x - sinA * y; 
		yp = sinA * x + cosA * y; 
		return xp - r, yp, r, cosA, sinA; 
	def get_x (our, state, A, xs, ys): 
		x, y, r, cosA, sinA = our.get_xy_intermediate (state, A, xs, ys); 
		return cosA * x + sinA * y; 
	def get_y (our, state, A, xs, ys): 
		x, y, r, cosA, sinA = our.get_xy_intermediate (state, A, xs, ys); 
		return -sinA * x + cosA * y; 
	def get_z (our, state, A, xs, ys): 
		hs = our.get_hs (state); 
		shape_0 = our.find_shape_0 (xs, ys); 
		return hs.reshape (our.keep_shape_comp (shape_0, our.SHAPE.R)); 


# RingOptimizeContext 
# 
# A class to hold variables and parameters while 
# the optimization code does its minimizing. 
# 

class RingOptimizeContext (RingContext): 
	class Bounds: 
		heights = radii = angles = (); 
		def __init__ (our): 
			our.heights = (); 
			our.angles = (); 
			our.radii = (); 
	class DummyLastResult: 
		x = []; 
	PARAM = OptimizeWhat (); 
	METHOD = FieldTargetMethod (); 
	cell_params = {}; 
	optimize_params = {}; 
	b_field_calc = {}; 
	desired_strength = (profile_Keplerian,); 
	min_rs = []; 
	last_result = DummyLastResult (); 
	raw_last_result = DummyLastResult (); 
	def __init__ (our, cell_parameters, optimize_parameters): 
		optimize_parameters.readCellParameters (cell_parameters); 
		our.cell_params = cell_parameters; 
		our.optimize_params = optimize_parameters; 
		our.b_field_calc = MagneticFieldCalculation (our); 
		our.init_grids (our.cell_params); 
		our.last_result.x = our.make_state (our.h0, our.magnet_ring_radii, our.zero_delta_angles); # Initial value. 
		return; 
	def update (our): 
		our.init_grids (our.cell_params); 
		our.init_radii (our.cell_params.r_min, our.cell_params.r_max); 
		our.calc_min_rs (); 
	def calc_min_rs (our): 
		S = chord_length (R = our.cell_params.r_min, r = magnet_radius, d = our.cell_params.r_min); 
		dR = S + our.cell_params.m_min_sep; 
		min_circ = numpy.maximum (our.cell_params.M_num * dR, our.cell_params.r_min); 
		our.min_rs = min_circ / (2 * numpy.pi); 
		return; 
	def get_all_bounds (our): 
		our.init_radii (our.cell_params.r_min, our.cell_params.r_max); 
		our.calc_min_rs (); 
		b_hs = tuple ((our.cell_params.m_height_min, our.cell_params.m_height_max) for i in range (0, len (our.cell_params.S_up))); 
		b_rs = tuple ((our.min_rs[i], our.cell_params.m_constrain_max) for i in range (0, len (our.cell_params.S_up))); 
		b_as = tuple ((0, numpy.pi / (2 * our.cell_params.M_num[i])) for i in range (0, len (our.cell_params.S_up))); 
		bounds = RingOptimizeContext.Bounds (); 
		bounds.heights = b_hs; 
		bounds.radii = b_rs; 
		bounds.angles = b_as; 
		return bounds; 
	def get_params_from_state_param (our, state_param, init_state, optimize_what): 
		i = 0; 
		s = int (len (init_state) / 3); 
		if optimize_what & our.PARAM.HS: 
			hs = state_param[i : i + s]; 
			i += s; 
		else: 
			hs = our.get_hs (init_state); 
		if optimize_what & our.PARAM.RS: 
			rs = state_param[i : i + s]; 
			i += s; 
		else: 
			rs = our.get_rs (init_state); 
		if optimize_what & our.PARAM.DA: 
			da = state_param[i : i + s]; 
			i += s; 
		else: 
			da = our.get_da (init_state); 
		return hs, rs, da; 
	def find_profile (our, desired_strength = profile_Keplerian, x0 = None, \
					optimize_what = PARAM.HS + PARAM.RS + PARAM.DA, \
					method = METHOD.B_MAX): 
		our.init_radii (our.cell_params.r_min, our.cell_params.r_max); 
		our.desired_strength = (desired_strength,); 
		our.calc_min_rs (); 
		b_hs = tuple ((our.cell_params.m_height_min, our.cell_params.m_height_max) for i in range (0, len (our.h0))); 
		b_rs = tuple ((our.min_rs[i], our.cell_params.m_constrain_max) for i in range (0, len (our.magnet_ring_radii))); 
		b_as = tuple ((0, numpy.pi / (2 * our.cell_params.M_num[i])) for i in range (0, len (our.zero_delta_angles))); 
		b = (); 
		if optimize_what & our.PARAM.HS: 
			b += b_hs; 
		if optimize_what & our.PARAM.RS: 
			b += b_rs; 
		if optimize_what & our.PARAM.DA: 
			b += b_as; 
		init_state = x0; 
		if init_state is None: 
			init_state = our.last_result.x; 
		init_lists = (); 
		now_hs = our.get_hs (init_state); 
		now_rs = our.get_rs (init_state); 
		now_da = our.get_da (init_state); 
		if optimize_what & our.PARAM.HS: 
			init_lists += (now_hs,); 
		if optimize_what & our.PARAM.RS: 
			init_lists += (now_rs,); 
		if optimize_what & our.PARAM.DA: 
			init_lists += (now_da,); 
		init_param = numpy.concatenate (init_lists); 
		our.raw_last_result = scipy.optimize.minimize (to_minimize_inZ, \
					init_param, \
					(our, our.b_field_calc, init_state, optimize_what, method), \
			bounds = b); 
		our.last_result.x = our.make_state (*our.get_params_from_state_param (our.raw_last_result.x, init_state, optimize_what)); 
		return our.last_result; 


class RingResultPlotter: 
	plot_params = {}; 
	optimize_context = {}; 
	b_calc = {}; 
	def __init__ (our, optimize_context, plot_params): 
		our.optimize_context = optimize_context; 
		our.plot_params = plot_params; 
		our.b_calc = MagneticFieldCalculation (optimize_context); 
	def plot3 (our, state): 
		our.plot3a (state); 
		our.plot3b (state); 
		our.plot3c (state); 
	def plot4 (our, state): 
		our.plot4b (state); 
	def plot3a (our, state, save_as = "desired-color-mesh.png", save_dir = "Optimized-Plots"): 
		ppar = our.plot_params; 
		octx = our.optimize_context; 
		pp.figure (figsize = (4, 4)); 
		pp.title ("Desired Profile"); 
		pp.xlabel ("Measurement Position X (mm)"); 
		pp.ylabel ("Measurement Position Y (mm)"); 
		pp.xlim (ppar.p_min, ppar.p_max); 
		pp.ylim (ppar.p_min, ppar.p_max); 
		grid_r = get_r_2d (ppar.probe_xs, ppar.probe_ys); 
		fxn = octx.desired_strength[0]; 
		Bz = our.b_calc.net_field_z (state, ppar.probe_xs.reshape ((-1, 1)), \
									ppar.probe_ys.reshape ((1, -1))); 
		desired_grid = fxn (grid_r, Bz, octx, octx.METHOD.PROFILE); 
		desired_grid[grid_r < ppar.donut_minR] = 0; 
		desired_grid[grid_r > ppar.donut_maxR] = 0; 
		pp.pcolormesh (ppar.probe_mesh_coord_x, ppar.probe_mesh_coord_y, desired_grid, label = "Desired Profile"); 
		if not os.path.exists (save_dir): 
			os.makedirs (save_dir); 
		pp.savefig (save_dir + "/" + save_as); 
	def plot3b (our, state, save_as = "actual-color-mesh.png", save_dir = "Optimized-Plots"): 
		ppar = our.plot_params; 
		Bz = our.b_calc.net_field_z (state, ppar.probe_xs.reshape ((-1, 1)), \
									ppar.probe_ys.reshape ((1, -1))); 
		pp.figure (figsize = (4, 4)); 
		pp.title ("Actual Theoretical"); 
		pp.xlabel ("Measurement Position X (mm)"); 
		pp.ylabel ("Measurement Position Y (mm)"); 
		pp.xlim (ppar.p_min, ppar.p_max); 
		pp.ylim (ppar.p_min, ppar.p_max); 
		grid_r = get_r_2d (ppar.probe_xs, ppar.probe_ys); 
		Bz[grid_r < ppar.donut_minR] = 0; 
		Bz[grid_r > ppar.donut_maxR] = 0; 
		pp.pcolormesh (ppar.probe_mesh_coord_x, ppar.probe_mesh_coord_y, Bz); 
		if not os.path.exists (save_dir): 
			os.makedirs (save_dir); 
		pp.savefig (save_dir + "/" + save_as); 
	def plot3c (our, state, save_as = "desired-vs-field-slice.png", save_dir = "Optimized-Plots"): 
		ppar = our.plot_params; 
		octx = our.optimize_context; 
		using_xs = ppar.slice_rs.reshape ((-1, 1)); 
		using_ys = numpy.array ([0]).reshape ((1, -1)); 
		Bz = our.b_calc.net_field_z (state, using_xs, using_ys); 
		Bz_slice = Bz[:, 0]; 
		pp.figure (); 
		pp.title ("Desired vs Actual - Radial Profile"); 
		pp.xlabel ("Radial Position (mm)"); 
		pp.ylabel ("Field Strength (gauss)"); 
		pp.xlim (ppar.slice_rs.min (), ppar.slice_rs.max ()); 
		fxn = octx.desired_strength[0]; 
		pp.plot (ppar.slice_rs, fxn (get_r_2d (using_xs, using_ys), \
								Bz, octx, octx.METHOD.PROFILE), label = "Desired"); 
		pp.plot (ppar.slice_rs, Bz_slice, label = "Actual"); 
		pp.legend (loc = "best"); 
		if not os.path.exists (save_dir): 
			os.makedirs (save_dir); 
		pp.savefig (save_dir + "/" + save_as); 
	# This function draws circles of how high each magnet ring is: 
	def plot4b (our, state, save_as = "ring-heights.png", save_dir = "Optimized-Plots"): 
		octx = our.optimize_context; 
		hs = octx.get_hs (state); 
		rs = octx.get_rs (state); 
		fig = pp.figure (); 
		pp.title ("Magnet Ring Heights"); 
		if len (hs) > 0: 
			h_max = hs.max (); 
		else: 
			h_max = 0; 
		ax1 = fig.add_subplot (111, aspect='equal'); 
		
		for i in range (0, min (len (rs), len (hs))): 
			# Adapted from http://matthiaseisen.com/pp/patterns/p0203/ code: 
			ax1.add_patch ( 
			    patches.Rectangle ( 
			        (rs[i] - magnet_radius, 2 * h_max - hs[i] - magnet_height / 2),   # (x,y)
			        2 * magnet_radius,          # width
			        magnet_height,          # height
			    ) 
			); 
		pp.plot (rs, 2 * h_max - hs, 'ro', label = "Ring Heights"); 
		if not os.path.exists (save_dir): 
			os.makedirs (save_dir); 
		pp.savefig (save_dir + "/" + save_as); 


# MagneticFieldCalculation 
# 
# A class for doing the actual magnetic field calculations. 
# Used by the optimization functions in this module. 
# 

class MagneticFieldCalculation: 
	context = {}; 
	def __init__ (our, b_calc_context): 
		our.context = b_calc_context; 
	def inshape_getXYZ (our, state, xs, ys): 
		return our.context.get_x (state, our.context.getA (xs.shape, ys.shape), xs, ys), \
			  our.context.get_y (state, our.context.getA (xs.shape, ys.shape), xs, ys), \
			  our.context.get_z (state, our.context.getA (xs.shape, ys.shape), xs, ys); 
	def inshape_getArgs (our, state, xs, ys): 
		return (our.context.getM (xs.shape, ys.shape),) + our.inshape_getXYZ (state, xs, ys); 
	# The functions that return the B-field grids in the correct array shapes (the returned arrays are "in shape"): 
	def inshape_fieldX (our, state, xs, ys): 
		return B_x (*our.inshape_getArgs (state, xs, ys)); 
	def inshape_fieldY (our, state, xs, ys): 
		return B_y (*our.inshape_getArgs (state, xs, ys)); 
	def inshape_fieldZ (our, state, xs, ys): 
		return B_z (*our.inshape_getArgs (state, xs, ys)); 
	
	# This function returns a grid of the net X,Y,Z magnetic field: 
	def net_field_xyz (our, state, xs, ys): 
		args = our.inshape_getArgs (state, xs, ys); 
		netX = B_x (*args).sum (axis = 1).sum (axis = 0); 
		netY = B_y (*args).sum (axis = 1).sum (axis = 0); 
		netZ = B_z (*args).sum (axis = 1).sum (axis = 0); 
		return numpy.array ([netX, netY, netZ]); 
	
	# These functions return just a 2D grid of the magnetic field: 
	def net_field_x (our, state, xs, ys): 
		return our.inshape_fieldX (state, xs, ys).sum (axis = 1).sum (axis = 0); 
	def net_field_y (our, state, xs, ys): 
		return our.inshape_fieldY (state, xs, ys).sum (axis = 1).sum (axis = 0); 
	def net_field_z (our, state, xs, ys): 
		return our.inshape_fieldZ (state, xs, ys).sum (axis = 1).sum (axis = 0); 
	def net_field_abs (our, state, xs, ys): 
		return numpy.sqrt ((our.net_field_xyz (state, xs, ys) ** 2).sum (axis = 0)); 





# B-field measurement for a magnetic dipole of magnetization M: 
def B_x (M, x, y, z): 
	r = xyz_to_r (x, y, z); 
	return 3 * M * x * z / r ** 5; # B_x 
def B_y (M, x, y, z): 
	r = xyz_to_r (x, y, z); 
	return 3 * M * y * z / r ** 5; # B_y 
def B_z (M, x, y, z): 
	r = xyz_to_r (x, y, z); 
	return M * (3 * z ** 2 - r ** 2) / r ** 5; # B_z 






# Miscellaneous functions: 

def xyz_to_r (x, y, z): 
	return numpy.sqrt (x * x + y * y + z * z); 

def get_r_2d (x, y): 
	return numpy.sqrt (x.reshape ((len (x), 1)) ** 2 + y.reshape ((1, len (y))) ** 2); 

def norm (x): 
	return numpy.sqrt (numpy.sum (x * x)); 

def to_scalar (x): 
	return norm (x); 



# Functions for doing the actual minimization cost function evaluations: 

# get_sterics_grid (): 
# 
# The code to help the check_sterics () function, below. 
# 
# See the check_sterics () documentation below to find out 
# why these two functions were implemented in the first place, 
# and why they are not being used now. 
# 
def get_sterics_grid (state, optimize_context): 
	# Checks how close magnets are to each other, etc. 
	hs = optimize_context.get_hs (state); 
	rs = optimize_context.get_rs (state); 
	sHs = len (hs); 
	sRs = len (rs); 
	needDR = optimize_context.cell_params.minDR; 
	needDZ = optimize_context.cell_params.minDZ; 
	overlap_hs = numpy.abs (hs.reshape ((sHs, 1)) - hs.reshape ((1, sHs))); 
	overlap_rs = numpy.abs (rs.reshape ((sRs, 1)) - rs.reshape ((1, sRs))); 
	gaussian_hs = n.pdf (overlap_hs / needDZ); 
	gaussian_rs = n.pdf (overlap_rs / needDR); 
	g_both = gaussian_hs * gaussian_rs; 
	return g_both; 

# check_sterics (): 
# 
# The idea was, this function would check how much the magnets overlap 
# each other in space, and return a very small value, near 0, if they 
# all fit just fine, but return a larger value if they overlap or are 
# too close to each other. This was back when Ruvim and the Borrero Lab 
# trusted the Python's application-blind optimization algorithm. 
# 
# This code has not worked for us, so we replaced it with a user-optimize 
# program, which asks the user to move sliders and position magnets - 
# the user is a lot smarter than the algorithm, so we believe it was a 
# good choice for us to make that transition. 
# 
def check_sterics (state, optimize_context): 
	g_both = get_sterics_grid (state, optimize_context); 
	numpy.fill_diagonal (g_both, 0); # Diagonals show self-interactions of rings; no need! 
	quotient = g_both.shape[0]; 
	corrected = (g_both * 1e3 / (n.pdf (1) * quotient)); 
	return corrected.sum (); 

# grid_to_minimize_z (): 
# 
# Returns a grid of Bz(actual) - Bz(needed) values. Used for automatic 
# (yet application-blind) optimization in Python. 
# 
def grid_to_minimize_z (state, optimize_context, b_field_context, method): 
	opp = optimize_context.optimize_params; 
	r = get_r_2d (opp.probe_xs, opp.probe_ys); 
	actual = b_field_context.net_field_z (state, opp.probe_xs, opp.probe_ys); 
	desired = optimize_context.desired_strength[0] (r, actual, optimize_context, method); 
	grid =  desired - actual; 
	grid[r < opp.donut_minR] = 0; 
	grid[r > opp.donut_maxR] = 0; 
	return grid; 

# to_minimize_inZ (): 
# 
# Takes the result of grid_to_minimize_z (), and turns it into a scalar 
# for the scipy.optimize.minimize () algorithm to know how well the 
# current state is optimized. 
# 
def to_minimize_inZ (state_param, optimize_context, b_field_context, init_state, optimize_what, method): 
	params = optimize_context.get_params_from_state_param (state_param, init_state, optimize_what); 
	state = optimize_context.make_state (*params); 
	return to_scalar (grid_to_minimize_z (state, optimize_context, b_field_context, method)); # + \
		#check_sterics (state, optimize_context); 

# grid_to_minimize_abs (): 
# 
# Same as grid_to_minimize_z (), but for optimizing the magnetic field 
# *magnitude* rather than just the Z component. 
# 
def grid_to_minimize_abs (state, optimize_context, b_field_context, method): 
	opp = optimize_context.optimize_params; 
	r = get_r_2d (opp.probe_xs, opp.probe_ys); 
	actual = b_field_context.net_field_abs (state, opp.probe_xs, opp.probe_ys); 
	desired = optimize_context.desired_strength[0] (r, actual, optimize_context, method); 
	grid = desired - actual; 
	grid[r < opp.donut_minR] = 0; 
	grid[r > opp.donut_maxR] = 0; 
	return grid; 

# to_minimize_inAbs (): 
# 
# Same as to_minimize_inZ (), but for optimizing the magnetic field 
# *magnitude* rather than just the Z component. 
# 
def to_minimize_inAbs (state_param, optimize_context, b_field_context, init_state, optimize_what, method): 
	params = optimize_context.get_params_from_state_param (state_param, init_state, optimize_what); 
	state = optimize_context.make_state (*params); 
	return to_scalar (grid_to_minimize_abs (state, optimize_context, b_field_context, method)); # + \
		#check_sterics (state, optimize_context); 

# Again, these functions are useful for having the computer blindly optimize 
# the magnet arrangement, but this application-blind computer-run process 
# itself is not very trustworthy as of right now, because computers are not 
# smart by themselves - it takes clever code to make them smart - but we did 
# not have the time to invest into making an artificial intelligence program 
# - this was a physics project, after all, not a computer science project. 




