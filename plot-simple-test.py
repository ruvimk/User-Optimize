# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:13:23 2017

@author: Borrero Lab
"""

import os, sys; 

sys.path.append (os.path.abspath (".")); 
import ring_optimize as ro; 


program = ro.SimpleSetup (); 
program.set_up (); 

program.plot (); 


