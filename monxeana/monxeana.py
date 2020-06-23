
# This Python3 library contains code focussing on the analysis of MonXe stuff.



#######################################
### Imports
#######################################


import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import pprint
import os
import random
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox





#######################################
### Generic Definitions
#######################################


input_pathstring = "./Input/"
output_pathstring = "./Output/"


# colors
uni_blue = '#004A9B'
uni_red = '#C1002A'




