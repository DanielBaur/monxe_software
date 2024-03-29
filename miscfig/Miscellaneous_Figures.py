

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
from scipy import ndimage
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox





#######################################
### Generic Definitions
#######################################


input_pathstring = "./input/"
output_pathstring = "./output/"
pathstring_thesis_images = "/home/daniel/Desktop/arbeitsstuff/20200720__thesis/images/miscfigimages/"
abspath_mastertalk_images = "/home/daniel/Desktop/arbeitsstuff/20210416__mastertalk/images/"
abspath_miscfig = "/home/daniel/Desktop/arbeitsstuff/monxe/software/miscfig/"
relpath_input = "./input/"
relpath_output = "./output/"


# colors
uni_blue = '#004A9B'
uni_red = '#C1002A'
colorstring_gemsemint = "#5a8fa3"
colorstring_citirok1 = "#007990" # RGB: 0, 120, 144
colorstring_citirok2 = "#024e7b" # ""
colorstring_citirok3 = "#024167" # "blue"
colorstring_citirok4 = "#1d7968" # "green"
colorstring_petrol = "#22555e" # randomly color picked


colorstring_darwin_blue = '#004A9B'

standard_miscfig_scale = 160
standard_figsize = (5.670, 3.189) # corresponds to full thesis textwidth in inch in 16:9 format
standard_figsize_x_inch = 5.670 # corresponds to full thesis textwidth in inch

standard_width = 160

image_format_dict = {
    "16_9" : {
        "figsize" : standard_figsize,
        "axes" : (standard_width, 9/16 *standard_width)
    },
    "monxe_thesis_photograph" : {
        "figsize" : [standard_figsize[0], standard_figsize[1]],
        "axes" : (standard_width, 9/16 *standard_width)
    },
    "talk" : {
        "figsize" : (standard_figsize[0], standard_figsize[1]*(75/90)),
        "axes" : (standard_width, 7.5/16 *standard_width)
    },
    "talk_slim" : {
        "figsize" : (standard_figsize[0], standard_figsize[1]*(65/90)),
        "axes" : (standard_width, 6.5/16 *standard_width)
    },
    "paper" : {
        "figsize" : (standard_figsize[0], standard_figsize[1]*(75/90)),
        "axes" : (standard_width, 7.5/16 *standard_width)
    }
}


def invproplinfunc(x, m, t):
    return m*(1/x) +t


def proplinfunc(x, m, t):
    return m*x +t

    
def get_y_from_x_on_parabola(x,xy_vertex,xy_ref):
    """
    This function returns the y value of a parabola with vertex 'xy_vertex' and reference point 'xy_ref' for a given x-value 'x'.
    reminder: y = const *(x-x_vertex)^2 +y_vertex
    """
    const = (xy_ref[1]-xy_vertex[1])/(xy_ref[0]-xy_vertex[0])**2
    return const *(x-xy_vertex[0])**2 +xy_vertex[1]

        
# biased: data generation
def get_vertex_y_from_m_on_parabola(
    m,
    x_vertex,
    xy_ref,
):
    """
    This function returns the y-value of the vertex point of a parabola for a specified reference point on the wanted parabola 'xy_ref' and the slope at that point 'm'.
    reminder: y = const *(x-x_vertex)^2 +y_vertex
    """
    c = 0.5*m/(xy_ref[0]-x_vertex)
    y_vertex = xy_ref[1] -(c*(xy_ref[0]-x_vertex)**2)
    return y_vertex, c


# This function is used to retrieve m and t from a function of the form y=m*(1/x)+t.
# Therefore two points (x_1,y_1), (x_2, y_2) lying on the graph are required.
def get_m_and_t_from_invproplinfunc(y_1, x_1, y_2, x_2):
    m = (y_1-y_2)/((1/x_1)-(1/x_2))
    t = y_1 -m*(1/x_1)
    print(f"y = m*(1/x)+t,  with m={m} and t={t}")
    return m, t


# This function is used to retrieve m and t from a function of the form y= m*x +t.
# Therefore two points (x_1,y_1), (x_2, y_2) lying on the graph are required.
def get_m_and_t_from_proplinfunc(y_1, x_1, y_2, x_2):
    m = (y_1-y_2)/(x_1-x_2)
    t = y_1 -m*x_1
    print(f"y = m*x+t,  with m={m} and t={t}")
    return m, t


# function to return the sign of a float or int (i.e. +1 or -1)
def signum(number):
    if number > 0:
        return +1
    elif number < 0:
        return -1
    elif number == 0:
        return 0
    else:
        print("You did not insert a number. Accordingly the sign cannot be computed.")


# function to return datestring (e.g.: 20190714)
def datestring():
    return str(datetime.datetime.today().year) + str(datetime.datetime.today().month).zfill(2) + str(datetime.datetime.today().day).zfill(2)


# function to return timestring (e.g.: 1725 for 17:25h)
def timestring():
    return str(datetime.datetime.now().time().hour).zfill(2) + str(datetime.datetime.now().time().minute).zfill(2)


def convert_string_to_ordinal(input_string):
    if input_string[-1] in ["1"]:
        ordinal_string = "st"
    elif input_string[-1] in ["2"]:
        ordinal_string = "nd"
    elif input_string[-1] in ["3"]:
        ordinal_string = "rd"
    else:
        ordinal_string = "th"
    return ordinal_string


def convert_int_to_month_string(input_int, flag_output=["October", "Oct", "Oktober", "Okt"][1]):
    conversion_dict = {
        "1" : {
            "October" : "January",
            "Oct" : "Jan",
            "Oktober" : "Januar",
            "Okt" : "Jan",
        },
        "2" : {
            "October" : "February",
            "Oct" : "Feb",
            "Oktober" : "Februar",
            "Okt" : "Feb",
        },
        "3" : {
            "October" : "March",
            "Oct" : "Mar",
            "Oktober" : "März",
            "Okt" : "Mär",
        },
        "4" : {
            "October" : "April",
            "Oct" : "Apr",
            "Oktober" : "April",
            "Okt" : "Apr",
        },
        "5" : {
            "October" : "May",
            "Oct" : "May",
            "Oktober" : "Mai",
            "Okt" : "Mai",
        },
        "6" : {
            "October" : "June",
            "Oct" : "Jun",
            "Oktober" : "Juni",
            "Okt" : "Jun",
        },
        "7" : {
            "October" : "July",
            "Oct" : "Jul",
            "Oktober" : "Juli",
            "Okt" : "Jul",
        },
        "8" : {
            "October" : "August",
            "Oct" : "Aug",
            "Oktober" : "August",
            "Okt" : "Aug",
        },
        "9" : {
            "October" : "September",
            "Oct" : "Sep",
            "Oktober" : "September",
            "Okt" : "Sep",
        },
        "10" : {
            "October" : "October",
            "Oct" : "Oct",
            "Oktober" : "Oktober",
            "Okt" : "Okt",
        },
        "11" : {
            "October" : "November",
            "Oct" : "Nov",
            "Oktober" : "November",
            "Okt" : "Nov",
        },
        "12" : {
            "October" : "December",
            "Oct" : "Dec",
            "Oktober" : "Dezember",
            "Okt" : "Dez",
        },
    }
    return conversion_dict[str(input_int)][flag_output]


# function to generate an output list that can be plotted
def generate_plot_list(x_list,func,**kwargs):
    func_list = np.zeros_like(x_list)
    for i in range(len(func_list)):
        func_list[i] = func(t_list[i], **kwargs)
    return func_list
# EXAMPLE: N_4_ana_list = generate_plot_list(x_list=t_list, func=N_4_ana, l1=l1, l2=l2, l3=l3, l4=l4, N_1_0=N_1_0, N_2_0=N_2_0, N_3_0=N_3_0, N_4_0=N_4_0)


# function to read in a .csv file that is generated via the Web Plot Digitizer online application (https://automeris.io/WebPlotDigitizer/) and write the data into two lists
def wpd_to_lists(filestring, pathstring=input_pathstring):
    file = open(pathstring +filestring)
    x_data = []
    y_data = []
    for line in file:
        row = line.strip().split(',')
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
    return x_data, y_data
# EXAMPLE: xl, yl = wpd_to_lists(filestring="dm_sensitivity_plot_xenon1t_data.csv")


# This function is used to calculate the vector sum (vs) of a 2tuple.
def vs(p1, p2, sn=False):
    if sn == False:
        p3 = (p1[0]+p2[0], p1[1]+p2[1])
    elif sn == True:
        p3 = (p1[0]-p2[0], p1[1]-p2[1])
    else:
        p3 = (0,0)
    return p3


# function to calculate the distance between two points given as two tuples
def vecdist(p1,p2):
    vec = vs(p1,(-p2[0],-p2[1]))
    length = np.sqrt(vec[0]**2+vec[1]**2)
    return length


# function to scale a vector to length 1
def norm_vec(two_tuple_vector):
    length = np.sqrt(two_tuple_vector[0]**2+two_tuple_vector[1]**2)
    norm_two_tuple_vector = (two_tuple_vector[0]/length, two_tuple_vector[1]/length)
    return norm_two_tuple_vector


# function to scale a two tuple vector by a certain length
def scale_vec(two_tuple_vector, lambd):
    return (two_tuple_vector[0]*lambd, two_tuple_vector[1]*lambd)


# function to generate a vector of length 1 that is orthogonal to a line defined by two points
def norm_orth_vec(p1,p2):
    if p1[0]==p2[0]:
        nov = (-1,0)
    else:
        v = vs(p1,p2,sn=True)
        ov = (-v[1]/v[0],1)
        nov = norm_vec(ov)
    return nov


# This function is used to annotate a label onto a plot that is sitting in between two arrows.
def draw_label_at_foot_of_arrows(
    text_r,
    text_label,
    text_fontsize,
    text_color,
    arrow_tips,
    arrow_gap,
    arrow_linewidth,
    arrow_linecolor,
    arrow_tipwidth,
    arrow_tiplength,
    zorder,
    ax):
    # inferring the arrow feet
    arrow_feet = []
    for arrow_tip in arrow_tips:
        arrow_foot = list(vs(arrow_tip, [-text_r[0],-text_r[1]]))
        arrow_foot = list(norm_vec(arrow_foot))
        arrow_foot = list(scale_vec(arrow_foot, arrow_gap))
        arrow_foot = list(vs(text_r, arrow_foot))
        arrow_feet.append(arrow_foot)
    # drawing the arrows
    for arrow_coordinates in zip(arrow_feet, arrow_tips):
        plot_arrow_connect_points(
            ax = ax,
            points_list = list(arrow_coordinates),
            linewidth = arrow_linewidth,
            color = arrow_linecolor,
            tip_width = arrow_tipwidth,
            tip_length = arrow_tiplength,
            linestyle = "-",
            flag_single_connections = True,
            input_zorder = zorder)
    # annotating the label
    ax.text(
        s = text_label,
        x = text_r[0],
        y = text_r[1],
        fontsize = text_fontsize,
        color = text_color,
        horizontalalignment = "center",
        verticalalignment = "center")
    return


def significant_figure_latex_string(
    mean, # mean value
    lower, # lower 68th percentile interval
    upper, # upper 68th percentile interval
    n = 1, # number of significant figures of the uncertainty
    decade_shift = 0, # exponent of a factor of 10, by which the output is modified
    flag_braces = ["()", "", "[]", "{}"][0], # what braces to use
    flag_symmetry = [ # display symmetric or asymmetric uncertainty intervals
        "auto", # depending on whether the displayed interval limits would be symmetric or not, a symmetric syntax is printed or not
        "asym", # printing asymmetric uncertainty interval limits
        "sym", # printing symmetric uncertainty interval limits, utilizing the lower uncertainty interval width
        "none", # printing no uncertainty interval limits at all, this might be useful to determine the precision-respecting representation of a number of integer counts
        "single_number", # printing no uncertainty interval limits, instead making uncertain figures zero,  this might be useful to determine the precision-respecting representation of a number of integer counts
        "single_number_auto", # printing no uncertainty interval limits, instead making uncertain figures zero and decade-shift correspondingly,  this might be useful to determine the precision-respecting representation of a number of integer counts
    ][0], # default is 'auto'
    flag_display_decade_shift = [False,True][0], # flag indicating whether or not the decade shift is printed
):
    """
    This function is used to determine the representation of a measurement value, given as mean and lower/upper uncertainty intervals, in terms of significant figures.
    The output is a latex-formatted string that I use to annotate plots.
    NOTE: This function assumes that the lower uncertainty interval width is lower than that of the upper uncertainty interval.
    general: https://www.physics.uoguelph.ca/significant-digits-tutorial#:~:text=Zeroes%20placed%20before%20other%20digits,7.90%20has%20three%20significant%20digits.
    general: https://www2.southeastern.edu/Academics/Faculty/rallain/plab194/error.html#:~:text=Uncertainties%20are%20almost%20always%20quoted,example%3A%20%C2%B10.0012%20kg).&text=Always%20round%20the%20experimental%20measurement,decimal%20place%20as%20the%20uncertainty.
    """

    # initializing variables
    m = float(mean)/(10**decade_shift)
    l = float(lower)/(10**decade_shift)
    u = float(upper)/(10**decade_shift)

    # writing 'lower' as: 'figures_before_delimiter' +'.' +'sandwiched_zeros_after_delimiter' +'non_zero_figures_after_sandwiched_zeros'
    figures_before_delimiter = list(str(l).split("."))[0] # no unsignificant leading zeroes since l is of type 'float'
    figures_after_delimiter = list(str(l).split("."))[1] # no unsignificant trailing zeroeas since l is of type 'float'
    sandwiched_zeros_after_delimiter = "0"*(len(figures_after_delimiter) -len(figures_after_delimiter.lstrip("0")))
    non_zero_figures_after_sandwiched_zeros = figures_after_delimiter.lstrip("0")
    n_significant_figures_before_delimiter = len(figures_before_delimiter) if figures_before_delimiter!="0" else 0
    n_insignificant_figures_after_delimiter = len(sandwiched_zeros_after_delimiter) if n_significant_figures_before_delimiter==0 else 0
    n_significant_figures_after_delimiter = len(figures_after_delimiter) -n_insignificant_figures_after_delimiter

    # determining the number of decimals to be printed
    if n_significant_figures_before_delimiter==0:
        if n_insignificant_figures_after_delimiter==0:
            n_decimals = n
        elif n_insignificant_figures_after_delimiter>0:
            n_decimals = n_insignificant_figures_after_delimiter +n
    elif n_significant_figures_before_delimiter>0:
        n_decimals = n -n_significant_figures_before_delimiter
        
    # manual decade shift
    #m = m*(10**decade_shift)
    #l = l*(10**decade_shift)
    #u = u*(10**decade_shift)
    #n_decimals -= decade_shift
    
    # case: n_decimals < 0
    force_decade_shift = decade_shift
    if n_decimals<0:
        #while n_decimals<0:
        #    n_decimals+=1
        #    force_decade_shift += 1
        raise Exception(f"sfls(): case catch: the number of specified significant figures of the uncertainty ('n'={n}) is smaller than the number of digits before the decimal point ({n_significant_figures_before_delimiter}), even after the decade shift ('decade_shift'={decade_shift})")
        #print(f"sfls(): case catch: the number of specified significant figures of the uncertainty ('n'={n}) is smaller than the number of digits before the decimal point ({n_significant_figures_before_delimiter}), even after the decade shift ('decade_shift'={decade_shift})")
        #print(f"sfls(): => forcefully increasing decimal shift by '{force_decade_shift-decade_shift}' orders of magnitude (was '{decade_shift}', now '{force_decade_shift}')")
        
    # printing the constituent strings
    m_string = f"{m:.{n_decimals}f}"
    l_string = f"{l:.{n_decimals}f}"
    u_string = f"{u:.{n_decimals}f}"
    decade_shift_string = "" if flag_display_decade_shift==False else r" \cdot 10^{" +f"{force_decade_shift:.0f}" +"}"
    if flag_braces=="":
        braces_string_left = r""
        braces_string_right = r""
    elif flag_braces=="()":
        braces_string_left = r"\left("
        braces_string_right = r"\right)"
    elif flag_braces=="[]":
        braces_string_left = r"\left["
        braces_string_right = r"\right]"
    elif flag_braces=="{}":
        braces_string_left = r"\left{}"
        braces_string_right = r"\right}"
    else:
        raise Exception(f"invalid argument for keyword 'flag_br': {flag_br}")

    # case: only one significant figure which is equal to one
    if l_string[-1]=="1":
        print(f"sfls(): case catch: last significant digit of uncertainty is '1'")
        print(f"sfls(): => increasing number of significant figures by one")
        n_decimals += 1
        m_string = f"{m:.{n_decimals}f}"
        l_string = f"{l:.{n_decimals}f}"
        u_string = f"{u:.{n_decimals}f}"
        
    # catch: no decade shift output but non-zero value of 'decade_shift'
    if flag_display_decade_shift==False and decade_shift!=0:
        print(f"sfls(): WARNING: you chose to print no decade shift though 'decade_shift'={decade_shift}")

    # catch: no braces despite 'flag_display_decade_shift' being 'True'
    if flag_display_decade_shift==True and flag_braces=="":
        print(f"sfls(): WARNING: you chose to not print braces though 'flag_display_decade_shift'={flag_display_decade_shift}")

    # case: 'flag_symmetry'=='singel_number_auto'
    if flag_symmetry=='single_number_auto':
        # determining which index of 'm_string' is uncertain due to the value of 'l_string'
        highest_non_zero_uncertainty_index = len(l_string) -min([l_string.index(non_zero_val) for non_zero_val in l_string.replace("0","").replace(".","")])
        if int(l_string[-highest_non_zero_uncertainty_index])<5:
            round_index = highest_non_zero_uncertainty_index
        else:
            if m_string[-(highest_non_zero_uncertainty_index+1)]==".":
                round_index = highest_non_zero_uncertainty_index +2
            else:
                round_index = highest_non_zero_uncertainty_index +1
        # setting all undetermined digits to zero and rounding the 'round_index'ed number
        m_list = list(m_string)
        if m_list[-round_index+1] == ".":
            round_check_digit = int(m_list[-round_index+2])
        else:
            round_check_digit = int(m_list[-round_index+1])
        if round_check_digit>=5:
            if m_list[-round_index]==9:
                m_list[-round_index] = "0"
                m_list[-(round_index+1)] = str(int(m_list[-(round_index+1)])+1)
            else:
                m_list[-round_index] = str(int(m_list[-round_index])+1)
        m_list_rounded = [m_list[i] if (i<=len(m_list)-round_index or m_list[i]==".") else "0" for i in range(len(m_list))]
        m_string = "".join(m_list_rounded)
        # determining the decade shift exponent
        figures_before_delimiter = list(str(m_string).split("."))[0] # no unsignificant leading zeroes since l is of type 'float'
        figures_after_delimiter = list(str(m_string).split("."))[1] # no unsignificant trailing zeroeas since l is of type 'float'
        sandwiched_zeros_after_delimiter = "0"*(len(figures_after_delimiter) -len(figures_after_delimiter.lstrip("0")))
        non_zero_figures_after_sandwiched_zeros = figures_after_delimiter.lstrip("0")
        n_significant_figures_before_delimiter = len(figures_before_delimiter) if figures_before_delimiter!="0" else 0
        n_insignificant_figures_after_delimiter = len(sandwiched_zeros_after_delimiter) if n_significant_figures_before_delimiter==0 else 0
        n_significant_figures_after_delimiter = len(figures_after_delimiter) -n_insignificant_figures_after_delimiter
        if n_significant_figures_before_delimiter>0:
            decade_shift_exponent = n_significant_figures_before_delimiter-1
        else:
            decade_shift_exponent = n_insignificant_figures_after_delimiter +1
        m_string = str(float(m_string)/(10**decade_shift_exponent))
        decade_shift_string = "" if decade_shift_exponent==0 else r" \cdot 10^{" +f"{decade_shift_exponent}" +"}"

    # setting the 'uncertainty' syntax
    if flag_symmetry=="asym":
        uncertainty_string = r"_{-" +l_string +r"}^{+" +u_string +r"}"
    elif flag_symmetry=="sym":
        uncertainty_string = r"\pm " +l_string
    elif flag_symmetry in ["single_number_auto", "none"]:
        uncertainty_string = r""
    elif flag_symmetry=="auto":
        if l_string == u_string:
            uncertainty_string = r"\pm " +l_string
        else:
            uncertainty_string = r"_{-" +l_string +r"}^{+" +u_string +r"}"
    else:
        raise Exception(f"invalid argument for keyword 'flag_symmetry': {flag_symmetry}")

    # piecing together the output string
    sig_fig_latex_string = r"$" +braces_string_left +m_string +uncertainty_string +braces_string_right +decade_shift_string +r"$"

    return sig_fig_latex_string


# This function is used to plot a line from one 2tuple to another.
def plot_line(start_tuple, end_tuple, linewidth=2, linecolor='black', zorder=5, **kwargs):
    plt.plot( [start_tuple[0],end_tuple[0]], [start_tuple[1],end_tuple[1]], linewidth=linewidth, color=linecolor, zorder=zorder, **kwargs)
    return


# This function is used to connect a list of points with each other utilizing the 'plot_line' function defined above.
def plot_line_connect_points(
    points_list,
    input_ax,
    linewidth = 2,
    linestyle = "-",
    linecolor = "black",
    flag_connect_last_with_first = True,
    flag_single_connections = False,
    input_zorder=30):

    x_list = []
    y_list = []
    # generating the lists of x and y coordinates of the points within the passed list
    for i in range(len(points_list)):
        x_list.append(points_list[i][0])
        y_list.append(points_list[i][1])
    # ensuring the first and last points are connected as well
    if flag_connect_last_with_first == True:
        x_list.append(points_list[0][0])
        y_list.append(points_list[0][1])    
    # plotting the line connecting the dots
    if flag_single_connections == True:
        for i in range(len(x_list)-1):
            input_ax.plot( [x_list[i], x_list[i+1]], [y_list[i], y_list[i+1]], linewidth=linewidth, linestyle=linestyle, color=linecolor, zorder=input_zorder)
    else:
        input_ax.plot( x_list, y_list, linewidth=linewidth, linestyle=linestyle, color=linecolor, zorder=input_zorder)
    return


# This function is used to annotate a feature within a plot with an arrow.
def annotate_image_with_arrow(
    input_ax,
    input_annotatestring,
    input_arrowvertices,
    input_xdist,
    # arrow properties
    input_arrow_linewidth,
    input_arrow_color,
    input_arrow_tipwidth,
    input_arrow_tiplength,
    input_arrowzorder = 30,
    # text properties
    input_fontsize = 11,
    input_rotation = 0,
    input_verticalalignment = "center",
    input_horizontalalignment = "center",
    input_textcolor = "black"):

    # printing the annotatestring
    astring = plt.text(
        x = input_arrowvertices[0][0],
        y = input_arrowvertices[0][1],
        s = input_annotatestring,
        fontsize = input_fontsize,
        rotation = input_rotation,
        verticalalignment = input_verticalalignment,
        horizontalalignment = input_horizontalalignment,
        color = input_textcolor)
    # plotting the arrow
    plot_arrow_connect_points(
        ax = input_ax,
        points_list = input_arrowvertices,
        linewidth = input_arrow_linewidth,
        color = input_arrow_color,
        tip_width = input_arrow_tipwidth,
        tip_length = input_arrow_tiplength,
        flag_single_connections = True,
        input_zorder = input_arrowzorder)
    return


# This function is used to calculate the sector of an ellipse given by two points.
# Returns the ellipse arc points (from p1 to p2) as plottable x- and y-lists.
# reminder: x**2/a**2 + y**2/b**2 = 1
def calc_ellipse_sector_points_list(p1, p2, sector=["bottom", "top"][0], nx=100, flag_verbose=False):

    # computing parameters of ellipse around origin
    a = np.sqrt((p2[0]-p1[0])**2)
    b = np.sqrt((p2[1]-p1[1])**2)
    upper = p1 if p1[1]>p2[1] else p2
    lower = p1 if p1[1]<p2[1] else p2
    left = p1 if p1[0]<p2[0] else p2
    right = p1 if p1[0]>p2[0] else p2
    if (left==p1 and lower==p1 and sector=="top") or (left==p2 and lower==p2 and sector=="top"):
        wedge = "top left"
    elif (left==p1 and lower==p2 and sector=="top") or (left==p2 and lower==p1 and sector=="top"):
        wedge = "top right"
    elif (left==p1 and lower==p2 and sector=="bottom") or (left==p2 and lower==p1 and sector=="bottom"):
        wedge = "bottom left"
    elif (left==p1 and lower==p1 and sector=="bottom") or (left==p2 and lower==p2 and sector=="bottom"):
        wedge = "bottom right"
    if flag_verbose:
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"upper = {upper}")
        print(f"lower = {lower}")
        print(f"left = {left}")
        print(f"right = {right}")
        print(f"wedge = {wedge}")

    # computing x- and y-list for ellipse around orign
    if wedge in ["top left", "bottom left"]:
        x_list = np.linspace(start=-a, stop=0, num=nx, endpoint=True)
    elif wedge in ["top right", "bottom right"]:
        x_list = np.linspace(start=0, stop=a, num=nx, endpoint=True)
    y_list = [np.sqrt((1-(x**2/a**2))*(b**2)) for x in x_list]
    if "bottom" in wedge : y_list = [-y for y in y_list]

    # computing the center point
    if sector=="top":
        if p1==upper:
            x_center = p1[0]
            y_center = p2[1]
        elif p1==lower:
            x_center = p2[0]
            y_center = p1[1]
    elif sector=="bottom":
        if p1==upper:
            x_center = p2[0]
            y_center = p1[1]
        elif p1==lower:
            x_center = p1[0]
            y_center = p2[1]
    if flag_verbose: print(f"\ncenter point = [{x_center,y_center}]")

    # correcting for center position
    x_list = [x+x_center for x in x_list]
    y_list = [y+y_center for y in y_list]

    return x_list, y_list


# This function is used to draw an arrow from a list of points
def plot_arrow_connect_points(
    ax,
    points_list,
    linewidth,
    color,
    tip_width,
    tip_length,
    linestyle = "-",
    flag_single_connections = True,
    input_zorder = 30,
    flag_correct_tip_size_for_non_equal_axes = False,
    flag_ignore_last_n_points = 0,):

    # modifying the last point in the list so the tip of the arrow tip ends at the last point in 'points_list'
    line_points = points_list[:-1]
    recessed_point = scale_vec(norm_vec(two_tuple_vector=vs(points_list[len(points_list)-1], (-points_list[len(points_list)-2][0], -points_list[len(points_list)-2][1]))), tip_length)
    line_points.append(vs(points_list[len(points_list)-1], (-recessed_point[0], -recessed_point[1])))
    if flag_ignore_last_n_points != 0: line_points = line_points[:-flag_ignore_last_n_points]
    # correcting for non-equal axis scaling
    if flag_correct_tip_size_for_non_equal_axes == True:
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = (ax.get_ylim()[1]-ax.get_ylim()[0]) / (ax.get_xlim()[1]-ax.get_xlim()[0])
        rr = disp_ratio/data_ratio
    else:
        rr = 1
    # generating the points forming the tip of the arrow
    tip_endpoint = points_list[len(points_list)-1]
    tip_center = line_points[len(line_points)-1]
    n = norm_orth_vec(p1=tip_center, p2=tip_endpoint)
    n = [n[0], n[1]/rr]
    tip_left_point = vs(tip_center, scale_vec(n, 0.5*tip_width))
    tip_right_point = vs(tip_center, scale_vec(n, -0.5*tip_width))
    tip_points = [tip_endpoint, tip_left_point, tip_right_point]
    # plotting
    plot_line_connect_points(points_list=line_points, input_ax=ax, linewidth=linewidth, linestyle=linestyle, linecolor=color, flag_connect_last_with_first=False, flag_single_connections=flag_single_connections, input_zorder = input_zorder)
    p = patches.Polygon(tip_points, facecolor=color, closed=True, zorder = input_zorder)
    ax.add_patch(p)
    return


# This function is used to plot a box with text in it
# It is used e.g. within the code of  "PTFE Market Scheme"
def plot_box_with_multiline_text(
    ax,
    box_center = [80,45],
    box_height = 10,
    box_width = 10,
    box_edgecolor = "black",
    box_fillcolor = "white",
    box_linewidth = 1,
    box_zorder = 30,
    text_stringlist = [""],
    text_color = "black",
    text_fontsize = 11,
    text_verticalspacing = 5,
    text_horizontalalignment = "center",
    text_verticalalignment = "center"):

    # plotting the box
    x1 = vs(box_center, (-0.5*box_width, +0.5*box_height))
    x2 = vs(box_center, (+0.5*box_width, +0.5*box_height))
    x3 = vs(box_center, (+0.5*box_width, -0.5*box_height))
    x4 = vs(box_center, (-0.5*box_width, -0.5*box_height))
    p = patches.Polygon([x1,x2,x3,x4], facecolor=box_fillcolor, closed=True, zorder=box_zorder-1)
    ax.add_patch(p)
    plot_line_connect_points(
        points_list = [x1, x2, x3, x4],
        linewidth = box_linewidth,
        linecolor = box_edgecolor,
        flag_connect_last_with_first = True,
        flag_single_connections = True,
        input_zorder = box_zorder,
        input_ax = ax)

    # printing the text
    textposlist = [vs(box_center, (0, -n*text_verticalspacing +(len(text_stringlist)-1)*text_verticalspacing*0.5)) for n in range(len(text_stringlist))]
    #for i in range(len(text_stringlist)):
    for string,pos in zip(text_stringlist, textposlist):
        plt.text(
            x = pos[0],
            y = pos[1],
            s = string,
            horizontalalignment = text_horizontalalignment,
            verticalalignment = text_verticalalignment,
            #transform = ax1.transAxes,
            fontsize = text_fontsize,
            color = text_color,
            zorder = box_zorder+1)

    return


# function to draw a number onto a scheme
def draw_circled_number(ax, r, num="42", radius=2.8, circlecolor='black', edgecolor="black", textcolor='white', linewidth=1.1, textsize=11, izorder=24, num_offset=(0,0)):
    circle = patches.Circle(xy=r, radius=radius, facecolor=circlecolor, zorder=izorder, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
    plt.text(x=r[0]+num_offset[0], y=r[1]+num_offset[1], s=num, color=textcolor, fontsize=textsize, verticalalignment='center', horizontalalignment='center', zorder=izorder+1)
    return


# This function is used to plot a double-tipped arrow connecting two points.
def two_tip_arrow(
    input_ax,
    input_points_list,
    input_linewidth,
    input_color,
    input_tip_width,
    input_tip_length,
    zorder=30,
    flag_text="",
    text_offset=(0,0),
    text_fontsize = 11,
    text_rotation = 0,
    text_horizontalalignment = "center",
    text_verticalalignment = "center",
):
    center_point = vs(input_points_list[0], (0.5*(input_points_list[1][0]-input_points_list[0][0]),0.5*(input_points_list[1][1]-input_points_list[0][1])))
    plot_arrow_connect_points(
        ax=input_ax,
        points_list=[center_point, input_points_list[1]],
        linewidth=input_linewidth,
        color=input_color,
        tip_width=input_tip_width,
        tip_length=input_tip_length,
        flag_single_connections=True,
        input_zorder=30)
    plot_arrow_connect_points(
        ax=input_ax,
        points_list=[center_point, input_points_list[0]],
        linewidth=input_linewidth,
        color=input_color,
        tip_width=input_tip_width,
        tip_length=input_tip_length,
        flag_single_connections=True,
        input_zorder=30)
    if flag_text != "":
        input_ax.text(
            x = center_point[0]+text_offset[0],
            y = center_point[1]+text_offset[1],
            s = flag_text,
            horizontalalignment = text_horizontalalignment,
            verticalalignment = text_verticalalignment,
            #transform = ax1.transAxes,
            fontsize = text_fontsize,
            zorder = zorder,
            rotation = text_rotation,
        )
    return


# function to format the pct string plotted into the pie chart
# This seems to be necessary since LaTeX appears to come with some problems printing the % string.
def stringfunc(pct,data):
    return '{:.1f}\,\%'.format(pct)


# function to plot an arrow with a label onto a plot
# INPUT:
#    a,b (2 tuples): foot and tip of the arrow, respectively
#    ax (plt.ax): axis the arrow will be annotated to
#    ts (string): arrow label
#    fs (float): font size of the arrow label
#    c (string): color of the arrow
#    dfa (float): distance of the arrow label from the arrow patch
#    a_hl, a_hw, a_tw, a_ms (float): head length, head width, tail width and arrow mutation scale respectively
#    add_rot (float): additional rotation
#    align_horizontally (bool): If True the arrow label is printed horizontally instead of parallel to the arrow. Also dfa is used to shift the arrow label towards the left/right.
# OUTPUT:
#    none
def label_arrow(
    a,
    b,
    ax,
    ts = '', # label text string
    fs = 19, # label text font size
    c = 'black', # arrow color
    stringcolor = 'black', # text color
    dfa = 1.5,
    a_hl = 0.8,
    a_hw = 0.8,
    a_tw = 0.2,
    a_ms = 12,
    add_rot = 0,
    align_horizontally = False,
    zorder=20):

    # sanity check
    if a == b:
        print('You tried to plot an arrow pointing from ({},{}) to ({},{}).'.format(a[0],a[1],b[0],b[1]))
        print('Tip and foot of the arrow need to be seperated spatially!')
        print('But you knew that, right?')
        return
    # drawing a custom arrow onto the plot
    custom = patches.ArrowStyle('simple', head_length=a_hl, head_width=a_hw, tail_width=a_tw)
    arrow = patches.FancyArrowPatch(posA=a, posB=b, color=c, shrinkA=1, shrinkB=1, arrowstyle=custom, mutation_scale=a_ms, linewidth=0.01, linestyle="--", zorder=zorder)
    ax.add_patch(arrow)
    ### determining both the position and orientation of the text
    # gemetric quantities
    mid_arrow = (a[0]+0.5*(b[0]-a[0]),a[1]+0.5*(b[1]-a[1])) # center of the arrow
    if a[0]-b[0] == 0:
        if b[1]>a[1]:
            alpha_in_deg = 270
            alpha_in_rad = deg_to_rad(alpha_in_deg)
        elif b[1]<a[1]:
            alpha_in_deg = 90
            alpha_in_rad = deg_to_rad(alpha_in_deg)
        else:
            print('Something strange has happened. (1)')
            return
    elif a[0]-b[0] != 0:
        alpha_in_deg = rad_to_deg(np.arctan((b[1]-a[1])/(b[0]-a[0]))) # angle between x- and y-coordinate of the arrow
        alpha_in_rad = np.arctan((b[1]-a[1])/(b[0]-a[0])) # angle between x- and y-coordinate of the arrow
    else:
        print('Something strange has happened. (2)')
        return
    delta_x = np.sqrt((dfa *np.sin(deg_to_rad(alpha_in_deg)))**2) # shift in x-direction in order to shift the text annotation by dfa
    delta_y = np.sqrt((dfa *np.cos(deg_to_rad(alpha_in_deg)))**2) # shift in y-direction in order to shift the text annotation by dfa
    # determining how to shift the text printed along with the arrow in order to maintain a distance of dfa
    if b[0]-a[0] < 0: # arrow is pointing towards the left
        if b[1]-a[1] < 0:
            arrow_pos = vs(mid_arrow,(-delta_x*signum(dfa),+delta_y*signum(dfa)))
        elif b[1]-a[1] > 0:
            arrow_pos = vs(mid_arrow,(+delta_x*signum(dfa),+delta_y*signum(dfa)))
        elif b[1]-a[1] == 0:
            arrow_pos = vs(mid_arrow,(0,dfa))
        else:
            print('Something strange has happened. (3)')
            return
    elif b[0]-a[0] > 0: # arrow is pointing towards the right
        if b[1]-a[1] < 0:
            arrow_pos = vs(mid_arrow,(+delta_x*signum(dfa),+delta_y*signum(dfa)))
        elif b[1]-a[1] > 0:
            arrow_pos = vs(mid_arrow,(-delta_x*signum(dfa),+delta_y*signum(dfa)))
        elif b[1]-a[1] == 0:
            arrow_pos = vs(mid_arrow,(0,dfa))
        else:
            print('Something strange has happened. (4)')
            return
    elif a[0]-b[0] == 0: # arrow is pointing upwards or downwards
        if b[1] > a[1]:
            arrow_pos = vs(mid_arrow,(+dfa,0))
        elif b[1] < a[1]:
            arrow_pos = vs(mid_arrow,(-dfa,0))
        else:
            print('Something strange has happened. (5)')
            return
    else: # catching any other (probably impossible) case
        print('Something strange has happened. (6)')
        return
    # determining how to rotate the text in order to be aligned with the arrow drawn
    rot = alpha_in_deg +add_rot
    # printing the text
    if align_horizontally == False:
        plt.text(x=arrow_pos[0], y=arrow_pos[1], s=ts, fontsize=fs, rotation=rot, verticalalignment='center', horizontalalignment='center', color=stringcolor)
    elif align_horizontally == True:
        plt.text(x=mid_arrow[0]+dfa, y=mid_arrow[1], s=ts, fontsize=fs, rotation=0, verticalalignment='center', horizontalalignment='center', color=stringcolor)
    else:
        print('The keyword argument "align_horizontally" is supposed to be a boolean, i.e. either True or False.')
        return
    return


# conversion: deg -> rad
def deg_to_rad(deg):
    rad = ((2*np.pi)/360)*deg
    return rad


# converstion: rad -> deg
def rad_to_deg(rad):
    deg = (360/(2*np.pi))*rad
    return deg


# function to label stuff within a plot
def label_das_scheme(r=(80,45), name_string='Science Stuff', function_string='', y_offset=-4, fs_ns=23, fs_fs=19, align='left'):
    plt.text(x=r[0], y=r[1], s=name_string, fontsize=fs_ns, rotation=0, verticalalignment='center', horizontalalignment=align)
    plt.text(x=r[0], y=r[1]+y_offset, s=function_string, fontsize=fs_fs, rotation=0, verticalalignment='center', horizontalalignment=align)
    return


# function to rotate a coordinate given by a two tuple by a specified angle (deg not rad) around another coordinate given by a two tuple
# translation: Gradmaß <-> degree measure, Bogenmaß <-> radian measure
def rotate_two_tuple_around_two_tuple(ptr, angle, cor):
    alpha = deg_to_rad(angle)
    h = (ptr[0]-cor[0],ptr[1]-cor[1])
    h_rot = (h[0]*np.cos(alpha)-h[1]*np.sin(alpha),h[0]*np.sin(alpha)+h[1]*np.cos(alpha))
    h_rot_trans = (h_rot[0]+cor[0],h_rot[1]++cor[1])
    return h_rot_trans


def draw_sign_inscribed_into_circle(
    ax,
    circle_center,
    circle_radius,
    circle_linewidth = 0.5,
    circle_linesstyle = "-",
    circle_zorder = 1,
    circle_fillcolor = "white",
    circle_linecolor = "black",
    sign_sign = ["+", "-"][0],
    sign_width_rel_to_radius = 0.8,
    sign_linewidth = 0.5,
    sign_linestyle = "-",
    sign_zorder = 1,
    sign_linecolor = "black",
):

    """
    This function is used to draw a charge, i.e., a circle with either a plus or a minus sign inscribed into it.
    """

    # adjust for axis display and data ratios (see: https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes)
    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = (ax.get_ylim()[1]-ax.get_ylim()[0]) / (ax.get_xlim()[1]-ax.get_xlim()[0])
    rr = disp_ratio/data_ratio
    print(w, h, disp_ratio, data_ratio, rr)

    # drawing the circle (which is an ellipse with width and heighta adjusted to match the figure aspect and data ratio)
    circle = patches.Ellipse(
        xy = circle_center,
        width = circle_radius,
        height = circle_radius/rr,
        facecolor = circle_fillcolor,
        zorder = circle_zorder,
        edgecolor = circle_linecolor,
        linewidth = circle_linewidth)
    ax.add_patch(circle)
    
    # drawing the inscribed sign
    sign_plot_list = [[
        [circle_center[0]-0.5*sign_width_rel_to_radius*circle_radius, circle_center[0]+0.5*sign_width_rel_to_radius*circle_radius],
        [circle_center[1], circle_center[1]],
        ],]
    if sign_sign == "+":
        sign_plot_list.append([
            [circle_center[0], circle_center[0]],
            [circle_center[1]-0.5*sign_width_rel_to_radius*circle_radius, circle_center[1]+0.5*sign_width_rel_to_radius*circle_radius],
        ])
    for line in sign_plot_list:
        ax.plot(
            line[0],
            line[1],
            linestyle = sign_linestyle,
            zorder = sign_zorder,
            linewidth = sign_linewidth,
            color = sign_linecolor,)
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()
    return


def draw_sign_inscribed_into_square(
    ax,
    square_center,
    square_edgelength,
    square_linewidth = 0.5,
    square_linesstyle = "-",
    square_zorder = 1,
    square_fillcolor = "white",
    square_linecolor = "black",
    sign_sign = ["+", "-"][0],
    sign_width_rel_to_radius = 0.8,
    sign_linewidth = 0.5,
    sign_linestyle = "-",
    sign_zorder = 1,
    sign_linecolor = "black",
):

    """
    This function is used to draw a charge, i.e., a circle with either a plus or a minus sign inscribed into it.
    """

    # adjust for axis display and data ratios (see: https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes)
    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = (ax.get_ylim()[1]-ax.get_ylim()[0]) / (ax.get_xlim()[1]-ax.get_xlim()[0])
    rr = disp_ratio/data_ratio
    print(w, h, disp_ratio, data_ratio, rr)

    # drawing the square (which is a rectangle with width and heighta adjusted to match the figure aspect and data ratio)
#    cap = patches.Rectangle(xy=vs(r,(-0.5*cap_width,0)), width=cap_width, height=-cap_height-0.3, angle=0.0, linewidth=0.001, edgecolor=input_linecolor, facecolor=input_linecolor, zorder=0)
    circle = patches.Rectangle(
        xy = vs(square_center, [-0.5*square_edgelength, -0.5*square_edgelength/rr]),
        width = square_edgelength,
        height = square_edgelength/rr,
        facecolor = square_fillcolor,
        zorder = square_zorder,
        edgecolor = square_linecolor,
        linewidth = square_linewidth)
    ax.add_patch(circle)
    
    # drawing the inscribed sign
    sign_plot_list = [[
        [square_center[0]-0.5*sign_width_rel_to_radius*square_edgelength, square_center[0]+0.5*sign_width_rel_to_radius*square_edgelength],
        [square_center[1], square_center[1]],
        ],]
    if sign_sign == "+":
        sign_plot_list.append([
            [square_center[0], square_center[0]],
            [square_center[1]-0.5*sign_width_rel_to_radius*square_edgelength, square_center[1]+0.5*sign_width_rel_to_radius*square_edgelength],
        ])
    for line in sign_plot_list:
        ax.plot(
            line[0],
            line[1],
            linestyle = sign_linestyle,
            zorder = sign_zorder,
            linewidth = sign_linewidth,
            color = sign_linecolor,)
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()
    return


# function to plot a valve into an image
def plot_valve(ax, r, lw=3, lc='black', width=5, height=3, add_rot=0, facecolor='white', edgecolor='black', zorder=25):
    ### plotting the valve lines
    # coordinates of the corner vertices
    xl = r[0]-0.5*width # x coordinate left
    xr = r[0]+0.5*width # x coordinate right
    yh = r[1]+0.5*height # y coordinate high
    yl = r[1]-0.5*height # y coordinate low
    a = (xl,yh) # upper left vertex
    b = (xl,yl) # lower left vertex
    c = (xr,yh) # upper right vertex
    d = (xr,yl) # lower right vertex
    # rotating the corner vertices
    a = rotate_two_tuple_around_two_tuple(ptr=a, angle=add_rot, cor=r)
    b = rotate_two_tuple_around_two_tuple(ptr=b, angle=add_rot, cor=r)
    c = rotate_two_tuple_around_two_tuple(ptr=c, angle=add_rot, cor=r)
    d = rotate_two_tuple_around_two_tuple(ptr=d, angle=add_rot, cor=r)
    # connecting the corner vertices to draw the lines
    #plot_line(start_tuple=a, end_tuple=b, linewidth=lw, linecolor=lc) # left line
    #plot_line(start_tuple=c, end_tuple=d, linewidth=lw, linecolor=lc) # right line
    #plot_line(start_tuple=a, end_tuple=d, linewidth=lw, linecolor=lc) # diagonal line from top right to bottom left
    #plot_line(start_tuple=b, end_tuple=c, linewidth=lw, linecolor=lc) # diagonal line from bottom right to top left
    valve1 = patches.Polygon(xy=[[r[0],r[1]], [a[0],a[1]], [b[0],b[1]]], closed=True, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, linewidth=lw, linestyle="-")
    ax.add_patch(valve1)
    valve2 = patches.Polygon(xy=[[r[0],r[1]], [c[0],c[1]], [d[0],d[1]]], closed=True, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder, linewidth=lw, linestyle="-")
    ax.add_patch(valve2)
    ### printing the label text
    return


# function to plot a schematic radon trap
# INPUT: see below
# OUTPUT: none
def plot_radon_trap(
        ax, # axis
        r, # positon of the radon trap; center of the middle valve
        lw, # linewidth
        w, # width of the radon trap; from left to right vertical gas line
        depth, # 'depth' of the radon trap
        orientation = 'above', # 'above'--> the vessel is printed ABOVE the y-coordinate of r, 'below'--> the vessel is printed below the y-coordinate of r
        # parameters of the cf vessel containing the activated charcoal
        d_f=40, # d_f determines the length of the cf vessel containing the activated charcoal
        h_f=20, # h_f determines the starting position of the flange
        t_f=3, # height of the cf vessel flanges
        do_f=7, # diameter of the cf vessel flanges
        di_f=5, # diameter of the cylindrical cf vessel
        lw_f=3.3, # linewidth of the cf vessel
        d_r=15,
        l_r=7,
        h_r=4,
        charcoal_poslist = [[1,2]],
        input_linecolor="black",
        flag_rupture_discs = [False,True][0],
        color_trap_line = "black",
        color_trap_fill = "black",
        charcoal_file = "eh_black_black.png"
    ):
    ### defining important positions
    if orientation == 'below':
        # main frame
        a = vs(r, (-0.5*w,0)) # upper left corner
        b = vs(r, (+0.5*w,0)) # upper right corner
        c = vs(a, (0,-depth)) # lower left corner
        d = vs(b, (0,-depth)) # lower right corner
        u = vs(r, (-w,0)) # left edge of the gas line
        v = vs(r, (+w,0)) # right edge of the gas line
        # charcoal vessel
        i = vs(b,(0,-h_f)) # upper end of the radon trap vessel
        h = vs(i,(0,-d_f)) # lower end of the radon trap vessel
        j1 = vs(i,(-0.5*do_f,0)) # upper left vertex of upper flange
        j2 = vs(i,(+0.5*do_f,0)) # upper right vertex of upper flange
        j3 = vs(j1,(0,-t_f)) # lower left vertex of upper flange
        j4 = vs(j2,(0,-t_f)) # lower right vertex of upper flange
        k1 = vs(h,(-0.5*do_f,0)) # lower left vertex of lower flange
        k2 = vs(h,(+0.5*do_f,0)) # lower right vertex of lower flange
        k3 = vs(k1,(0,+t_f)) # upper left vertex of lower flange
        k4 = vs(k2,(0,+t_f)) # upper right vertex of lower flange
        diff = 0.5*(do_f-di_f)
        l1 = vs(j3,(+diff,0)) # upper left vertex of central cylinder
        l2 = vs(j4,(-diff,0)) # upper right vertex of central cylinder
        l3 = vs(k3,(+diff,0)) # lower left vertex of central cylinder
        l4 = vs(k4,(-diff,0)) # lower right vertex of central cylinder
        # rupture discs
        m = vs(a, (0,-d_r)) # position of the left rupture disc
        n = vs(b, (0,-d_r)) # position of the right rupture disc
        o = vs(m, (-l_r,0)) # end of the left rupture disc
        p = vs(n, (+l_r,0)) # end of the right rupture disc
        q = vs(m, (-0.5*l_r,+0.5*h_r)) # upper end of the left rupture disc
        z = vs(m, (-0.5*l_r,-0.5*h_r)) # lower end of the left rupture disc
        s = vs(n, (+0.5*l_r,+0.5*h_r)) # upper end of the right rupture disc
        t = vs(n, (+0.5*l_r,-0.5*h_r)) # lower end of the right rupture disc
    if orientation == 'above':
        # main frame
        a = vs(r, (-0.5*w,0)) # upper left corner
        b = vs(r, (+0.5*w,0)) # upper right corner
        c = vs(a, (0,+depth)) # lower left corner
        d = vs(b, (0,+depth)) # lower right corner
        u = vs(r, (-w,0)) # left edge of the gas line
        v = vs(r, (+w,0)) # right edge of the gas line
        # charcoal vessel
        i = vs(b,(0-w,+h_f)) # upper end of the radon trap vessel
        h = vs(i,(0,+d_f)) # lower end of the radon trap vessel
        j1 = vs(i,(-0.5*do_f,0)) # upper left vertex of upper flange
        j2 = vs(i,(+0.5*do_f,0)) # upper right vertex of upper flange
        j3 = vs(j1,(0,+t_f)) # lower left vertex of upper flange
        j4 = vs(j2,(0,+t_f)) # lower right vertex of upper flange
        k1 = vs(h,(-0.5*do_f,0)) # lower left vertex of lower flange
        k2 = vs(h,(+0.5*do_f,0)) # lower right vertex of lower flange
        k3 = vs(k1,(0,-t_f)) # upper left vertex of lower flange
        k4 = vs(k2,(0,-t_f)) # upper right vertex of lower flange
        diff = 0.5*(do_f-di_f)
        l1 = vs(j3,(+diff,0)) # upper left vertex of central cylinder
        l2 = vs(j4,(-diff,0)) # upper right vertex of central cylinder
        l3 = vs(k3,(+diff,0)) # lower left vertex of central cylinder
        l4 = vs(k4,(-diff,0)) # lower right vertex of central cylinder
        # rupture discs
        m = vs(a, (0,+d_r)) # position of the left rupture disc
        n = vs(b, (0,+d_r)) # position of the right rupture disc
        o = vs(m, (-l_r,0)) # end of the left rupture disc
        p = vs(n, (+l_r,0)) # end of the right rupture disc
        q = vs(m, (-0.5*l_r,-0.5*h_r)) # upper end of the left rupture disc
        z = vs(m, (-0.5*l_r,+0.5*h_r)) # lower end of the left rupture disc
        s = vs(n, (+0.5*l_r,-0.5*h_r)) # upper end of the right rupture disc
        t = vs(n, (+0.5*l_r,+0.5*h_r)) # lower end of the right rupture disc
    ### plotting
    # pipes
    plot_line(start_tuple=u, end_tuple=v, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=a, end_tuple=i, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=h, end_tuple=c, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=b, end_tuple=d, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=c, end_tuple=d, linewidth=lw, linecolor=input_linecolor)
    # radon trap
#    poslist = []
#    for i in range(35):
#        x = (random.randrange(4,97)/100)*(np.sqrt((j2[0]-j1[0])**2))
#        y = (random.randrange(4,97)/100)*(np.sqrt((j2[1]-k1[1])**2))
#        poslist.append(vs(j1,(x,y)))
#    print(poslist)            charcoal_poslist = poslist_2,

    for i in range(len(charcoal_poslist)):
        image_onto_plot(
            filestring = charcoal_file,
            ax = ax,
            position = charcoal_poslist[i],
            pathstring = "./input/",
            zoom = 0.009,
            zorder = 24)
    radon_trap_vessel = patches.Polygon(xy=[j1, j2, k2, k1], closed=True, edgecolor=color_trap_line, facecolor=color_trap_fill, linewidth=lw_f, zorder=23)
    ax.add_patch(radon_trap_vessel)
    # rupture discs
    if flag_rupture_discs == True:
        plot_line(start_tuple=m, end_tuple=o, linewidth=lw, linecolor=input_linecolor)
        plot_line(start_tuple=n, end_tuple=p, linewidth=lw, linecolor=input_linecolor)
        plot_line(start_tuple=q, end_tuple=z, linewidth=lw, linecolor=input_linecolor)
        plot_line(start_tuple=s, end_tuple=t, linewidth=lw, linecolor=input_linecolor)
    return


# function to plot a schematic radon trap
# INPUT: see below
# OUTPUT: none
#def plot_radon_trap(
#        ax, # axis
#        r, # positon of the radon trap; center of the middle valve
#        lw, # linewidth
#        w, # width of the radon trap; from left to right vertical gas line
#        depth, # 'depth' of the radon trap
#        orientation = 'above', # 'above'--> the vessel is printed ABOVE the y-coordinate of r, 'below'--> the vessel is printed below the y-coordinate of r
#        # parameters of the cf vessel containing the activated charcoal
#        d_f=40, # d_f determines the length of the cf vessel containing the activated charcoal
#        h_f=20, # h_f determines the starting position of the flange
#        t_f=3, # height of the cf vessel flanges
#        do_f=7, # diameter of the cf vessel flanges
#        di_f=5, # diameter of the cylindrical cf vessel
#        lw_f=3.3, # linewidth of the cf vessel
#        d_r=15,
#        l_r=7,
#        h_r=4,
#        input_linecolor="black"
#    ):
#    ### defining important positions
#    if orientation == 'below':
#        # main frame
#        a = vs(r, (-0.5*w,0)) # upper left corner
#        b = vs(r, (+0.5*w,0)) # upper right corner
#        c = vs(a, (0,-depth)) # lower left corner
#        d = vs(b, (0,-depth)) # lower right corner
#        u = vs(r, (-w,0)) # left edge of the gas line
#        v = vs(r, (+w,0)) # right edge of the gas line
#        # charcoal vessel
#        i = vs(b,(0,-h_f)) # upper end of the radon trap vessel
#        h = vs(i,(0,-d_f)) # lower end of the radon trap vessel
#        j1 = vs(i,(-0.5*do_f,0)) # upper left vertex of upper flange
#        j2 = vs(i,(+0.5*do_f,0)) # upper right vertex of upper flange
#        j3 = vs(j1,(0,-t_f)) # lower left vertex of upper flange
#        j4 = vs(j2,(0,-t_f)) # lower right vertex of upper flange
#        k1 = vs(h,(-0.5*do_f,0)) # lower left vertex of lower flange
#        k2 = vs(h,(+0.5*do_f,0)) # lower right vertex of lower flange
#        k3 = vs(k1,(0,+t_f)) # upper left vertex of lower flange
#        k4 = vs(k2,(0,+t_f)) # upper right vertex of lower flange
#        diff = 0.5*(do_f-di_f)
#        l1 = vs(j3,(+diff,0)) # upper left vertex of central cylinder
#        l2 = vs(j4,(-diff,0)) # upper right vertex of central cylinder
#        l3 = vs(k3,(+diff,0)) # lower left vertex of central cylinder
#        l4 = vs(k4,(-diff,0)) # lower right vertex of central cylinder
#        # rupture discs
#        m = vs(a, (0,-d_r)) # position of the left rupture disc
#        n = vs(b, (0,-d_r)) # position of the right rupture disc
#        o = vs(m, (-l_r,0)) # end of the left rupture disc
#        p = vs(n, (+l_r,0)) # end of the right rupture disc
#        q = vs(m, (-0.5*l_r,+0.5*h_r)) # upper end of the left rupture disc
#        z = vs(m, (-0.5*l_r,-0.5*h_r)) # lower end of the left rupture disc
#        s = vs(n, (+0.5*l_r,+0.5*h_r)) # upper end of the right rupture disc
#        t = vs(n, (+0.5*l_r,-0.5*h_r)) # lower end of the right rupture disc
#    if orientation == 'above':
#        # main frame
#        a = vs(r, (-0.5*w,0)) # upper left corner
#        b = vs(r, (+0.5*w,0)) # upper right corner
#        c = vs(a, (0,+depth)) # lower left corner
#        d = vs(b, (0,+depth)) # lower right corner
#        u = vs(r, (-w,0)) # left edge of the gas line
#        v = vs(r, (+w,0)) # right edge of the gas line
#        # charcoal vessel
#        i = vs(b,(0-w,+h_f)) # upper end of the radon trap vessel
#        h = vs(i,(0,+d_f)) # lower end of the radon trap vessel
#        j1 = vs(i,(-0.5*do_f,0)) # upper left vertex of upper flange
#        j2 = vs(i,(+0.5*do_f,0)) # upper right vertex of upper flange
#        j3 = vs(j1,(0,+t_f)) # lower left vertex of upper flange
#        j4 = vs(j2,(0,+t_f)) # lower right vertex of upper flange
#        k1 = vs(h,(-0.5*do_f,0)) # lower left vertex of lower flange
#        k2 = vs(h,(+0.5*do_f,0)) # lower right vertex of lower flange
#        k3 = vs(k1,(0,-t_f)) # upper left vertex of lower flange
#        k4 = vs(k2,(0,-t_f)) # upper right vertex of lower flange
#        diff = 0.5*(do_f-di_f)
#        l1 = vs(j3,(+diff,0)) # upper left vertex of central cylinder
#        l2 = vs(j4,(-diff,0)) # upper right vertex of central cylinder
#        l3 = vs(k3,(+diff,0)) # lower left vertex of central cylinder
#        l4 = vs(k4,(-diff,0)) # lower right vertex of central cylinder
#        # rupture discs
#        m = vs(a, (0,+d_r)) # position of the left rupture disc
#        n = vs(b, (0,+d_r)) # position of the right rupture disc
#        o = vs(m, (-l_r,0)) # end of the left rupture disc
#        p = vs(n, (+l_r,0)) # end of the right rupture disc
#        q = vs(m, (-0.5*l_r,-0.5*h_r)) # upper end of the left rupture disc
#        z = vs(m, (-0.5*l_r,+0.5*h_r)) # lower end of the left rupture disc
#        s = vs(n, (+0.5*l_r,-0.5*h_r)) # upper end of the right rupture disc
#        t = vs(n, (+0.5*l_r,+0.5*h_r)) # lower end of the right rupture disc
#    ### plotting
#    # pipes
#    plot_line(start_tuple=u, end_tuple=v, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=a, end_tuple=i, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=h, end_tuple=c, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=b, end_tuple=d, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=c, end_tuple=d, linewidth=lw, linecolor=input_linecolor)
#    # radon trap
#    plot_line(start_tuple=j1, end_tuple=j2, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=j1, end_tuple=j3, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=j3, end_tuple=j4, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=j4, end_tuple=j2, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=k1, end_tuple=k2, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=k1, end_tuple=k3, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=k3, end_tuple=k4, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=k4, end_tuple=k2, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=l1, end_tuple=l2, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=l1, end_tuple=l3, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=l3, end_tuple=l4, linewidth=lw_f, linecolor=input_linecolor)
#    plot_line(start_tuple=l4, end_tuple=l2, linewidth=lw_f, linecolor=input_linecolor)
#    # rupture discs
#    plot_line(start_tuple=m, end_tuple=o, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=n, end_tuple=p, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=q, end_tuple=z, linewidth=lw, linecolor=input_linecolor)
#    plot_line(start_tuple=s, end_tuple=t, linewidth=lw, linecolor=input_linecolor)
#    return


# This function is used to plot a circle.
def plot_circle(center=(0,0), radius=1, phicoverage=(0,2), linewidth=2, linecolor='cyan', numberofpoints=1000, x1x2=[(0,0),(0,0)], izorder=2, flag_returnpointsinsteadofdrawing=True, ax=""):
    x_list = []
    y_list = []
    # given A: center, radius, phicoverage
    if x1x2 == [(0,0),(0,0)]:
        phi_list = np.linspace(phicoverage[0], phicoverage[1], num=numberofpoints, endpoint=True)
        for i in range(len(phi_list)):
            x_list.append(radius*(np.cos(phi_list[i]*np.pi))+center[0])
            y_list.append(radius*(np.sin(phi_list[i]*np.pi))+center[1])
    # given B: center, and two points on the circle (the circle line is drawn in between those two points)
    else:
        r1 = (x1x2[0][0]-center[0], x1x2[0][1]-center[1])
        r2 = (x1x2[1][0]-center[0], x1x2[1][1]-center[1])
        alpha1 = np.sqrt((np.arctan(r1[1]/r1[0])/np.pi)**2)
        alpha2 = np.sqrt((np.arctan(r2[1]/r2[0])/np.pi)**2)
        # case A: 
        # nope
        if r1[0]>0 and r1[1]==0.0:
            phi_list = np.linspace(2-alpha1-0.009, 2-alpha2, num=numberofpoints, endpoint=True)
        # nope
        elif r1[0]<0 and r1[1]==0.0:
            phi_list = np.linspace(1+alpha1+0.009, 1+alpha2, num=numberofpoints, endpoint=True)
        # passt
        elif r1[0]<0 and r1[1]>0:
            phi_list = np.linspace(1+alpha1, 1+alpha2, num=numberofpoints, endpoint=True)
        # 
        elif r1[0]<0 and r1[1]<0:
            phi_list = np.linspace(1+alpha1, 1+alpha2, num=numberofpoints, endpoint=True)
        # passt
        elif r1[0]>0 and r1[1]>0:
            phi_list = np.linspace(0+alpha1, 0+alpha2, num=numberofpoints, endpoint=True)
        # linke feldlinien, unterer teil
        elif r1[0]>0 and r1[1]<0:
            phi_list = np.linspace(0-alpha1, 0-alpha2, num=numberofpoints, endpoint=True)
        else:
            print(r1[0],r1[1])
            print("your case was not caught")
            phi_list = np.linspace(0, 2, num=numberofpoints, endpoint=True)
        for i in range(len(phi_list)):
            x_list.append(radius*(np.cos(phi_list[i]*np.pi))+center[0])
            y_list.append(radius*(np.sin(phi_list[i]*np.pi))+center[1])
    if flag_returnpointsinsteadofdrawing == True:
        points_list = []
        for i in range(len(x_list)):
            points_list.append((x_list[i], y_list[i]))
        return points_list
    else:
        if ax=="":
            plt.plot( x_list, y_list, linewidth=linewidth, color=linecolor, zorder=izorder)
        else:
            ax.plot( x_list, y_list, linewidth=linewidth, color=linecolor, zorder=izorder)
        points_list = []
        for i in range(len(x_list)):
            points_list.append((x_list[i], y_list[i]))
        return points_list


# function to draw an electrical field line into the radon emanation chamber scheme
def draw_field_line(ax, rsp_x, rep_x, x1_length, x2_length, anchor, diode_thickness, sphere_center, sphere_radius, efield_linewidth, efield_color):
    # check input
    # calculate important points
    x_i = vs(anchor,(+rsp_x,-diode_thickness))
    x_f = vs(sphere_center, (+rep_x, -np.sqrt(sphere_radius**2-rep_x**2)))
    x_1 = vs(x_i, (0,-x1_length))
    n = norm_vec(vs(sphere_center,(-x_f[0],-x_f[1])))
    x_2 = vs(x_f, (x2_length*n[0], x2_length*n[1]))
    x_1h = vs(x_1,scale_vec(norm_vec(vs(p1=x_2, p2=x_1, sn=True)),vecdist(p1=x_i,p2=x_1)))
    x_2h = vs(x_2,scale_vec(norm_vec(vs(p1=x_1, p2=x_2, sn=True)),vecdist(p1=x_f,p2=x_2)))
    ### plotting the field line
    # as a smooth line
    if vecdist(p1=x_1,p2=x_2) >=  vecdist(p1=x_i,p2=x_1) +  vecdist(p1=x_2,p2=x_f):
        print("the field line can be drawn")
        c1 = calc_circle_center(x1=x_i,x2=x_1h,rc=x_1)
        c2 = calc_circle_center(x1=x_f,x2=x_2h,rc=x_2)
        #plt.scatter([c1[0],c2[0]],[c1[1],c2[1]])
        plot_circle(center=c1, radius=vecdist(c1,x_1h), phicoverage=(1,2), linewidth=efield_linewidth, linecolor=efield_color, numberofpoints=1000, x1x2=[x_i,x_1h]) #[(0,0),(0,0)])#
        plot_circle(center=c2, radius=vecdist(c2,x_2h), phicoverage=(0,1), linewidth=efield_linewidth, linecolor=efield_color, numberofpoints=1000, x1x2=[x_f,x_2h])
        plot_line(start_tuple=x_1h, end_tuple=x_2h, linewidth=efield_linewidth, linecolor=efield_color)
    # as a polygonal line
    else:
        print("the field line cannot be drawn")
        points_list = [x_i, x_1, x_2, x_f]
        # plotting
        x_list = []
        y_list = []
        for i in points_list:
            x_list.append(i[0])
            y_list.append(i[1])
        plt.plot(x_list, y_list, linewidth=efield_linewidth, color=efield_color)
    # plotting the arrowhead
    arrowhead = patches.Polygon(xy=[[x_i[0],x_i[1]], [x_i[0]+1.2,x_i[1]-1.8], [x_i[0]-1.2,x_i[1]-1.8]], closed=True, facecolor=efield_color, zorder=23)
    ax.add_patch(arrowhead)
    return


def gen_position_list(
    track_start,
    track_end,
    y_dist_max,
    num=50
):
    pos_list = []
    dir_vec = vs(track_end, (-track_start[0], -track_start[1]))
    for i in range(num):
        rand_scale_y_dist = random.randrange(0,100)/100
        rand_scale_xy_dist = random.randrange(-10,95)/100
        xy = vs(track_start, (rand_scale_xy_dist*dir_vec[0], rand_scale_xy_dist*dir_vec[1]))
        xy = vs(xy, (0, rand_scale_y_dist*y_dist_max))
        pos_list.append((xy[0], xy[1], xy[1]-2*rand_scale_y_dist*y_dist_max))
    return pos_list
#def gen_position_list(arrow_start, arrow_end, i_bottom, i_top, num=50):
#    pos_list = []
#    for i in range(num):
#        x = random.randrange(int(round(arrow_end[0]*100,0)), int(round(arrow_start[0]*100,0))+1, 1)/100
#        y_hole = random.randrange(int(round((i_top-2.3)*100,0)), int(round((i_top+0.3)*100,0))+1, 1)/100
#        y_electron = random.randrange(int(round((i_bottom-0.3)*100,0)), int(round((i_bottom+2.3)*100,0))+1, 1)/100
#        pos_list.append((x, y_hole, y_electron))
#    return pos_list


# function to draw an arrow indicating the direction of movement onto the rec scheme
def rec_arrow(ax, start, direction, length=11, ms=8, zorder=24, col='black'):
    custom = patches.ArrowStyle('simple', head_length=0.8, head_width=0.8, tail_width=0.2)
    l = np.sqrt(direction[0]**2 +direction[1]**2)
    scaled_dir = (direction[0]*length/l,direction[1]*length/l)
    arrow = patches.FancyArrowPatch(posA=start, posB=vs(start,scaled_dir), color=col, arrowstyle=custom, mutation_scale=ms, linewidth=0.01, zorder=zorder)
    ax.add_patch(arrow)
    return


# function to scale a vector to a certain length
def scale_vector(vec, length):
    l = np.sqrt(vec[0]**2 +vec[1]**2)
    scaled_vec = (vec[0]*length/l,vec[1]*length/l)
    return scaled_vec    


# function to plot a schematic detection or emanation vessel
# INPUT: see below
# OUTPUT: none
def plot_vessel(
        # general
        ax, # axis the vessel is plotted onto
        r, # positon of the radon trap; center of the middle valve
        orientation, # 'above'--> the vessel is printed ABOVE the y-coordinate of r, 'below'--> the vessel is printed below the y-coordinate of r
        shape, #'rectangle'--> the vessel is printed with a rectangle shape, 'hemisphere'--> the vessel is printed with a hemispherical shape
        # pipes
        pipes_lw, # linewidth
        pipes_w, # distance between the pipes
        pipes_h, # length of the pipes; from main line to vessel
        # vessel
        vessel_height = 50, # height of the rectangular vessel
        vessel_width = 50, # width of the rectangular vessel
        vessel_lw = 3,
        input_linecolor = "black",
        color_vessel_fill = "blue",
        color_vessel_line = "red",
        input_zorder = 5
    ):
    ### defining important positions
    # pipes
    a = vs(r, (-0.5*pipes_w,0)) # r ---> left
    b = vs(r, (+0.5*pipes_w,0)) # r ---> right
    if orientation == 'below':
        c = vs(a, (0,-pipes_h)) # r ---> left ---> down
        d = vs(b, (0,-pipes_h)) # r ---> right ---> down
    if orientation == 'above':
        c = vs(a, (0,+pipes_h)) # r ---> left ---> up
        d = vs(b, (0,+pipes_h)) # r ---> right ---> up
    # vessel
    if orientation == 'below':
        e = vs(r, (-0.5*vessel_width,-pipes_h)) # r ---> down ---> left (upper left corner of the vessel)
        f = vs(r, (+0.5*vessel_width,-pipes_h)) # r ---> down ---> right (upper right corner of the vessel)
        g = vs(f, (0,-vessel_height)) # f ---> down (lower right corner of the vessel)
        h = vs(e, (0,-vessel_height)) # e ---> down (lower left corner of the vessel)
        z = vs(r, (0,-vessel_height-pipes_h))
    if orientation == 'above':
        e = vs(r, (-0.5*vessel_width,+pipes_h)) # r ---> down ---> left (upper left corner of the vessel)
        f = vs(r, (+0.5*vessel_width,+pipes_h)) # r ---> down ---> right (upper right corner of the vessel)
        g = vs(f, (0,+vessel_height)) # f ---> down (lower right corner of the vessel)
        h = vs(e, (0,+vessel_height)) # e ---> down (lower left corner of the vessel)
        z = vs(r, (0,+vessel_height+pipes_h))
        
    ### plotting
    # pipes
    plot_line(start_tuple=a, end_tuple=c, linewidth=pipes_lw, linecolor=input_linecolor, zorder=input_zorder)
    plot_line(start_tuple=b, end_tuple=d, linewidth=pipes_lw, linecolor=input_linecolor, zorder=input_zorder)
    # vessel
    if shape == 'rectangle':
#        plot_line(start_tuple=e, end_tuple=f, linewidth=vessel_lw, linecolor=input_linecolor)
#        plot_line(start_tuple=f, end_tuple=g, linewidth=vessel_lw, linecolor=input_linecolor)
#        plot_line(start_tuple=g, end_tuple=h, linewidth=vessel_lw, linecolor=input_linecolor)
#        plot_line(start_tuple=h, end_tuple=e, linewidth=vessel_lw, linecolor=input_linecolor)
        points_list = [e,f,g,h]
    elif shape == 'hemisphere':
#        plot_line(start_tuple=e, end_tuple=f, linewidth=vessel_lw, linecolor=input_linecolor)
#        plot_line(start_tuple=f, end_tuple=g, linewidth=vessel_lw, linecolor=input_linecolor)
#        plot_line(start_tuple=h, end_tuple=e, linewidth=vessel_lw, linecolor=input_linecolor)
        points_list = [g,f,e,h]
        if orientation == 'above':
            circle_points = plot_circle_2(center=z, radius=0.5*vessel_width, phicoverage=(0,1), linewidth=vessel_lw, linecolor=input_linecolor, numberofpoints=1000, izorder=input_zorder, flag_plot_circle=False, flag_return_points_list = True)
        if orientation == 'below':
            circle_points = plot_circle_2(center=z, radius=0.5*vessel_width, phicoverage=(1,2), linewidth=vessel_lw, linecolor=input_linecolor, numberofpoints=1000, izorder=input_zorder, flag_plot_circle=False, flag_return_points_list = True)
        points_list = points_list +circle_points
    vessel = patches.Polygon(xy=points_list, closed=True, edgecolor=color_vessel_line, facecolor=color_vessel_fill, linewidth=vessel_lw, zorder=input_zorder)
    ax.add_patch(vessel)
    #elif shape == 'hemisphere_without_vessel':
    #    print("no hemispherical vessel printed")
        #plot_line(start_tuple=e, end_tuple=f, linewidth=vessel_lw)
        #plot_line(start_tuple=f, end_tuple=g, linewidth=vessel_lw)
        #plot_line(start_tuple=g, end_tuple=h, linewidth=vessel_lw)
        #plot_line(start_tuple=h, end_tuple=e, linewidth=vessel_lw)
        #if orientation == 'above':
        #    plot_circle(center=z, radius=0.5*vessel_width, phicoverage=(0,1), linewidth=vessel_lw, linecolor='black', numberofpoints=1000, izorder=32)
        #if orientation == 'below':
        #    plot_circle(center=z, radius=0.5*vessel_width, phicoverage=(1,2), linewidth=vessel_lw, linecolor='black', numberofpoints=1000, izorder=32)
    return


# function to plot a bottle onto the MonXe gas system scheme
def plot_gas_bottle(ax, r, diameter, height, cap_width, cap_height, linewidth, input_linecolor="black"):
    # geometry
    center = vs(r,(0,-cap_height-0.5*diameter)) # r corresponds to the top of the cap, c is the center of the circle
    a = vs(center,(+0.5*diameter,0))
    b = vs(a,(0,-height))
    d = vs(center,(-0.5*diameter,0))
    c = vs(d,(0,-height))
    # plotting
    plot_line(start_tuple=a, end_tuple=b, linewidth=linewidth, linecolor=input_linecolor)
    plot_line(start_tuple=b, end_tuple=c, linewidth=linewidth, linecolor=input_linecolor)
    plot_line(start_tuple=c, end_tuple=d, linewidth=linewidth, linecolor=input_linecolor)
    plot_circle_2(center=center, radius=0.5*diameter, phicoverage=(0,1), linewidth=linewidth, linecolor=input_linecolor, numberofpoints=1000, izorder=32)
    cap = patches.Rectangle(xy=vs(r,(-0.5*cap_width,0)), width=cap_width, height=-cap_height-0.3, angle=0.0, linewidth=0.001, edgecolor=input_linecolor, facecolor=input_linecolor, zorder=0)
    ax.add_patch(cap)
    return


# function to draw a pump symbol onto the MonXe gas system scheme
def plot_pump(ax, r, radius, linewidth, orientation="down", izorder=24, triangle_offset=(0,0)):
    circle = patches.Circle(xy=r, radius=radius, facecolor='white', zorder=izorder, edgecolor='black', linewidth=linewidth)
    ax.add_patch(circle)
    if orientation == "down":
        triangle = patches.Polygon(xy=[[r[0],r[1]-radius], [r[0]-radius,r[1]], [r[0]+radius,r[1]]], closed=True, facecolor='black', zorder=izorder+1)
    elif orientation =="right":
        triangle = patches.Polygon(xy=[[r[0],r[1]-radius], [r[0],r[1]+radius], [r[0]+radius,r[1]]], closed=True, facecolor='black', zorder=izorder+1)
    ax.add_patch(triangle)
    return


# generating dtype for the data to be generated
data_dtype = np.dtype([
    ("time", np.float32),
    ("N1", np.float32),
    ("N2", np.float32),
    ("N3", np.float32),
    ("N4", np.float32),
    ("N5", np.float32),
])





#######################################
### Electronic Schematics
#######################################


# This function is used to draw a GND symbol onto a plot
def draw_gnd_symbol(
    ax,
    anchor,
    width,
    depth,
    linewidth,
    color = "black",
    zorder = 1):
    bc = vs(anchor, [0,-depth])
    bl = vs(bc, [-0.5*(2/6)*width, 0])
    br = vs(bc, [0.5*(2/6)*width, 0])
    cc = vs(anchor, [0, -0.5*depth])
    cl = vs(cc, [-0.5*(4/6)*width, 0])
    cr = vs(cc, [0.5*(4/6)*width, 0])
    tl = vs(anchor, [-0.5*(6/6)*width, 0])
    tr = vs(anchor, [0.5*(6/6)*width, 0])
    for line in [[bl, br], [cl, cr], [tl, tr]]:
        ax.plot(
            [line[0][0], line[1][0]],
            [line[0][1], line[1][1]],
            linewidth = linewidth,
            color = color,
            zorder = zorder)


# This function is used to draw a resistor symbol onto a plot
def draw_resistor(
    ax,
    position,
    width,
    height,
    rot = 0,
    linewidth = 1,
    rotation = 0,
    linecolor = "black",
    fillcolor = "white",
    zorder = 1):
    resistor_box = patches.Rectangle(
        xy = vs(position, [-0.5*width, -0.5*height]) if rot==0 else vs(position, [-0.5*height, -0.5*width]),
        width = width if rot==0 else height,
        height = height if rot==0 else width,
        angle = rotation,
        linewidth = linewidth,
        edgecolor = linecolor,
        facecolor = fillcolor,
        zorder = zorder)
    ax.add_patch(resistor_box)
    return


# This function is used to draw a battery symbol onto a matplotlib plot
def draw_battery(
    ax,
    position,
    width,
    height,
    nubsiwidth_rel = 0.18,
    nubsiheigth_rel = 0.6,
    linewidth = 1,
    rotation = 0,
    linecolor = "black",
    fillcolor = "white",
    zorder = 1):
    # drawing the body
    battery_body = patches.Rectangle(
        xy = vs(position, [-0.5*width, -0.5*height]),
        width = width,
        height = height,
        angle = 0,
        linewidth = linewidth,
        edgecolor = linecolor,
        facecolor = fillcolor,
        zorder = zorder)
    ax.add_patch(battery_body)
    # drawing the nubsi
    nubsi_pos = vs(position, [0.5*width, 0])
    battery_nubsi = patches.Rectangle(
        xy = vs(nubsi_pos, [0, -0.5*height*nubsiheigth_rel]),
        width = width*nubsiwidth_rel,
        height = height*nubsiheigth_rel,
        angle = 0,
        linewidth = linewidth,
        edgecolor = linecolor,
        facecolor = fillcolor,
        zorder = zorder+1)
    ax.add_patch(battery_nubsi)
    return


# This function is used to draw a resistor symbol onto a plot
def draw_capacitor(
    ax,
    position,
    width,
    height,
    linewidth = 1,
    rotation = 0,
    linecolor = "black",
    fillcolor = "white",
    zorder = 1):
    # filling the dielectric
    dielectric_box = patches.Rectangle(
        xy = vs(position, [-0.5*width, -0.5*height]),
        width = width,
        height = height,
        angle = rotation,
        linewidth = 0,
        edgecolor = fillcolor,
        facecolor = fillcolor,
        zorder = zorder)
    ax.add_patch(dielectric_box)
    # drawing the capacitor plates
    top_left = vs(position, [-0.5*width, +0.5*height])
    bottom_left = vs(position, [-0.5*width, -0.5*height])
    top_right = vs(position, [+0.5*width, +0.5*height])
    bottom_right = vs(position, [+0.5*width, -0.5*height])
    for plate in [[top_left, bottom_left], [top_right, bottom_right]]:
        ax.plot(
            [plate[0][0], plate[1][0]],
            [plate[0][1], plate[1][1]],
            color = linecolor,
            linewidth = linewidth,
            zorder = zorder)
    return


# This function is used to draw an inductor symbol onto a matplotlib plot.
def draw_inductor(
    ax,
    position,
    width,
    height,
    linewidth,
    rotation = 0,
    linecolor = "black",
    fillcolor = "white",
    zorder = 3):

    # filling the background
    background_box = patches.Rectangle(
        xy = vs(position, [-0.5*width, -0.5*height]),
        width = width,
        height = height,
        angle = rotation,
        linewidth = 0,
        edgecolor = fillcolor,
        facecolor = fillcolor,
        zorder = zorder)
    ax.add_patch(background_box)
    # drawing the inductor semi-circles
    r = width/8
    x1 = vs(position, [-3*r, 0])
    x2 = vs(position, [-1*r, 0])
    x3 = vs(position, [+1*r, 0])
    x4 = vs(position, [+3*r, 0])
    for x in [x1, x2, x3, x4]:
        plot_circle(
            center = x,
            radius = r,
            phicoverage = (0,1),
            linewidth = linewidth,
            linecolor = linecolor,
            numberofpoints = 600,
            x1x2=[(0,0),(0,0)],
            izorder=zorder)

    return



# This function is used to draw a pin connector onto a plot
def draw_pin(
    ax,
    pos,
    width,
    height,
    diagonal_width_rel = 0.8,
    linewidth = 1,
    linecolor = "black",
    fillcolor = "white",
    zorder = 1,
    layout = ["inout", "out", "in", "none"][0],
):
    # inferring vertices
    left_center = vs(pos, [-0.5*width, 0])
    left_top = vs(left_center, [0, 0.5*height])
    left_top_recessed = vs(left_top, [+diagonal_width_rel, 0])
    left_bot = vs(left_center, [0, -0.5*height])
    left_bot_recessed = vs(left_bot, [+diagonal_width_rel, 0])
    right_center = vs(pos, [+0.5*width, 0])
    right_top = vs(right_center, [0, 0.5*height])
    right_top_recessed = vs(right_top, [-diagonal_width_rel, 0])
    right_bot = vs(right_center, [0, -0.5*height])
    right_bot_recessed = vs(right_bot, [-diagonal_width_rel, 0])
    # inferring orientation
    if layout == "inout":
        polygon_array = [left_center, left_top_recessed, right_top_recessed, right_center, right_bot_recessed, left_bot_recessed]
    elif layout == "none":
        polygon_array = [left_top, right_top, right_bot, left_bot]
    elif layout == "in":
        polygon_array = [left_top, right_top_recessed, right_center, right_bot_recessed, left_bot]
    elif layout == "out":
        polygon_array = [left_center, left_top_recessed, right_top, right_bot, left_bot_recessed]
    # drawing polygon
    pin_poly = patches.Polygon(
        xy = polygon_array,
        linewidth = linewidth,
        edgecolor = linecolor,
        facecolor = fillcolor,
        zorder = zorder,
        closed = True)
    ax.add_patch(pin_poly)
    return


# This function is used to draw an amplifier triangle
def draw_amplifier(
    ax,
    pos,
    width,
    height,
    inoutpindist,
    linewidth = 1,
    linecolor = "black",
    fillcolor = "white",
    pin_linewidth = 1,
    pin_in_upper_pos = [],
    pin_in_lower_pos = [],
    pin_out_pos = [],
    pin_vin_upper_pos = [],
    pin_vin_lower_pos = [],
    zorder = 1, 
    label_xoffset_rel = 0.2,
    width_vsupply_rel = 0.3,
    label_yoffset_rel_vsupply = 0.06,
    label_fontsize = 11):

    # vertices
    a = vs(pos, [-0.5*width, 0])
    pin_upper = vs(a, [0, +0.5*inoutpindist])
    corner_upper_left = vs(a, [0, +0.5*height])
    corner_right = vs(pos, [0.5*width, 0])
    corner_lower_left = vs(a, [0, -0.5*height])
    pin_lower = vs(a, [0, -0.5*inoutpindist])
    # drawing the amplifier triangle
    amp_poly = patches.Polygon(
        xy = [a, corner_upper_left, corner_right, corner_lower_left],
        linewidth = linewidth,
        edgecolor = "black",
        facecolor = fillcolor,
        zorder = zorder,
        closed = True)
    ax.add_patch(amp_poly)
    # marking inverting and non-inverting input
    for input_pin_pos in [pin_upper, pin_lower]:
        if input_pin_pos != []:
            ax.text(
                s = r"$+$" if input_pin_pos == pin_lower else r"$-$",
                x = input_pin_pos[0] +label_xoffset_rel*width,
                y = input_pin_pos[1],
                fontsize = label_fontsize,
                color = "black",
                horizontalalignment = "center",
                verticalalignment = "center")            
    # drawing the amplifier connections
    amplifier_connections_list = []
    if pin_in_upper_pos != []:
        amplifier_connections_list.append([pin_upper, pin_in_upper_pos])
    if pin_in_lower_pos != []:
        amplifier_connections_list.append([pin_lower, pin_in_lower_pos])
    if pin_out_pos != []:
        amplifier_connections_list.append([pos, pin_out_pos])
    if pin_vin_upper_pos != []:
        amplifier_connections_list.append([pos, pin_vin_upper_pos])
    if pin_vin_lower_pos != []:
        amplifier_connections_list.append([pos, pin_vin_lower_pos])
    for amp_con in amplifier_connections_list:
        ax.plot(
            [amp_con[0][0],amp_con[1][0]],
            [amp_con[0][1],amp_con[1][1]],
            color = linecolor,
            linewidth = pin_linewidth,
            zorder = zorder-1)
    # labelling the supply voltage
    for input_pin_pos in [pin_vin_upper_pos, pin_vin_lower_pos]:
        if input_pin_pos != []:
            ax.plot(
                [input_pin_pos[0]-0.5*width_vsupply_rel*width, input_pin_pos[0]+0.5*width_vsupply_rel*width],
                [input_pin_pos[1], input_pin_pos[1]],
                color = linecolor,
                linewidth = linewidth,
                zorder = zorder)
            ax.text(
                s = r"$U_{\mathrm{cc}}$" if input_pin_pos == pin_vin_upper_pos else r"$U_{\mathrm{ee}}$",
                x = input_pin_pos[0],
                y = input_pin_pos[1] +label_yoffset_rel_vsupply*height if input_pin_pos == pin_vin_upper_pos else input_pin_pos[1] -label_yoffset_rel_vsupply*height,
                fontsize = label_fontsize,
                color = "black",
                horizontalalignment = "center",
                verticalalignment = "center")            
    return





#######################################
### Decay Chains
#######################################


isotopes_dict = {

    "U_238" : {
        "namestring" : "$\\boldsymbol{^{238}\mathrm{U}}$",
        "n" : 146,
        "z" : 92,
        "t_h" : '',
        "t_h_det" : '',
        "decay" : '',
        "e_alpha" : '',
        "e_alpha_det" : ''
    },

    "Ra_226" : {
        "namestring" : "$\\boldsymbol{^{226}\mathrm{Ra}}$",
        "n" : 138,
        "z" : 88,
        "t_h" : '$1600\,\mathrm{a}$',
        "t_h_det" : '$1600\,\mathrm{a}$',
        "decay" : 'alpha',
        "e_alpha" : '$4.8\,\mathrm{MeV}$',
        "e_alpha_det" : '$4.87062\,\mathrm{MeV}$'
    },

    "Rn_222" : {
        "namestring" : "$\\boldsymbol{^{222}\mathrm{Rn}}$",
        "n" : 136,
        "z" : 86,
        "t_h" : '$3.8\,\mathrm{d}$',
        "t_h_det" : '$3.8232\,\mathrm{d}$',
        "decay" : 'alpha',
        "e_alpha" : '$5.5\,\mathrm{MeV}$',
        "e_alpha_det" : '$5.5903\,\mathrm{MeV}$'
    },

    "Po_218" : {
        "namestring" : "$\\boldsymbol{^{218}\mathrm{Po}}$",
        "n" : 134,
        "z" : 84,
        "t_h" : '$3.1\,\mathrm{min}$',
        "t_h_det" : '$3.071\,\mathrm{min}$',
        "decay" : 'alpha',
        "e_alpha" : '$6.0\,\mathrm{MeV}$',
        "e_alpha_det" : '$6.11468\,\mathrm{MeV}$'
    },

    "Pb_214" : {
        "namestring" : "$\\boldsymbol{^{214}\mathrm{Pb}}$",
        "n" : 132,
        "z" : 82,
        "t_h" : '$26.9\,\mathrm{min}$',
        "t_h_det" : '$26.916\,\mathrm{min}$',
        "decay" : 'beta-'
    },

    "Bi_214" : {
        "namestring" : "$\\boldsymbol{^{214}\mathrm{Bi}}$",
        "n" : 131,
        "z" : 83,
        "t_h" : '$19.8\,\mathrm{min}$',
        "t_h_det" : '$19.8\,\mathrm{min}$',
        "decay" : 'beta-'
    },

    "Po_214" : {
        "namestring" : "$\\boldsymbol{^{214}\mathrm{Po}}$",
        "n" : 130,
        "z" : 84,
        "t_h" : '$162.3\,\mathrm{\mu s}$',
        "t_h_det" : '$162.3\,\mathrm{\mu s}$',
        "decay" : 'alpha',
        "e_alpha" : '$7.7\,\mathrm{MeV}$',
        "e_alpha_det" : '$7.83346\,\mathrm{MeV}$'
    },

    "Pb_210" : {
        "namestring" : "$\\boldsymbol{^{210}\mathrm{Pb}}$",
        "n" : 128,
        "z" : 82,
        "t_h" : '$22.2\,\mathrm{a}$',
        "t_h_det" : '$22.23\,\mathrm{a}$',
        "decay" : 'beta-',
    },

    "Bi_210" : {
        "namestring" : "$\\boldsymbol{^{210}\mathrm{Bi}}$",
        "n" : 127,
        "z" : 83,
        "t_h" : '$5.0\,\mathrm{d}$',
        "t_h_det" : '$5.011\,\mathrm{d}$',
        "decay" : 'beta-'
    },

    "Po_210" : {
        "namestring" : "$\\boldsymbol{^{210}\mathrm{Po}}$",
        "n" : 126,
        "z" : 84,
        "t_h" : '$138.4\,\mathrm{d}$',
        "t_h_det" : '$138.3763\,\mathrm{d}$',
        "decay" : 'alpha',
        "e_alpha" : '$5.3\,\mathrm{MeV}$',
        "e_alpha_det" : '$5.40745\,\mathrm{MeV}$'
    },

    "Pb_206" : {
        "namestring" : "$\\boldsymbol{^{206}\mathrm{Pb}}$",
        "n" : 124,
        "z" : 82,
        "t_h" : 'stable',
        "t_h_det" : 'stable',
        "decay" : 'stable'
    },

    "Ra_224" : {
        "namestring" : "$\\boldsymbol{^{224}\mathrm{Ra}}$",
        "n" : 136,
        "z" : 88,
        "t_h" : '$3.6\,\mathrm{d}$',
        "t_h_det" : '$3.631\,\mathrm{d}$',
        "decay" : 'alpha',
        "e_alpha" : '$5.7\,\mathrm{MeV}$',
        "e_alpha_det" : '$5.68548\,\mathrm{MeV}$'
    },

    "Rn_220" : {
        "namestring" : "$\\boldsymbol{^{220}\mathrm{Rn}}$",
        "n" : 134,
        "z" : 86,
        "t_h" : '$55.8\,\mathrm{s}$',
        "t_h_det" : '$55.8\,\mathrm{s}$',
        "decay" : 'alpha',
        "e_alpha" : '$6.4\,\mathrm{MeV}$',
        "e_alpha_det" : '$6.40467\,\mathrm{MeV}$'
    },

    "Po_216" : {
        "namestring" : "$\\boldsymbol{^{216}\mathrm{Po}}$",
        "n" : 132,
        "z" : 84,
        "t_h" : '$0.148\,\mathrm{s}$',
        "t_h_det" : '$0.148\,\mathrm{s}$',
        "decay" : 'alpha',
        "e_alpha" : '$6.9\,\mathrm{MeV}$',
        "e_alpha_det" : '$6.9063\,\mathrm{MeV}$'
    },

    "Pb_212" : {
        "namestring" : "$\\boldsymbol{^{212}\mathrm{Pb}}$",
        "n" : 130,
        "z" : 82,
        "t_h" : '$10.6\,\mathrm{h}$',
        "t_h_det" : '$10.64\,\mathrm{h}$',
        "decay" : 'beta-'
    },

    "Bi_212" : {
        "namestring" : "$\\boldsymbol{^{212}\mathrm{Bi}}$",
        "n" : 129,
        "z" : 83,
        "t_h" : '$60.5\,\mathrm{min}$',
        "t_h_det" : '$60.54\,\mathrm{min}$',
        "decay" : 'alpha',
        "e_alpha" : '$6.1\,\mathrm{MeV}$',
        "e_alpha_det" : '$6.0914\,\mathrm{MeV}$',
        "br_alpha" : '$35.9\,\\mathrm{\%}$'
    },

    "Po_212" : {
        "namestring" : "$\\boldsymbol{^{212}\mathrm{Po}}$",
        "n" : 128,
        "z" : 84,
        "t_h" : '$300\,\mathrm{ns}$',
        "t_h_det" : '$300\,\mathrm{ns}$',
        "decay" : 'alpha',
        "e_alpha" : '$8.8\,\mathrm{MeV}$',
        "e_alpha_det" : '$8,78517\,\mathrm{MeV}$'
    },

    "Pb_208" : {
        "namestring" : "$\\boldsymbol{^{208}\mathrm{Pb}}$",
        "n" : 126,
        "z" : 82,
        "t_h" : 'stable',
        "t_h_det" : 'stable',
        "decay" : 'stable'
    },

    "Ti_208" : {
        "namestring" : "$\\boldsymbol{^{208}\mathrm{Ti}}$",
        "n" : 127,
        "z" : 81,
        "t_h" : '$3.1\,\mathrm{min}$',
        "t_h_det" : '$3.058\,\mathrm{min}$',
        "decay" : 'beta-'
    }

}


# This function is used to plot an isotope box onto a plt. figure.
# USAGE: plot_decaybox()
def plot_isotope_box(
    ax,
    n,
    z,
    namestring,
    halflifestring,
    namestring_offset=(0,0),
    halflifestringstring_offset=(0,0),
    boxcolor='cyan',
    fs=20,
    col = "black",):

    box = patches.Rectangle(xy=(n,z), width=1, height=1, angle=0.0, linewidth=1, edgecolor='black', facecolor=boxcolor, zorder=30)
    ax.add_patch(box)
    #plt.text(0.05, 0.27, text_string, fontsize=22, transform = axes.transAxes)
    plt.text(x=n+0.13+namestring_offset[0], y=z+0.6-0.05+namestring_offset[1], s=namestring, fontsize=(21/20)*fs, zorder=31, color=col)
    plt.text(x=n+0.15+halflifestringstring_offset[0], y=z+0.19+halflifestringstring_offset[1], s=halflifestring, fontsize=(16/20)*fs, zorder=31, color=col)


# This function is used to plot an arrow representing an alpha decay onto a plt.figure.
# USAGE: plot_decaybox()
def plot_alphaarrow(
    ax,
    n,
    z,
    energy,
    fs = 20,
    arrowcolor = 'black',
    stringcolor = 'black',
    br = 100,
    flag_label = True):
    # correcting for the tiny littly offset between the arrows and the isotope boxes
    n = n +0.03
    z = z +0.03
    # printing the arrows
    if flag_label == True:
        label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts='$\\alpha$', fs=fs, c=arrowcolor, stringcolor=stringcolor, dfa=0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
        label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts=energy, fs=0.7*fs, c=arrowcolor, stringcolor=stringcolor, dfa=-0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
    else:
        label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts='', fs=fs, c=arrowcolor, stringcolor=stringcolor, dfa=0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
        label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts='', fs=0.7*fs, c=arrowcolor, stringcolor=stringcolor, dfa=-0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
    if br != 100:
        label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts=br, fs=0.7*fs, c=arrowcolor, stringcolor=stringcolor, dfa=-0.44, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
    #arrow = patches.Arrow(x=n, y=z, dx=-1, dy=-1, width=0.1, color='black')
    #ax.add_patch(arrow)
    #alphadistance_from_arrow = 0.12
    #energydistance_from_arrow = 0.12
    #plt.text(x=n-0.5-np.sqrt(2)*alphadistance_from_arrow, y=z-0.5+np.sqrt(2)*alphadistance_from_arrow, s='$\\alpha$', fontsize=(16/20)*fs, rotation=45)
    #plt.text(x=n-0.5-np.sqrt(2)*energydistance_from_arrow, y=z-0.5+np.sqrt(2)*energydistance_from_arrow, s=energy, fontsize=(15/20)*fs, rotation=45)
    return


# This function is used to plot an arrow representing an alpha decay onto a plt.figure.
# USAGE: plot_decaybox()
def plot_betaarrow(ax, n, z, fs=20, arrowcolor='black', stringcolor='black', br=100):
    # correcting for the tiny littly offset between the arrows and the isotope boxes
    n = n +0.03
    z = z -0.03
    # printing the arrow
    label_arrow(a=(n+0.5,z+1), b=vs((n+0.5,z+1),(-0.5,0.5)), ax=ax, ts='$\\beta$', fs=fs, c=arrowcolor, stringcolor=stringcolor, dfa=0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=0)


# This function is used to modify the 'n' and 'z' value of a isotope dictionary.
# USAGE: plot_decaybox()
def sd(d, sn, sz):
    d["n"]=d["n"]+sn
    d["z"]=d["z"]+sz
    return d


# This is the main function used within 'Miscellaneous_Figures.ipynb' to plot a box representing a nucleide onto a plt.figure.
def plot_decaybox(
    dictionary,
    dbax,
    dbnamestring_offset=(0,0),
    dbhalflifestringstring_offset=(0,0),
    dbboxcolor='cyan',
    dbfs=11,
    arrowfs=7,
    textcolor="black",
    arrowcolor='black',
    labelcolor='black',
    flag_plotarrow = True):

    # modifying the 'n' and 'z' value via sd() in order to match the definition of 'plot_alphaarrow', 'plot_betaarrow' and 'plot_decaybox'
    dictionary = sd(d=dictionary.copy(), sn=-0.5, sz=-0.5)
    # drawing the arrows
    if dictionary["decay"]=="alpha" and flag_plotarrow==True:
        if "br_alpha" in dictionary.keys():
            plot_alphaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], energy=dictionary["e_alpha"], fs=arrowfs, arrowcolor=arrowcolor, stringcolor=labelcolor, br=dictionary["br_alpha"])
        else:
            plot_alphaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], energy=dictionary["e_alpha"], fs=arrowfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    elif dictionary["decay"]=="beta-" and flag_plotarrow==True:
        plot_betaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], fs=arrowfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    elif dictionary["decay"]=="beta+" and flag_plotarrow==True:
        plot_betaplusarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], fs=arrowfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    elif dictionary["decay"]=="" and flag_plotarrow==True:
        plot_alphaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], energy="", fs=arrowfs, arrowcolor=arrowcolor, stringcolor=labelcolor, flag_label=False)
    # drawing the isotope box
    plot_isotope_box(ax=dbax, n=dictionary["n"], z=dictionary["z"], namestring=dictionary["namestring"], halflifestring=dictionary["t_h"], namestring_offset=dbnamestring_offset, halflifestringstring_offset=dbhalflifestringstring_offset, boxcolor=dbboxcolor, fs=dbfs, col=textcolor)





#######################################
### Radon Emanation Chamber Scheme
#######################################


# function to annotate a png image onto a plot
def image_onto_plot(filestring, ax, position, pathstring=input_pathstring, zoom=1, rotation=0, zorder=2):
    img = mpimg.imread(pathstring +filestring)
    if rotation != 0:
        img = ndimage.rotate(img, rotation)
    imagebox = OffsetImage(img, zoom=zoom)#, zorder=zorder)
    ab = AnnotationBbox(imagebox, position, frameon=False)
    ab.zorder = zorder
    ab.rotation = 90
    ax.add_artist(ab)
    return
# EXAMPLE: image_onto_plot(filestring='monxe_logo.png', ax=ax1, position=(80,55), pathstring=input_pathstring, zoom=0.3, zorder=25)


# function to draw a number onto a scheme
def draw_nucleon(ax, r, circlecolor, bordercolor, lw, radius=2.8, izorder=24):
    circle = patches.Circle(xy=r, radius=radius, facecolor=circlecolor, zorder=izorder, edgecolor=bordercolor, linewidth=lw, linestyle="-")
    ax.add_patch(circle)
    return

# function to convert polar coordinates into cartesian coordinates
def rphitoxy(r,phi):
    x = r *np.cos(phi*np.pi)
    y = r *np.sin(phi*np.pi)
    return (x,y)


# function to calculate the orthogonally projected distance from a point on the equator to the surface of the sphere (this is necessary for placing the HV lines)
def sphere_height(distance_from_center, radius):
    return np.sqrt(radius**2-distance_from_center**2)


# function to calculate the center of a circle to smooth out the polygonal field line approximation
def calc_circle_center(x1,x2,rc):
    n1 = norm_orth_vec(p1=rc,p2=x1)
    n2 = norm_orth_vec(p1=rc,p2=x2)
    chi2 = 1/(n2[0]*n1[1]/n1[0]-n2[1]) * (x2[1] -x1[1] -x2[0]*n1[1]/n1[0] +x1[0]*n1[1]/n1[0]) # calculated analytically
    c = vs(x2, (chi2*n2[0],chi2*n2[1]))
    return c


# This function is used to plot a circle.
def plot_circle_2(center=(0,0), radius=1, phicoverage=(0,2), linewidth=2, linecolor='cyan', numberofpoints=1000, x1x2=[(0,0),(0,0)], izorder=2, flag_return_points_list=False, flag_plot_circle=True):
    x_list = []
    y_list = []
    # given A: center, radius, phicoverage
    if x1x2 == [(0,0),(0,0)]:
        phi_list = np.linspace(phicoverage[0], phicoverage[1], num=numberofpoints, endpoint=True)
        for i in range(len(phi_list)):
            x_list.append(radius*(np.cos(phi_list[i]*np.pi))+center[0])
            y_list.append(radius*(np.sin(phi_list[i]*np.pi))+center[1])
        #plt.plot( x_list, y_list, linewidth=linewidth, color=linecolor)
    # given B: center, and two points on the circle (the circle line is drawn in between those two points)
    else:
        r1 = (x1x2[0][0]-center[0], x1x2[0][1]-center[1])
        r2 = (x1x2[1][0]-center[0], x1x2[1][1]-center[1])
        alpha1 = np.sqrt((np.arctan(r1[1]/r1[0])/np.pi)**2)
        alpha2 = np.sqrt((np.arctan(r2[1]/r2[0])/np.pi)**2)
        #print(f"r1: ({r1[0]}, {r1[1]})")
        #print(f"r2: ({r2[0]}, {r2[1]})")
        #print(f"alpha1: {alpha1}")
        #print(f"alpha2: {alpha2}")
        # case A: 
        if r1[0]>0 and r1[1]==0.0:
            phi_list = np.linspace(2-alpha1-0.009, 2-alpha2, num=numberofpoints, endpoint=True)
        elif r1[0]<0 and r1[1]==0.0:
            phi_list = np.linspace(1+alpha1+0.009, 1+alpha2, num=numberofpoints, endpoint=True)
        elif r1[0]<0 and r1[1]>0:
            phi_list = np.linspace(1-alpha1, 1-alpha2, num=numberofpoints, endpoint=True)
        elif r1[0]<0 and r1[1]<0:
            phi_list = np.linspace(1+alpha1, 1+alpha2, num=numberofpoints, endpoint=True)
        elif r1[0]>0 and r1[1]>0:
            phi_list = np.linspace(0+alpha1, 0+alpha2, num=numberofpoints, endpoint=True)
        elif r1[0]>0 and r1[1]<0:
            phi_list = np.linspace(2-alpha1, 2-alpha2, num=numberofpoints, endpoint=True)
        else:
            print("your case was not caught")
            phi_list = np.linspace(0, 2, num=numberofpoints, endpoint=True)
        for i in range(len(phi_list)):
            x_list.append(radius*(np.cos(phi_list[i]*np.pi))+center[0])
            y_list.append(radius*(np.sin(phi_list[i]*np.pi))+center[1])
    # plotting the circle line
    if flag_plot_circle == True:
            plt.plot( x_list, y_list, linewidth=linewidth, color=linecolor, zorder=izorder)
    # eventually returning the list of calculated points
    if flag_return_points_list == True:
        points_list = []
        for i in range(len(x_list)):
            points_list.append((x_list[i], y_list[i]))
        return points_list
    else:
        return


# function to draw an electrical field line into the radon emanation chamber scheme
def draw_field_line(ax, rsp_x, rep_x, x1_length, x2_length, anchor, diode_thickness, sphere_center, sphere_radius, efield_linewidth, efield_color, flag_polygon=False, flag_plothelpingpoints=False):
    # check input
    # calculate important points
    x_i = vs(anchor,(+rsp_x,-diode_thickness))
    x_f = vs(sphere_center, (+rep_x, -np.sqrt(sphere_radius**2-rep_x**2)))
    x_1 = vs(x_i, (0,-x1_length))
    n = norm_vec(vs(sphere_center,(-x_f[0],-x_f[1])))
    x_2 = vs(x_f, (x2_length*n[0], x2_length*n[1]))
    x_1h = vs(x_1,scale_vec(norm_vec(vs(p1=x_2, p2=x_1, sn=True)),vecdist(p1=x_i,p2=x_1)))
    x_2h = vs(x_2,scale_vec(norm_vec(vs(p1=x_1, p2=x_2, sn=True)),vecdist(p1=x_f,p2=x_2)))
    c1 = calc_circle_center(x1=x_i,x2=x_1h,rc=x_1)
    c2 = calc_circle_center(x1=x_f,x2=x_2h,rc=x_2)
    ### plotting the field line
    # as a smooth line
    if vecdist(p1=x_1,p2=x_2) >=  vecdist(p1=x_i,p2=x_1) +  vecdist(p1=x_2,p2=x_f) and flag_polygon == False:
        print("drawing smooth field line")
        #plt.scatter([c1[0],c2[0]],[c1[1],c2[1]])
        plot_circle(center=c1, radius=vecdist(c1,x_1h), phicoverage=(1,2), linewidth=efield_linewidth, linecolor=efield_color, numberofpoints=1000, x1x2=[x_i,x_1h]) #[(0,0),(0,0)])#
        plot_circle(center=c2, radius=vecdist(c2,x_2h), phicoverage=(0,1), linewidth=efield_linewidth, linecolor=efield_color, numberofpoints=1000, x1x2=[x_f,x_2h])
        plot_line(start_tuple=x_1h, end_tuple=x_2h, linewidth=efield_linewidth, linecolor=efield_color)
    # as a polygonal line
    else:
        print("drawing polygonal field line")
        points_list = [x_i, x_1, x_2, x_f]
        # plotting
        x_list = []
        y_list = []
        for i in points_list:
            x_list.append(i[0])
            y_list.append(i[1])
        plt.plot(x_list, y_list, linewidth=efield_linewidth, color=efield_color)
    if flag_plothelpingpoints == True:
        plt.scatter(x_i[0], x_i[1], marker="+", s=10, c="cyan")
        plt.scatter(x_f[0], x_f[1], marker="+", s=10, c="cyan")
        plt.scatter(x_1[0], x_1[1], marker="+", s=10, c="cyan")
        plt.scatter(x_2[0], x_2[1], marker="+", s=10, c="cyan")
        plt.scatter(x_1h[0], x_1h[1], marker="+", s=10, c="red")
        plt.scatter(x_2h[0], x_2h[1], marker="+", s=10, c="red")
        plt.scatter(c1[0], c1[1], marker="+", s=20, c="blue")
        plt.scatter(c2[0], c2[1], marker="+", s=20, c="blue")
        print(f"c1_x: {c1[0]}")
        print(f"c1_y: {c1[1]}")
        print(f"c2_x: {c2[0]}")
        print(f"c2_y: {c2[1]}")
    # plotting the arrowhead
    arrowhead = patches.Polygon(xy=[[x_i[0],x_i[1]], [x_i[0]+1.2,x_i[1]-1.8], [x_i[0]-1.2,x_i[1]-1.8]], closed=True, facecolor=efield_color, zorder=23)
    ax.add_patch(arrowhead)
    return







