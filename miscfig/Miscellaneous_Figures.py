

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


colorstring_darwin_blue = '#004A9B'
standard_figsize = (5.670, 3.189)

image_format_dict = {
    "16_9" : {
        "figsize" : standard_figsize,
        "axes" : (160, 90)
    },
    "talk" : {
        "figsize" : (standard_figsize[0], standard_figsize[1]*(75/90)),
        "axes" : (160, 75)
    },
    "paper" : {
        "figsize" : (standard_figsize[0], standard_figsize[1]*(75/90)),
        "axes" : (160, 90)
    }
}


def invproplinfunc(x, m, t):
    return m*(1/x) +t


def proplinfunc(x, m, t):
    return m*x +t

    
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


# This function is used to plot a line from one 2tuple to another.
def plot_line(start_tuple, end_tuple, linewidth=2, linecolor='black', **kwargs):
    plt.plot( [start_tuple[0],end_tuple[0]], [start_tuple[1],end_tuple[1]], linewidth=linewidth, color=linecolor, **kwargs)
    return


# This function is used to connect a list of points with each other utilizing the 'plot_line' function defined above.
def plot_line_connect_points(points_list, linewidth=2, linecolor="black", flag_connect_last_with_first=True, flag_single_connections=False):
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
            plt.plot( [x_list[i], x_list[i+1]], [y_list[i], y_list[i+1]], linewidth=linewidth, color=linecolor)
    else:
        plt.plot( x_list, y_list, linewidth=linewidth, color=linecolor)
    return


# This function is used to draw an arrow from a list of points
def plot_arrow_connect_points(ax, points_list, linewidth, color, tip_width, tip_length, flag_single_connections=True):
    # modifying the last point in the list so the tip of the arrow tip ends at the last point in 'points_list'
    line_points = points_list[:-1]
    recessed_point = scale_vec(norm_vec(two_tuple_vector=vs(points_list[len(points_list)-1], (-points_list[len(points_list)-2][0], -points_list[len(points_list)-2][1]))), tip_length)
    line_points.append(vs(points_list[len(points_list)-1], (-recessed_point[0], -recessed_point[1])))
    # generating the points forming the tip of the arrow
    tip_endpoint = points_list[len(points_list)-1]
    tip_center = line_points[len(line_points)-1]
    n = norm_orth_vec(p1=tip_center, p2=tip_endpoint)
    tip_left_point = vs(tip_center, scale_vec(n, 0.5*tip_width))
    tip_right_point = vs(tip_center, scale_vec(n, -0.5*tip_width))
    tip_points = [tip_endpoint, tip_left_point, tip_right_point]
    # checking
    print(line_points)
    print(tip_points)
    # plotting
    plot_line_connect_points(points_list=line_points, linewidth=linewidth, linecolor=color, flag_connect_last_with_first=False, flag_single_connections=flag_single_connections)
    p = patches.Polygon(tip_points, facecolor=color, closed=True)
    ax.add_patch(p)
    return


# function to draw a number onto a scheme
def draw_number(ax, r, num="42", radius=2.8, circlecolor='black', textcolor='white', textsize=11, izorder=24, num_offset=(0,0)):
    circle = patches.Circle(xy=r, radius=radius, facecolor=circlecolor, zorder=izorder, edgecolor='black', linewidth=1.1)
    ax.add_patch(circle)
    plt.text(x=r[0]+num_offset[0], y=r[1]+num_offset[1], s=num, color=textcolor, fontsize=textsize, verticalalignment='center', horizontalalignment='center', zorder=izorder+1)
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
def label_arrow(a, b, ax, ts='', fs=19, c='black', stringcolor='black', dfa=1.5, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=12, add_rot=0, align_horizontally = False, zorder=20):
    # sanity check
    if a == b:
        print('You tried to plot an arrow pointing from ({},{}) to ({},{}).'.format(a[0],a[1],b[0],b[1]))
        print('Tip and foot of the arrow need to be seperated spatially!')
        print('But you knew that, right?')
        return
    # drawing a custom arrow onto the plot
    custom = patches.ArrowStyle('simple', head_length=a_hl, head_width=a_hw, tail_width=a_tw)
    arrow = patches.FancyArrowPatch(posA=a, posB=b, color=c, shrinkA=1, shrinkB=1, arrowstyle=custom, mutation_scale=a_ms, linewidth=0.01, zorder=zorder)
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
        input_linecolor="black"
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
    plot_line(start_tuple=j1, end_tuple=j2, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=j1, end_tuple=j3, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=j3, end_tuple=j4, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=j4, end_tuple=j2, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=k1, end_tuple=k2, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=k1, end_tuple=k3, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=k3, end_tuple=k4, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=k4, end_tuple=k2, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=l1, end_tuple=l2, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=l1, end_tuple=l3, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=l3, end_tuple=l4, linewidth=lw_f, linecolor=input_linecolor)
    plot_line(start_tuple=l4, end_tuple=l2, linewidth=lw_f, linecolor=input_linecolor)
    # rupture discs
    plot_line(start_tuple=m, end_tuple=o, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=n, end_tuple=p, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=q, end_tuple=z, linewidth=lw, linecolor=input_linecolor)
    plot_line(start_tuple=s, end_tuple=t, linewidth=lw, linecolor=input_linecolor)
    return


# This function is used to plot a circle.
def plot_circle(center=(0,0), radius=1, phicoverage=(0,2), linewidth=2, linecolor='cyan', numberofpoints=1000, x1x2=[(0,0),(0,0)], izorder=2, flag_returnpointsinsteadofdrawing=False):
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
    if flag_returnpointsinsteadofdrawing == False:
        points_list = []
        for i in range(len(x_list)):
            points_list.append((x_list[i], y_list[i]))
        return points_list
    else:
        plt.plot( x_list, y_list, linewidth=linewidth, color=linecolor, zorder=izorder)
        return


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


def gen_position_list(arrow_start, arrow_end, i_bottom, i_top, num=50):
    pos_list = []
    for i in range(num):
        x = random.randrange(int(round(arrow_end[0]*100,0)), int(round(arrow_start[0]*100,0))+1, 1)/100
        y_hole = random.randrange(int(round((i_top-2.3)*100,0)), int(round((i_top+0.3)*100,0))+1, 1)/100
        y_electron = random.randrange(int(round((i_bottom-0.3)*100,0)), int(round((i_bottom+2.3)*100,0))+1, 1)/100
        pos_list.append((x, y_hole, y_electron))
    return pos_list


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
        input_linecolor = "black"
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
    plot_line(start_tuple=a, end_tuple=c, linewidth=pipes_lw, linecolor=input_linecolor)
    plot_line(start_tuple=b, end_tuple=d, linewidth=pipes_lw, linecolor=input_linecolor)
    # vessel
    if shape == 'rectangle':
        plot_line(start_tuple=e, end_tuple=f, linewidth=vessel_lw, linecolor=input_linecolor)
        plot_line(start_tuple=f, end_tuple=g, linewidth=vessel_lw, linecolor=input_linecolor)
        plot_line(start_tuple=g, end_tuple=h, linewidth=vessel_lw, linecolor=input_linecolor)
        plot_line(start_tuple=h, end_tuple=e, linewidth=vessel_lw, linecolor=input_linecolor)
    elif shape == 'hemisphere':
        plot_line(start_tuple=e, end_tuple=f, linewidth=vessel_lw, linecolor=input_linecolor)
        plot_line(start_tuple=f, end_tuple=g, linewidth=vessel_lw, linecolor=input_linecolor)
        #plot_line(start_tuple=g, end_tuple=h, linewidth=vessel_lw)
        plot_line(start_tuple=h, end_tuple=e, linewidth=vessel_lw, linecolor=input_linecolor)
        if orientation == 'above':
            plot_circle(center=z, radius=0.5*vessel_width, phicoverage=(0,1), linewidth=vessel_lw, linecolor=input_linecolor, numberofpoints=1000, izorder=32)
        if orientation == 'below':
            plot_circle(center=z, radius=0.5*vessel_width, phicoverage=(1,2), linewidth=vessel_lw, linecolor=input_linecolor, numberofpoints=1000, izorder=32)
    elif shape == 'hemisphere_without_vessel':
        print("no hemispherical vessel printed")
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
    plot_circle(center=center, radius=0.5*diameter, phicoverage=(0,1), linewidth=linewidth, linecolor=input_linecolor, numberofpoints=1000, izorder=32)
    cap = patches.Rectangle(xy=vs(r,(-0.5*cap_width,0)), width=cap_width, height=-cap_height-0.3, angle=0.0, linewidth=0.001, edgecolor=input_linecolor, facecolor=input_linecolor, zorder=0)
    ax.add_patch(cap)
    return


# function to draw a pump symbol onto the MonXe gas system scheme
def plot_pump(ax, r, radius, linewidth, izorder=24, triangle_offset=(0,0)):
    circle = patches.Circle(xy=r, radius=radius, facecolor='white', zorder=izorder, edgecolor='black', linewidth=linewidth)
    ax.add_patch(circle)
    triangle = patches.Polygon(xy=[[r[0],r[1]-radius], [r[0]-radius,r[1]], [r[0]+radius,r[1]]], closed=True, facecolor='black', zorder=izorder+1)
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
### Decay Chains
#######################################


isotopes_dict = {

    "Ra_226" : {
        "namestring" : "$\\boldsymbol{^{226}\mathrm{Ra}}$",
        "n" : 138,
        "z" : 88,
        "t_h" : '$1600\,\mathrm{a}$',
        "t_h_det" : '$1600\,\mathrm{a}$',
        "decay" : 'alpha',
        "e_alpha" : '$4.9\,\mathrm{MeV}$',
        "e_alpha_det" : '$4.87062\,\mathrm{MeV}$'
    },

    "Rn_222" : {
        "namestring" : "$\\boldsymbol{^{222}\mathrm{Rn}}$",
        "n" : 136,
        "z" : 86,
        "t_h" : '$3.8\,\mathrm{d}$',
        "t_h_det" : '$3.8232\,\mathrm{d}$',
        "decay" : 'alpha',
        "e_alpha" : '$5.6\,\mathrm{MeV}$',
        "e_alpha_det" : '$5.5903\,\mathrm{MeV}$'
    },

    "Po_218" : {
        "namestring" : "$\\boldsymbol{^{218}\mathrm{Po}}$",
        "n" : 134,
        "z" : 84,
        "t_h" : '$3.1\,\mathrm{min}$',
        "t_h_det" : '$3.071\,\mathrm{min}$',
        "decay" : 'alpha',
        "e_alpha" : '$6.1\,\mathrm{MeV}$',
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
        "e_alpha" : '$7.8\,\mathrm{MeV}$',
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
        "e_alpha" : '$5.4\,\mathrm{MeV}$',
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
def plot_isotope_box(ax, n, z, namestring, halflifestring, namestring_offset=(0,0), halflifestringstring_offset=(0,0), boxcolor='cyan', fs=20):
    box = patches.Rectangle(xy=(n,z), width=1, height=1, angle=0.0, linewidth=1, edgecolor='black', facecolor=boxcolor, zorder=30)
    ax.add_patch(box)
    #plt.text(0.05, 0.27, text_string, fontsize=22, transform = axes.transAxes)
    plt.text(x=n+0.13+namestring_offset[0], y=z+0.6-0.05+namestring_offset[1], s=namestring, fontsize=(21/20)*fs, zorder=31)
    plt.text(x=n+0.15+halflifestringstring_offset[0], y=z+0.19+halflifestringstring_offset[1], s=halflifestring, fontsize=(16/20)*fs, zorder=31)


# This function is used to plot an arrow representing an alpha decay onto a plt.figure.
# USAGE: plot_decaybox()
def plot_alphaarrow(ax, n, z, energy, fs=20, arrowcolor='black', stringcolor='black', br=100):
    # correcting for the tiny littly offset between the arrows and the isotope boxes
    n = n +0.03
    z = z +0.03
    # printing the arrows
    label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts='$\\alpha$', fs=fs, c=arrowcolor, stringcolor=stringcolor, dfa=0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
    label_arrow(a=(n,z), b=vs((n,z),(-1,-1)), ax=ax, ts=energy, fs=0.7*fs, c=arrowcolor, stringcolor=stringcolor, dfa=-0.17, a_hl=0.8, a_hw=0.8, a_tw=0.2, a_ms=3, add_rot=0, align_horizontally=False, zorder=20)
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
def plot_decaybox(dictionary, dbax, dbnamestring_offset=(0,0), dbhalflifestringstring_offset=(0,0), dbboxcolor='cyan', dbfs=11, arrowcolor='black', labelcolor='black'):
    # modifying the 'n' and 'z' value via sd() in order to match the definition of 'plot_alphaarrow', 'plot_betaarrow' and 'plot_decaybox'
    dictionary = sd(d=dictionary.copy(), sn=-0.5, sz=-0.5)
    # drawing the arrows
    if dictionary["decay"]=="alpha":
        if "br_alpha" in dictionary.keys():
            plot_alphaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], energy=dictionary["e_alpha"], fs=dbfs, arrowcolor=arrowcolor, stringcolor=labelcolor, br=dictionary["br_alpha"])
        else:
            plot_alphaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], energy=dictionary["e_alpha"], fs=dbfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    if dictionary["decay"]=="beta-":
        plot_betaarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], fs=dbfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    if dictionary["decay"]=="beta+":
        plot_betaplusarrow(ax=dbax, n=dictionary["n"], z=dictionary["z"], fs=dbfs, arrowcolor=arrowcolor, stringcolor=labelcolor)
    # drawing the isotope box
    plot_isotope_box(ax=dbax, n=dictionary["n"], z=dictionary["z"], namestring=dictionary["namestring"], halflifestring=dictionary["t_h"], namestring_offset=dbnamestring_offset, halflifestringstring_offset=dbhalflifestringstring_offset, boxcolor=dbboxcolor, fs=dbfs)





#######################################
### Radon Emanation Chamber Scheme
#######################################


# function to annotate a png image onto a plot
def image_onto_plot(filestring, ax, position, pathstring=input_pathstring, zoom=1, zorder=2):
    img = mpimg.imread(pathstring +filestring)
    imagebox = OffsetImage(img, zoom=zoom)#, zorder=zorder)
    ab = AnnotationBbox(imagebox, position, frameon=False)
    ab.set_zorder(zorder)
    ax.add_artist(ab)
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
def plot_circle(center=(0,0), radius=1, phicoverage=(0,2), linewidth=2, linecolor='cyan', numberofpoints=1000, x1x2=[(0,0),(0,0)], izorder=2, flag_return_points_list=False, flag_plot_circle=True):
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







