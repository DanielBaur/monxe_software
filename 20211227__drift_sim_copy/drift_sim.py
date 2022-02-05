




###########################################################
### Import Stuff
###########################################################


import sys
pathstring_miscellaneous_figures = "/home/daniel/Desktop/arbeitsstuff/monxe/software/miscfig/"
pathstring_monxeana = "/home/daniel/Desktop/arbeitsstuff/monxe/software/monxeana/"

import os
import numpy as np
from math import floor, log
from time import time
from copy import copy, deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import scipy.constants as constants
from scipy import optimize
import random
from collections import Counter
#plt.style.use("danielsmplstyle.mplstyle")
plt.style.use(pathstring_miscellaneous_figures +"danielsmplstyle.mplstyle")


isotope_dict = {
    "rn222" : {
        "half_life" : 3.8232 *24 *60 *60, # 3.8232 d in seconds
        "decay_constant" : np.log(2)/(3.8232 *24 *60 *60),
        "initial_number_of_isotopes" : 50,
        "latex_label" : r"$^{222}\,\mathrm{Rn}$",
        "next_in_chain" : "po218",
    },
    "po218" : {
        "half_life" : 3.071*60, # 3.071 min
        "decay_constant" : np.log(2)/(3.071*60),
        #"half_life" : 3.071/100, # 3.071 min
        #"decay_constant" : np.log(2)/(3.071/100),
        "initial_number_of_isotopes" : 40,
        "latex_label" : r"$^{218}\,\mathrm{Po}$",
        "next_in_chain" : "pb214",
    },
    "pb214" : {
        "half_life" : 26.916 *60, # 26.916 min
        "decay_constant" : np.log(2)/(26.916 *60),
        "initial_number_of_isotopes" : 30,
        "latex_label" : r"$^{214}\,\mathrm{Pb}$",
        "next_in_chain" : "bi214",
    },
    "bi214" : {
        "half_life" : 19.8 *60, # 19.8 min
        "decay_constant" : np.log(2)/(19.8 *60),
        "initial_number_of_isotopes" : 20,
        "latex_label" : r"$^{214}\,\mathrm{Bi}$",
        "next_in_chain" : "po214",
    },
    "po214" : {
        "half_life" : 162.3 *10**(-6), # 162.3 µs
        "decay_constant" : np.log(2)/(162.3 *10**(-6)),
        "initial_number_of_isotopes" : 10,
        "latex_label" : r"$^{214}\,\mathrm{Po}$",
        "next_in_chain" : "none",
    },
}
# print(isotope_dict["rn222"]["next_in_chain"])


def txt_to_sa(file, modfac=1):
    """
    This function takes a comsol electric field file and converts it into
    a structured array. It erases every line with a % in it. it then outputs
    an array of tupels. it assigns each value a designation 
    ("r", "z", "er", "ez"). 
    
    Args: 
        file (textfile): a file that contains electric field data
        modfac (scalar): a scalar that linearly modifies the electrical field strength, i.e., a value of modfac=2 would double both the r- and z-component of the output field strength
    
    Returns:
        struc_array (np structured array): an array of the form [("r", "z", "er", "ez"),...]
    
    Example: 
        >>>txt_to_sa("hemisphere.txt")
        array([(0.001132  , 0.00122363,  1.67467868e+00, -113.2275344 ),
               (0.        , 0.00177273, -7.84132656e-03, -114.90858985),
               (0.        , 0.        , -2.58425539e-02, -109.72440572), ...,
               (0.07094793, 0.0862313 ,  1.04995220e+01,   23.64970681),
               (0.07274185, 0.08589922,  1.09808819e+01,   17.07039356),
               (0.07458506, 0.08624701,  9.21222933e+00,   10.86727718)],
              dtype=[('r', '<f16'), ('z', '<f16'), ('er', '<f16'), ('ez', '<f16')])
    
    """
    
    with open(file) as f:
        content = f.readlines()
    content = [line.split() for line in content
              if not '%' in line]
    #content = [tuple(np.float128(y) for y in x) for x in content]
    content = [tuple(np.float128(y)*modfac if x.index(y) in [2,3] else np.float128(y) for y in x) for x in content]

    struc_array = np.array(content, np.dtype([('r',np.float128),('z',np.float128),('er',np.float128),('ez',np.float128)]))
    return struc_array





###########################################################
### Particle Classs
###########################################################


class Particle:
    """
    This is the class of the particle. instances of this class are the 
    instances, that represent the particles in the detector chamber
    """
    
    
    def __init__(self, textfile, sensor_pos, modfac=1, **kwargs):
        """
        This function initiates the particle. 
        Special attention to the bins. They define how fine the grid will be.
        If they are chosen too large, the grid will be too fine and there will 
        be "empty cells" inside the detector volume. if the particle travels 
        through one of these, it will assume, it is outside the the detector 
        volume and be flagged as such.
        So the number of bins must not be chosen too large.
        
        Args:
            textfile (.txt file): a file that contains electic field data.
            sensor_pos (array): an array containing the radius and z position
                of the sensor in the vessel, where z=0 is the lowest point of 
                the detector chamber.
            modfac (float): a parameter scaling the electrical drift field, see the 'txt_to_sa()' function
            kwargs: key word arguments. see below.
            
        Kwargs:
            mobility (float): the mobility value of the particles.
            ds (float): the step size.
            mean_dist (float): the mean distance traveled by particles through
                diffusion in one second.
            neutral_halflife (float): dictates the neutralalization halflife of the 
                particles.
            bins (array): an array of two values, that contains the amount
                of x_bins and y_bins.
            start_pos: (array): an array containing the r and z coordinates
                of the starting position of the particle. The particle will
                then be spawned at this position. If set to None, a random
                position is chosen.
            nucleus (str): dictates the starting point of the decay chain.
                default is "po218"
            interpolation (str): a string which dictates, weather
                both interpolations, one interpolation or no interpolation is used.
                Possible values are "both", "jump", "point" or "none".
            diff (boolean): dictates if diffusion is enabled.
            decay (boolean): dictates, if decay is enabled.
            neutral (boolean): dictaes, if neutralization is enabled.
            sim_chain (boolean): dictates if entire decay chain is simulated.
                normally enabled or disabled by the in-built functions.
        
        Example:
            Particle("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888])
            >>> (Particle instance)
        """
        self.efield = txt_to_sa(file=textfile, modfac=modfac) # array of tupel (r, z, er, ez)
        
        """
        for index in range(len(self.efield)):
            element = self.efield[index]
            new_element = (-element[0], element[1], -element[2], element[3])
            new_element = np.array(new_element, np.dtype([('r',np.float128),('z',np.float128),
                                                          ('er',np.float128),('ez',np.float128)]))
            self.efield = np.append(self.efield, new_element)
        """
        
        self.ef = [list(x) for x in self.efield[['r','z']]] # contains only pos of efield [[r, z],...]
        self.width = self.efield["r"][self.efield["r"].argmax()] + 10**(-6)
        self.height = self.efield["z"][self.efield["z"].argmax()] + 10**(-6)
        
        self.time = 0. # total drifttime
        #self.n_col = 0 # total number of colisions
        self.path =[] # list of all positions
        self.sensor_pos = sensor_pos # pos of sensor
        
        self.set_kwargs(**kwargs)
        
        self.diff_velocity = np.sqrt(8 * constants.k * 300 / # diffusion velocity
                                     (np.pi * 222 * constants.physical_constants['atomic mass constant'][0]))
        self.timelist = [] # list of all time steps
        self.flag = 0 # flag tells the state of the particle
        self.field_strength = self.get_ef(self.position)
        
    
    def set_kwargs(self, **kwargs):
        """
        This function sets specific variables according to specific
        key word arguments.
        """
        
        if "mobility" in kwargs:
            self.mobility = kwargs["mobility"]
        else:
            self.mobility = 0.0016
        
        if "ds" in kwargs:
            self.ds = kwargs["ds"]
        else:
            self.ds = 0.001
        
        if "mean_dist" in kwargs:
            self.mean_dist = kwargs["mean_dist"]
        else:
            self.mean_dist = 2.15*10**(-3)
        
        if "neutral_halflife" in kwargs:
            self.neutral_halflife = kwargs["neutral_halflife"]
        else:
            self.neutral_halflife = 1
        
        if "bins" in kwargs and kwargs["bins"] is not None:
            self.x_bins, self.y_bins = kwargs["bins"][0], kwargs["bins"][1]
        else:
            self.calc_bins(None)
        
        # This is found here because it needs to be computed after bins are set
        # but before start_pos is set.
        self.ef_sorted, self.ef_strength_sorted = self.setup()
        
        if "start_pos" in kwargs:
            self.position = np.array(kwargs["start_pos"])
            self.path.append((self.position[0], self.position[1]))
        else:
            self.set_init_position()
            
        if "nucleus" in kwargs:
            self.nucleus = kwargs["nucleus"]
            self.init_nucleus = self.nucleus
        else:
            self.nucleus = "po218"
            self.init_nucleus = self.nucleus
        
        if "interpolation" in kwargs:
            self.interpolation = kwargs["interpolation"]
        else:
            self.interpolation = "both"
        
        if "diff" in kwargs:
            self.diff = kwargs["diff"]
        else:
            self.diff = True
        
        if "decay" in kwargs:
            self.decay = kwargs["decay"]
        else:
            self.decay = True
        
        if "neutral" in kwargs:
            self.neutral = kwargs["neutral"]
        else:
            self.neutral = True
        
        if "sim_chain" in kwargs:
            self.sim_chain = kwargs["sim_chain"]
        else:
            self.sim_chain = True
    
    
    def calc_bins(self, bins):
        """
        This function calculates the amount of bins needed on the x-axis
        and the y-axis. It determines how large the area of each bin
        must be, to contain on average 4 efield points. It then calculates
        how many bins are on each axis.
        If any parameters other than None were given for bins, it simply
        sets the x_bins and y_bins to this value.
        
        Args:
            bins (array): an array of two values, that contains the amount
                of x_bins and y_bins.
        """
        if bins == None:
            a = 4 * self.width * self.height/len(self.efield)  # area needed per bin
            self.x_bins = floor(self.width/np.sqrt(a))  # sqrt to get side length of square
            self.y_bins = floor(self.height/np.sqrt(a))
        else:
            self.x_bins, self.y_bins = bins[0], bins[1]
    
    
    def setup(self):
        """
        This function sorts the electric field and its field strength
        into a 2d array. The points are sorted by their position. The 2d array is 
        interpreted as a grid, which is layed over the detector vessel
        crosssection. Each entry is saved in the corresponding position
        in the array. 
        To calculate the coordinates of the array position
        where the entry is stored, each electric field coordinate is 
        scaled by the width/height of the vessle crosssection and the
        number of bins. Then we use the floor of that value as array
        coordinates.
        The strength of the electric field is saved in the same
        corresponding position in a second array.
        
        Returns:
            ef_sorted (2d array), ef_strength_sorted(2d array): 
                two 2d arrays with the sorted entries of electric
                field and electric field strength.
        """
        ef_sorted = []
        ef_strength_sorted = []
        
        # generate empty 2d array
        for i in range(self.y_bins):
            line = []
            line2 = []
            for j in range(self.x_bins):
                line.append([])
                line2.append([])
            ef_sorted.append(line)
            ef_strength_sorted.append(line2)
        
        # sort enntries
        for element in self.efield:
            x, y = (floor(self.x_bins * element[0] / self.width),
                    floor(self.y_bins * element[1] / self.height))
            if (x < 0):
                # important for negative x values because floor(0.5) = 0 but floor(-0.5) = -1
                x += 1
            ef_sorted[y][x].append([element[0], element[1]])
            ef_strength_sorted[y][x].append([element[2], element[3]])
        
        # check for "holes" in grid, which could be an indicator for to many bins
        problem = False
        for i in range(self.y_bins):
            encountered_empty = False
            for j in range(self.x_bins):
                if (not ef_sorted[i][j]):
                    encountered_empty = True
                if (ef_sorted[i][j] and encountered_empty):
                    problem = True
        if problem:
            print("Potential problem: bins chosen too small or object inside detector vessel")
            print("If first is true, choose smaller bin size")
        
        return ef_sorted, ef_strength_sorted
    
    
    def set_init_position(self):
        """
        This function sets the initial position of the particle. It generates
        a position, projects it into a two dimensional plane, and checks if 
        the particle is inside the detector volume. it repeats the process 
        until a position is generated, which is inside the volume.
        The initial position is saved as self.init_position.
        """
        # create inital position
        x_pos = self.width*random.random() 
        y_pos = self.width*random.random()
        z_pos = self.height*random.random()
        r = np.sqrt(x_pos**2 + y_pos**2)  # calculate r
        self.position = np.array([r, z_pos]) 
        x, y = (floor(self.x_bins * self.position[0] / self.width),
                floor(self.y_bins * self.position[1] / self.height))
        while (x >= self.x_bins or x < 0 or  # while outside volume
               y >= self.y_bins or y < 0 or
               self.ef_sorted[y][x] == []):
            
            # generate new position
            x_pos = self.width*random.random()
            y_pos = self.width*random.random()
            z_pos = self.height*random.random()
            r = np.sqrt(x_pos**2 + y_pos**2)  # calculate r
            self.position = np.array([r, z_pos])
            self.init_pos = self.position
            x, y = (floor(self.x_bins * self.position[0] / self.width),
                    floor(self.y_bins * self.position[1] / self.height))
        self.init_position = self.position
        self.path.append((self.position[0], self.position[1]))
    
    
    def get_ef(self, position):
        """
        This function gets the electric field of the entered position by 
        finding the electric field strengh of three points in
        self.ef_sorted. It then uses these points to interpolate the 
        electric field at the entered position. To find these points,
        this function looks up, which points are inside the cells of
        self.ef_sorted surrounding the entered position. It then chooses
        the three points closest to the entered position for interpolation.
        
        Args:
            position (array): an array containing the r and z coordinate of 
                the position where the electric field should be interpolated.
        
        Returns:
            field_strength (array): an array containing the field strength of
                the electric field at the entered position.
        """
        # calculate array coordinates
        x, y = (floor(self.x_bins * position[0] / self.width),
                floor(self.y_bins * position[1] / self.height))
        mirror = False
        if (x < 0):
            # important for negative x values because floor(0.5) = 0 but floor(-0.5) = -1
            x += 1
            mirror = True
        
        potential_ef = []
        potential_ef_s = []

        # find all potential points from the surrounding cells
        for i in range(-1,2):
            for j in range(-1,2):
                if (y+i < self.y_bins and y+i >=0 and
                        np.absolute(x)+j < self.x_bins and np.absolute(x)+j >=0):
                    potential_ef += (self.ef_sorted[y+i][np.absolute(x)+j])
                    potential_ef_s += (self.ef_strength_sorted[y+i][np.absolute(x)+j])
        
        if self.interpolation == "both" or self.interpolation == "point":
            # sorting points into larger and samller than 3rd point
            idx = np.argpartition(distance.cdist([[np.absolute(position[0]), position[1]]], potential_ef), 2)
            points = []
            # generating list with three closest points
            for i in range(3):
                points.append((potential_ef[idx[0][i]][0] ,potential_ef[idx[0][i]][1],
                               potential_ef_s[idx[0][i]][0] , potential_ef_s[idx[0][i]][1]))
            field_strength = (-1) * self.interpol(points, np.absolute(position[0]), position[1]) #interpolation
        else:
            
            closest_index = distance.cdist([[np.absolute(position[0]), position[1]]],
                                           potential_ef).argmin()
            field_strength = -1 * np.array(potential_ef_s[closest_index])
            #print(closest_index)
        
        if mirror: # if x in left hemisphere, er must be mirrored
            field_strength[0] *= -1
        return field_strength
    
    
    def interpol(self, points, x, y):
        """
        This function uses three points to interpolate the electic
        field at the x and y position. (point-interpolation)
        
        Args:
            points (array): an array of tupels containing r, z, er and ez
                coordinates of three points
            x (float): the r coordinate of the point where the electric
                field should be interpolated
            y (float): the z coordinate of the point where the electric
                field should be interpolated
        
        Retruns:
            fs (array): an array containing the er and ez component
                of the interpolated electric field.
        """
        x1, y1, er1, ez1 = points[0]
        x2, y2, er2, ez2 = points[1]
        x3, y3, er3, ez3 = points[2]
        
        e = ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        if (e == 0):
            # if division by zero happens: choose closest point instead of interpolation
            closest_index = distance.cdist([[x,y]], [[x1,y1], [x2,y2], [x3,y3]]).argmin()
            return np.array([[er1,ez1], [er2,ez2], [er3,ez3]][closest_index])
        
        # er interpolation
        z1, z2, z3 = er1, er2, er3
        
        a = (x-x1)*(y2-y1)*(z3-z1)
        b = (y-y1)*(z2-z1)*(x3-x1)
        c = (y3-y1)*(z2-z1)*(x-x1)
        d = (z3-z1)*(x2-x1)*(y-y1)
        
        er = (-a-b+c+d)/e + z1
        
        # ez interpolation
        z1, z2, z3 = ez1, ez2, ez3
        
        a = (x-x1)*(y2-y1)*(z3-z1)
        b = (y-y1)*(z2-z1)*(x3-x1)
        c = (y3-y1)*(z2-z1)*(x-x1)
        d = (z3-z1)*(x2-x1)*(y-y1)
        
        ez = (-a-b+c+d)/e + z1
        
        fs = np.array([er, ez])
        return fs
    
    
    def set_diffusion(self):
        """
        This function calculates the diffusion velocity vector. It uses 
        a gaussian probability distribution to calculate where the
        particle should be after a given time step. The used standard deviation
        has bin empirically determined with a different simulation. It can
        also be calculated using the diffusion coefficient or the mobility.
        """ 
        #empirically found value for standard deviation
        if self.diff == True:
            sigma = self.mean_dist * np.sqrt(self.dt)
        else: 
            sigma = 0
        r = random.gauss(0,sigma)
        z = random.gauss(0,sigma)
        self.diff_vel_vec = np.array([r, z])       
    
    
    def update_position(self):
        """
        This function updates the position of the paritcle by adding
        the diffusion velocity vector to the new position of the 
        particle. The new position is calculated in calc_dt.
        """
        self.position = self.new_position + self.diff_vel_vec
        #print(self.position)
        #print("")
        self.r = self.position[0]
        self.z = self.position[1]
    
    
    def update_time(self):
        """
        This function updates the total time of the particle.
        """
        self.time += self.dt
    
    
    def calc_dt(self):
        """
        This function calculates the time it takes to jump from one
        point to the next. In this process it also calculates the
        new position of the particle and the electric field at that 
        position. The jump interpolation happens in this function too.
        """
        # calculate the distance between sensor and start point
        d_start = np.sqrt(np.square(0 - self.position[0]) + 
                          np.square(self.sensor_pos[1] - self.position[1]))
        # electric field at start point
        ef_start = self.field_strength 
        # length of ef_start
        Es = np.sqrt(np.square(ef_start[0]) + np.square(ef_start[1]))
        
        # determine new position
        a = (1 / Es) * self.ds
        self.new_position = self.position + a * ef_start
        
        # check if new position inside chamber
        # if so, get ef at new position
        # if not, set ef_end to same value as ef_start
        self.check_sensor(self.new_position)
        self.check_wall(self.new_position)
        if (self.flag == 0):
            ef_end = self.get_ef(self.new_position)
        else:
            ef_end = ef_start
            self.flag = 0
        self.field_strength = ef_end
        
        # length of ef_end
        Ee = np.sqrt(np.square(ef_end[0]) + np.square(ef_end[1]))
        
        # calculate distance between old and new position
        d = np.sqrt(np.square(self.position[0] - self.new_position[0]) + 
                    np.square(self.position[1] - self.new_position[1]))
        # calculate distance between sonsor and new position
        d_end = np.sqrt(np.square(0 - self.new_position[0]) + 
                        np.square(self.sensor_pos[1] - self.new_position[1]))

        # interpolate based on assumption, that electric field strength is 
        # proportional to 1/r**2
        # uses positions as boundary conditions to compute proportionality constant k
        # then intigrates function and divides by width of integration area to 
        # calculate mean.
        k1 = Es * d_start**2
        k2 = Ee * d_end**2
        k = (k1 + k2) / 2
        if self.interpolation == "both" or self.interpolation == "jump":
            E_mean = k * (1/d_end - 1/d_start) / (d_start-d_end)
        else:
            E_mean = Es
        v = E_mean * self.mobility  # mean drift velocity over this jump
        dt_mean = d / v
        self.dt = dt_mean
    
    
    def update_nucleus(self):
        """
        This function updates the state of the nucleus of the particle.
        It checks if the particle was neutralized and then moves the partice
        according to diffusion and its mean life time by a specific distance.
        Then it changes the state of the nucleus.
        """
        if self.sim_chain:
            if self.flag == 4: # particle was neutralized
                #print("\tupdate_nucleus(): self.flag==4")
                # move particle due to free drift between neutralization and decay
                mean_life =  1 / isotope_dict[self.nucleus]["decay_constant"]
                if self.diff == True:
                    sigma = self.mean_dist * np.sqrt(mean_life)
                    r = random.gauss(0,sigma)
                    z = random.gauss(0,sigma)
                    move_dist = np.array([r, z])
                    self.position += move_dist
                    self.flag = 2
            if self.flag == 2: # particle decayed
                
                # check flag 3
                x, y = (floor(self.x_bins * self.position[0] / self.width),
                        floor(self.y_bins * self.position[1] / self.height))
                if (x < 0): # important for negative x values because floor(0.5) = 0 but floor(-0.5) = -1
                    x += 1
                if (np.absolute(x) >= self.x_bins or
                        y >= self.y_bins or y < 0 or 
                        self.ef_sorted[y][np.absolute(x)] == []):
                    self.flag = 3
                
                if self.flag == 2 and self.nucleus != "po214":
                    self.flag = 0
                    self.nucleus = isotope_dict[self.nucleus]["next_in_chain"]
                    self.timelist.append(self.time)
                    self.time = 0
                elif self.flag == 2 and self.nucleus == "po214":
                    self.timelist.append(self.time)
            
                
    
    def check_flag(self, position):
        """
        This function checks the flag of the particle and updates it.
        The flag tells in what state the particle is.
            flag = 0  => particle still in free flight
            flag = 1  => particle reached detector
            flag = 2  => particle did decay
            flag = 3  => particle left vessel volume
            flag = 4  => particle was neutralized
        """
        self.check_sensor(position)
        self.check_wall(position)
        self.check_decay(position)
        self.check_neutral()
        
        
    def check_sensor(self, position):
        """
        This function is part of check_flag. 
        It checks wheather the particle has reached the detector.
        
        Args:
            position (array): an array containing r and z 
            coordinate of the particle
        """
        if (self.flag == 0):  # if flag is still in free flight
            if (position[1] > self.sensor_pos[1] and 
                    np.absolute(position[0]) <= self.sensor_pos[0]):
                self.flag = 1
    
    
    def check_wall(self, position):
        """
        This function is part of check_flag. 
        It checks wheather the particle has touched a wall.
        
        Args:
            position (array): an array containing r and z 
            coordinate of the particle
        """
        if (self.flag == 0):  # if flag is still in free flight
            x, y = (floor(self.x_bins * position[0] / self.width),
                    floor(self.y_bins * position[1] / self.height))
            if (x < 0): # important for negative x values because floor(0.5) = 0 but floor(-0.5) = -1
                x += 1
            if (np.absolute(x) >= self.x_bins or
                    y >= self.y_bins or y < 0 or 
                    self.ef_sorted[y][np.absolute(x)] == []):
                self.flag = 3
    
    
    def check_decay(self, position):
        """
        This function is part of check_flag. 
        It checks wheather the particle has decayed.
        
        Args:
            position (array): an array containing r and z 
            coordinate of the particle
        """
        if (self.flag == 0 and self.decay):  # if flag is still in free flight and decay enabled
            self.decay_constant = isotope_dict[self.nucleus]["decay_constant"] # lambda = ln(2) / halflife
            random_number = random.random()
            p_exp = np.exp(-self.dt * self.decay_constant)
            if (random_number > p_exp):
                self.flag = 2
    
    
    def check_neutral(self):
        """	
        This function is part of check_flag. 
        It checks wheather the particle has been neutralized.
        
        """
        if (self.flag == 0 and self.neutral):  # if flag is still in free flight
            self.neutral_constant =  np.log(2) / self.neutral_halflife # lambda = ln(2) / halflife
            random_number = random.random()
            p_exp = np.exp(-self.dt * self.neutral_constant)
            if (random_number > p_exp):
                #print(f"neutralized!: p_exp={p_exp}, random_number={random_number}") # NOTE: due to numerical issues p_exp evaluates to a value <1, even if neutral_halflife=1, this results in some neutralizations in the case that 'neutral_halflife' was chosen as one, to circumvent this problem set 'neutral' to False
                self.flag = 4
    
    
    def reset(self):
        """
        This function resets all the values of a particle. That way 
        it is not necessary to create a new particle every time.
        """
        self.time = 0
        self.n_col = 0
        self.field_strength = [0,0]
        self.path =[]
        self.timelist = []
        self.set_init_position()
        self.field_strength = self.get_ef(self.position)
        self.calc_dt()
        self.flag = 0
        self.nucleus = self.init_nucleus
    
    
    def iterate(self):
        """
        This function runs one iteration of the particle.
        THIS IS WHERE THE MAGIC HAPPENS!!!
        """
        self.calc_dt()
        self.set_diffusion()
        self.update_position()
        self.path.append((self.position[0], self.position[1]))
        self.check_flag(self.position)
        self.update_nucleus()
        self.update_time()





###########################################################
### Simulation Functions
###########################################################


def simulate_1p(file, sensor_pos, modfac=1, **kwargs):
    """
    This function simulates one particle, by creating and 
    iterating a particle multiple times. the function returns the
    particle. Its properties can then be read out.
    The default settings represent the recommended settings.
    
    Args:
        file (.txt file): a file that contains electic field data.
        modfac (float): a scaling factor linearly scaling the electrical drift field
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        kwargs: key word arguments. See commentary in particle class.
    
    Returns:
        p (particle): an instance of the class particle.
    
    Example:
        simulate_1p("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888])
        >>> (Particle instance)
    """
    t = time()
    p = Particle(textfile=file, sensor_pos=sensor_pos, modfac=modfac, **kwargs)

    while (p.flag == 0):
        p.iterate()
    return p


def simulate_po218_only(file, sensor_pos, n, **kwargs):
    """
    This function simulates n particles, by creating and 
    iterating a particle multiple times. The function returns 
    the states of particles and the mean drift time of the
    particles together with its standard deviation. It also returns
    a structured array, containing the flags, drift times, and initial
    r and z coordinates of all particles.
    The default settings represent the recommended settings.
    
    Args:
        file (.txt file): a file that contains electic field data.
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        n (int): an integer that dictates how many particles are
            simulated.
        kwargs: key word arguments. See commentary in particle class.
    
    Kwargs:
        print_result (boolean): determines if results are also printed.
            True by default
    
    Retruns:
        flags (array): an array containing 5 elements displaying how many 
            particles had what flag.
        tim (np.array): an array containing mean drift time with standard deviation.
        info (array): a structured array containing the flags, drift times, and initial
            r and z coordinates of all particles.
    
    exaple:
        flags, t, info = simulate_po218_only("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888], 10)
        >>> particle states:
            in flight:  0
            collected:  8
            decayed:  0
            lost to wall:  0
            neutralized:  2

            mean drift time:  0.15036 +- 0.035912 sec
    """    
    print_result = True
    if "sim_chain" not in kwargs:
        kwargs ={**kwargs, **{'sim_chain': False}}
    if "print_result" in kwargs:
        print_result = kwargs["print_result"]
    
    p = Particle(file, sensor_pos, **kwargs)
    info = []
    flags = [0, 0, 0, 0, 0]
    average_t_list = []
    
    
    
    for i in range(n):
        p.reset()
        while (p.flag == 0):
            p.iterate()
        flags[p.flag] += 1
        info.append((p.time, p.flag, p.init_position[0], p.init_position[1]))
        if p.flag == 1:
            average_t_list.append(p.time)
            
    
    mean_t = sum(average_t_list)/len(average_t_list)
    stand_dev_t = 0
    for element in average_t_list:
        stand_dev_t += (mean_t - element)**2
    stand_dev_t = np.sqrt(stand_dev_t/n)/np.sqrt(n) # standdard deviation of the mean
    tim = np.array([mean_t, stand_dev_t])
    
    info = np.array(info, np.dtype([('drift_time',np.float128),('flag',np.float128),('init_r',np.float128),('init_z',np.float128)]))
    #tim = np.array(tim, np.dtype([('mean_time',np.float128),('sigma_mean_time',np.float128)]))
    #flags = np.array(flags)
    #flags = np.array(flags)
    
    if print_result:
        print("particle states:")
        print("in flight: ", flags[0])
        print("collected: ", flags[1])
        print("decayed: ", flags[2])
        print("lost to wall: ", flags[3])
        print("neutralized: ", flags[4])
        print("")
        print("mean drift time: ", round(tim[0],6), "+-", round(tim[1],6), "sec")
    return flags, tim, info


def simulate_entire_chain(file, sensor_pos, n, **kwargs):
    """
    This function simulates n particles, by creating and 
    iterating a particle multiple times. The function works like
    simulate_po218_only, but also simulates the other members of
    decay chain up to po214. It returns a particle summary containing 
    four different nucleus states. for each one, the number of particles 
    which ended its simulation with a specific flag is documented.
    
    Args:
        file (.txt file): a file that contains electic field data.
        n (int): an integer that dictates how many particles are
            simulated.
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        kwargs: key word arguments. See commentary in particle class.
    
    Retruns:
        particle_summary (dict): a dictionary containing all four 
            nucleus states, with each nucleus state containing all different 
            possible flags and the number of particles, which ended in this flag.
    
    example:
        simulate_entire_chain("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888], n = 100)
        >>> {'po218': {'in_flight': 0,
              'reached_detector': 78,
              'decayed': 0,
              'hit_wall': 8,
              'neutralized': 0},
             'pb214': {'in_flight': 0,
              'reached_detector': 12,
              'decayed': 0,
              'hit_wall': 2,
              'neutralized': 0},
             'bi214': {'in_flight': 0,
              'reached_detector': 0,
              'decayed': 0,
              'hit_wall': 0,
              'neutralized': 0},
             'po214': {'in_flight': 0,
              'reached_detector': 0,
              'decayed': 0,
              'hit_wall': 0,
              'neutralized': 0}}
    """
    
    particle_summary = {
        "po218" : {
            "in_flight" : 0,
            "reached_detector" : 0,
            "decayed" : 0,
            "hit_wall" : 0,
            "neutralized" : 0,
        },
        "pb214" : {
            "in_flight" : 0,
            "reached_detector" : 0,
            "decayed" : 0,
            "hit_wall" : 0,
            "neutralized" : 0,
        },
        "bi214" : {
            "in_flight" : 0,
            "reached_detector" : 0,
            "decayed" : 0,
            "hit_wall" : 0,
            "neutralized" : 0,
        },
        "po214" : {
            "in_flight" : 0,
            "reached_detector" : 0,
            "decayed" : 0,
            "hit_wall" : 0,
            "neutralized" : 0,
        },
    }
    p = Particle(file, sensor_pos, **kwargs)
    info = []
    flags = [0, 0, 0, 0, 0]
    average_t_list = []
    
    for i in range(n):
        p.reset()
        while (p.flag == 0):
            p.iterate()
        if p.flag == 0:
            particle_summary[p.nucleus]["in_flight"] += 1
        if p.flag == 1:
            particle_summary[p.nucleus]["reached_detector"] += 1
        if p.flag == 2:
            particle_summary[p.nucleus]["decayed"] += 1
        if p.flag == 3:
            particle_summary[p.nucleus]["hit_wall"] += 1
        if p.flag == 4:
            particle_summary[p.nucleus]["neutralized"] += 1
    else:
        return particle_summary


def get_neutral_halflife(file, sensor_pos, n, collection_eff = [0.5], **kwargs):
    """
    This function simulates n particles and calculates the effective
    amount of impurities in the detector chamber in dependence of the
    observed po218 collection efficiency. the function then prints these
    values and returns an info array, containing all the simulated 
    particles.
    
    Args:
        file (.txt file): a file that contains electic field data.
        n (int): an integer that dictates how many particles are
            simulated.
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        collection_eff (array): an array of values between 0 and 1 representing the
            observed po218 collection efficiencies.
        kwargs: key word arguments. See commentary in particle class.
    
    Retruns:
        mid_bounds (array): an array of the computed neutralization half lives.
    
    example:
        get_neutral_halflife("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888],100, collection_eff = [0.5, 0.9])
        >>> [0.17589736396357686, 1.3759962309205118]
    """
    
    
    def exp_decay(t, neutral_halflife):
        y = np.exp(-t * np.log(2) / neutral_halflife)
        return y


    def set_bounds():
        low_bound = 10**(-6)
        up_bound = 10**3

        if "low_bound" in kwargs:
            low_bound = kwargs["low_bound"]
        if "up_bound" in kwargs:
            up_bound = kwargs["up_bound"]
        return low_bound, up_bound
    
    
    if "neutral" not in kwargs:
        kwargs ={**kwargs, **{'neutral': False}}
    if "decay" not in kwargs:
        kwargs ={**kwargs, **{'decay': False}}
    
    start_time = time()
    t_list = simulate_po218_only(file, sensor_pos, n, print_result = False, **kwargs)[2]["drift_time"]
    y = np.histogram(t_list, 10000)
    #print("histogram calculation: ", round(time() - start_time,3), "sec")
    
    freq = y[0]
    t = np.array([])
    for i in range(len(y[1]) - 1):
        t = np.append(t, (y[1][i] + y[1][i+1]) / 2)
    
    mid_totals = []
    mid_bounds = []
    start_time = time()
    for index in range(len(collection_eff)):
        low_bound, up_bound = set_bounds()
        total_low = 0
        total_up = 0
        for i in range(len(freq)):
            total_low += freq[i] / n * exp_decay(t[i], low_bound)
            total_up += freq[i] / n * exp_decay(t[i], up_bound)
        if not (total_low < collection_eff[index] and total_up > collection_eff[index]):
            print("wanted collection efficiency out of bounds!")

        for iterations in range(100):
            total_mid = 0
            mid_bound = (low_bound + up_bound)/2
            #for i in range(len(freq)):
            #    total_mid += freq[i] / n * exp_decay(t[i], mid_bound)
            total_mid = sum(freq / n * exp_decay(t, mid_bound))
            if total_mid > collection_eff[index]:
                up_bound = mid_bound
            if total_mid < collection_eff[index]:
                low_bound = mid_bound
        mid_totals.append(total_mid)
        mid_bounds.append(mid_bound)
    print("numerical solver: ", round(time() - start_time,3), "sec")
    return mid_bounds


def calc_mean_dist(n = 10000, d = 3.4 * 10**(-10), p = 101325 , m = 218, T = 300, plots = False):
    """
    This function calculates the mean travel distance trhough diffusion of particles 
    in a specific gas composition. it also calculates the diffusion coefficient from this.
    The function will return these values with their respective standard deviations but
    it will also print the values. If plots is enabled, it will also print two plots of 
    the spatial distribution of the particles.
    The preset inputs are for simulation of po218 nuclei in helium at one atmosphere
    
    Args:
        n (int): determines how many particles are simulated.
        p (float): the pressure of the system.
        d (float): the mean diameter of the nucleus and carrier gas.
        T (float): Temperature.
        m (float): mass of particle in atomic mass units.
    
    Returns:
        [(D, sD), (mean, s_mean)] (array): an array containing
            tupels of the diffusion coefficient and the mean distance 
            traveled with their respective standard deviations.
    
    example:
        calc_mean_dist(n = 10000)
        >>>[(7.0275898692137897586e-06, 4.6787481222309470615e-08),
            (0.0021644999221088444556, 1.2479910302831864034e-05)]
    """
    
    def colision_time(d = d, p = p, T = T, m = m):
        """
        This function calculates the mean time between collisions of the polonium atom with other atoms.
        It uses the function for the calculation of the mean collision frequency and calculates
        its multiplicative inverse which is the mean time between collisions.
        
        for reference:
        d = 3.4 * 10**(-10) # mean diameter polonium/helium
        d = 326 * 10**(-12) # mean diameter hydrogen/nitrogen
        d = 360 * 10**(-12) # mean diameter radon/helium
        d = 345 * 10**(-12) # mean diameter air/co2
        d = 354 * 10**(-12) # mean diameter air/o2
        
        Args:
            p (float): the pressure of the system.
            d (float): the mean diameter of the nucleus and carrier gas.
            T (float): Temperature.
            m (float): mass of particle in atomic mass units.
        
        Returns:
            t (float): mean time between colisions.
        """
        t = np.sqrt(constants.k * T * m * constants.physical_constants['atomic mass constant'][0] /
                    (16 * np.pi * p**2 * d**4))
        return t

    def diff_vel_vec(T = T, m = m):
        """
        This function calculates the velocity vector for the particle through diffusion.
        It calculates the mean thermal velocity, then chooses a random direction and
        calculates the velocity vector from that.

        Args:
            T (float): Temperature.
            m (float): mass of particle in atomic mass units.
        
        Returns:
            diff_vel_vec (array): an array containing the diffusion velocity vector 
                of the particle in one step.
        """
        diff_velocity = np.sqrt(8 * constants.k * T / 
                                (np.pi * m * constants.physical_constants['atomic mass constant'][0]))
        #print(diff_velocity)
        # random_theta = np.pi * random.random()
        random_theta = sin_rand() # sine randomness needed to distribute points evenly on surface of a sphere.
        random_phi = 2 * np.pi * random.random()
        diff_vel_vec = (diff_velocity * 
                             np.array([np.sin(random_theta)*np.cos(random_phi), np.cos(random_theta)]))
        return diff_vel_vec


    def sin_rand():
        """
        This function generates random numbers. The first half of one period of the sine function 
        is used as probability density function.
        
        Returns:
            x (float): a random number between 0 and pi.
        """
        x = np.pi * random.random()
        y = random.random()
        while True:
            x = np.pi * random.random()
            y = random.random()
            if y <= np.sin(x):
                return x


    def gauss(x, mu, sigma):
        """
        This function contains the formuly of a gaussian function.

        Args:
            mu (float): The mean of the gaussian function.
            sigma (float): The standard deviation of the gaussian function.
        
        Retruns: 
            y (float/array depending on x input): the resulting values from the 
                gaussian function.
        """
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mu)**2 /(sigma**2))
        return y

    dt = colision_time(d = d, T = T, m = m) # mean collision time
    print(dt)
    steps = 10 # tells how many times diffusion model is iterated. noramlly 10 is enough
    interval = dt * steps # calculates the interval, that represents as many iterations as there are steps
    pos_list = []

    # generate points
    t = time() # for computation time only
    for i in range(n):
        position = [0, 0]
        for j in range(steps):
            position += diff_vel_vec(T = T, m = m) * dt
        pos_list.append((position[0], position[1]))

    pos_list = np.array(pos_list, np.dtype([('r',np.float128),('z',np.float128)]))
    x = pos_list["r"]
    y = pos_list["z"]

    #generate histogram
    y2 = np.histogram(x, 100)
    y3 = np.histogram(y, 100)
    x2 = np.linspace(-50,49,100)
    x3 = np.linspace(-50,49,100)

    dx = (x[x.argmax()] - x[x.argmin()]) / 100 # determines how big one x bin is
    dy = (y[y.argmax()] - y[y.argmin()]) / 100 # determines how big one y bin is
    
    if plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y, ".", color = "red")
        ax.set_title('Spatial distribution')
        ax.set_xlabel("r coordinate " r"$m$")
        ax.set_ylabel("z coordinate " r"$m$")
        ax.grid()
    # ax.legend()


    # fit
    params2, params_cov2 = optimize.curve_fit(gauss, x2, y2[0] / n, p0 = [0,15])
    params3, params_cov3 = optimize.curve_fit(gauss, x3, y3[0] / n, p0 = [0,15])
    
    if plots:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(y2[1][:-1],y2[0]/n, color = "blue", label = "data")
        ax.plot(y2[1][:-1], gauss(x2, params2[0], params2[1]), color = "red", label = "fit")
        ax.set_title('Spatial distribution')
        ax.set_xlabel("Distance traveled in " r"$m$")
        ax.set_ylabel("Relative frequency")
        ax.grid()
        ax.legend()
    
    mean_r = params2[1] * dx / np.sqrt(interval)
    s_mean_r = np.sqrt(params_cov2[1][1]) * dx / np.sqrt(interval)
    
    mean_z = params3[1] * dy / np.sqrt(interval)
    s_mean_z = np.sqrt(params_cov3[1][1]) * dy / np.sqrt(interval)
    
    mean = (mean_r + mean_z) / 2
    s_mean = np.sqrt((s_mean_r/2)**2 + (s_mean_z/2)**2)
    
    D = np.square(mean * np.sqrt(3))/2
    sD = mean * np.sqrt(3) * s_mean
    
    # print(dt, steps)
    print("computation time", round(time() - t, 3), "sec")

    # diffusion coefficient with standard deviation
    print("Diffusion coefficient:  (", D, "+-", sD, ") m^2/s")
    
    #median distance traveled with standard deviation
    print("Mean distance:          (", mean, "+-", s_mean, ") m^2/s")
    return [(D, sD), (mean, s_mean)]





###########################################################
### Histogram Calculation
###########################################################


def hist_calc(file, ax, bins = [40, 40], plot_both_sides = False):
    """
    This function creates a 2d histogram from a file containing
    electric field data. It does so by laying a grid over the 
    field data. then it calculates the mean strength of the electric
    field points in each cell and uses this value as the weight 
    in each bin of the histogram. It then generates a histogram of
    all the cells with their weights.
    
    Args:
        file (.txt file): A text file that contains electric field data.
        ax (a matplotlib axis): The axis in which the histogram is plotted.
        bins (array): an array containing the amount of x bins and ybins
            of the histogram.
    
    Returns:
        hist (a hist2d object): A histogram of the electric field.
    """
    efield = txt_to_sa(file)
    
    # width/height must be slightly larger than the biggest coordinate
    width = efield["r"][efield["r"].argmax()] + 10**(-6)
    height = efield["z"][efield["z"].argmax()] + 10**(-6)
    
    field_strength = np.sqrt(efield["er"]**2 + efield["ez"]**2)
    field_strength = field_strength.astype(float)
    ex = efield["r"]
    ey = efield["z"]

    hist = []
    for i in range(bins[1]):
        line = []
        for j in range(bins[0]):
            line.append([])
        hist.append(line)

    for i in range(len(ex)):
        x = floor(bins[0] * ex[i] / width) 
        y = floor(bins[1] * ey[i] / height)
        if (x < 0):
            # important for negative x values because floor(0.5) = 0 but floor(-0.5) = -1
            x += 1
        hist[y][x].append(field_strength[i])

    x = []
    y = []
    w = []
    scaling_factor_x = (bins[0] + 1) / bins[0]
    scaling_factor_y = (bins[1] + 1) / bins[1]
    for i in range(bins[1]):
        for j in range(bins[0]):
            if hist[i][j] == []:
                hist[i][j] = 0
            else:
                hist[i][j] = np.sum(hist[i][j])/len(hist[i][j])
            x.append(scaling_factor_x * (j + 0) * width / bins[0])
            y.append(scaling_factor_y * (i + 0) * height / bins[1])
            w.append(hist[i][j])
    bin_scale = 1
    if plot_both_sides:
        neg_x = []
        neg_y = []
        for element in x:
            neg_x.append(-element - width / bins[0])
        for element in y:
            neg_y.append(element)

        x += neg_x
        y += neg_y
        w += w
        bin_scale = 2
    #ax.plot(x,y,".")
    hist = ax.hist2d(x, y, bins = [bins[0] * bin_scale, bins[1]], weights=w,
                            norm=mpl.colors.LogNorm(), cmap=plt.cm.YlGnBu) # viridis, hsv, YlGnBu, GnBu, PuBu
    return hist





###########################################################
### Plot
###########################################################


def plot_diagram(file, sensor_pos, n = 1, **kwargs):
    """
    This function plots a histogram of the electric field with a 
    given amount of particles in it. It is designed to be used to 
    get a quick sketch of how the electric field looks and how the
    particles will move through it, without needing to input much.
    The function will save the plot as an image called drift_sim_plot.png.
    
    Args:
        file (.txt file): a file that contains electic field data.
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        n (int): an integer that dictates how many particles are
            simulated.
        kwargs: key word arguments. See commentary in particle class.
        
    Kwargs:
        plot_both_sides (boolean): determines if both or only the right
            half of the chamber is plotted. True plots both sides.
    example:
        plot_diagram("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888],10)
        >>> (prints) computing time:  3.08 sec
                     bins chosen:  47 53
            (plots diagram)
    """
    if "sim_chain" not in kwargs:
        kwargs ={**kwargs, **{'sim_chain': False}}
    
    bin_scale = 1
    plot_both_sides = False
    if "plot_both_sides" in kwargs:
        plot_both_sides = kwargs["plot_both_sides"]
    if plot_both_sides:
        bin_scale = 2
    
    bins = None
    if "bins" in kwargs:
        bins = kwargs["bins"]
    if bins == None:
        p = Particle(file, sensor_pos, **kwargs)
        bins = [p.x_bins, p.y_bins]
    xb = bins[0] # number of x_bins
    yb = bins[1] # number of y_bins
    
    
    fig, ax = plt.subplots(figsize=(1.5 * 5.670/2 * bin_scale, 3.189))
    #fig, ax = plt.subplots(figsize=(8 * bin_scale, 9))


    t = time() # for computation time
    # 'hemisphere.txt'
    p = None
    for i in range(n):  # number of particles plotted
        p = simulate_1p(file, sensor_pos, **kwargs)
        struc_path_array = np.array(p.path, np.dtype([('r',np.float128),('z',np.float128)]))
        ax.plot(struc_path_array["r"], struc_path_array["z"], label="Particle", color="red", alpha=0.8,
                linewidth = 0.5)

    hist = hist_calc(file, ax, [xb, yb], plot_both_sides) # histogram generation
    
    # labels
    #fig.colorbar(hist[3], ax=ax).set_label("Electric field strength in " r"$V/m$", fontsize = 20)
    fig.colorbar(hist[3], ax=ax).set_label("Electric field strength in " r"$V/m$")

    #ax.set(xlim=(-p.width, p.width), ylim=(0, p.height*2))
    #ax.set(xlim=(-p.width, p.width), ylim=(0, p.height))
    #ax.set_title('Path of a particle in electric field', fontsize = 20)
    #ax.set_xlabel("r coordinate in " r"$m$", fontsize = 20)
    #ax.set_ylabel("z coordinate in " r"$m$", fontsize = 20)
    ax.set_xlabel("r coordinate in " r"$m$")
    ax.set_ylabel("z coordinate in " r"$m$")


    fig.savefig("../images/chamber_both_sides.png")
    print("computing time: ", round(time() - t, 3), "sec")
    if n == 1:
        print("drifttime: ", p.time)
    print("bins chosen: ", p.x_bins, p.y_bins)
    #return p


def convergence(
    file,
    sensor_pos,
    start = -2,
    stop = -5,
    stepcount = 30,
    output_abspath_list = [],
    **kwargs
):
    """
    This function plots the drift time of the particles for stepsizes in a
    specific range. The range is chosen from a logarithmic distribution from
    start to stop. The resulting plot containes four graphs displaying 
    the drift time with interpolation, without interpolation, with 
    only point interpolation and with only jump interpolation.
    A spawning point for the particles can be set. if None is set, the point 
    is chosen randomly.
    Diffusion, neutralization, and decay is disabled for the calculation of 
    the drift time, because it would simply add unnecessary noise.
    The function will save the plot as an image called drift_time_convergence.png.
    IF specific is set to "all", it will also return a structured array containing
    x and y positions of all four lines.
    
    Args:
        file (.txt file): a file that contains electic field data.
        sensor_pos (array): an array containing the radius and z position
            of the sensor in the vessel, where z=0 is the lowest point of 
            the detector chamber.
        start (float): The order of magnitude of the start of the stepsize range.
        stop (float): The order of magnitude of the end of the stepsize range.
        stepcount (int): The amount of points in the range.
        kwargs: key word arguments. See commentary in particle class.
        
        special kwargs for this function
            specific (str): a string that dictates if only specific curves are drawn
                to save time. if not set, all curves are drawn.
                popossible values are:
                "all", "none", "jump", "point", "both".
    Returns:
        struc_array (tupel): a tuple containing arrays of x and y coordinates for every
            line
    
    examples:
        convergence("../txt_files/hemisphere_1000V.txt", [0.005, 0.0888], stepcount = 3)
        >>> array([(1.00000000e-02, 0.24346915, 0.24220063, 0.19275638, 0.19184073),
                   (3.16227766e-04, 0.19728438, 0.1972638 , 0.19585621, 0.195862  ),
                   (1.00000000e-05, 0.19593504, 0.19590977, 0.19588977, 0.19586565)],
                  dtype=[('x', '<f16'), ('none', '<f16'), ('point', '<f16'), ('jump', '<f16'), ('both', '<f16')])
            
    """
    t = time()
    x = np.logspace(start,stop,stepcount)

    y1 = []
    y2 = []
    y3 = []
    y4 = []
    
    # if not explicitly specified, add start_pos, diff and decay variables to kwargs
    if "start_pos" not in kwargs:
        kwargs ={**kwargs, **{'start_pos': [0.01, 0.01]}}
    if "diff" not in kwargs:
        kwargs ={**kwargs, **{'diff': False}}
    if "decay" not in kwargs:
        kwargs ={**kwargs, **{'decay': False}}
    if "neutral" not in kwargs:
        kwargs ={**kwargs, **{'neutral': False}}
    if "sim_chain" not in kwargs:
        kwargs ={**kwargs, **{'sim_chain': False}}
    
    specific = "all"
    if "specific" in kwargs:
        specific = kwargs["specific"]
    
    # set bins
    bins = None
    if "bins" in kwargs:
        bins = kwargs["bins"]
    if bins == None:
        p = Particle(file, sensor_pos, **kwargs)
        bins = [p.x_bins, p.y_bins]
    #print(bins)
    
    p = simulate_1p(file, sensor_pos, ds = 10**(stop), interpolation = "both", **kwargs)
    cv = p.time
    fig, ax = plt.subplots(figsize=(5.670, 3.189))
    
    if specific == "none" or specific == "all":
        for ds in x:
            p = simulate_1p(file, sensor_pos, ds = ds, interpolation = "none", **kwargs)
            y1.append(p.time)
        y1_div = [y/cv for y in y1]
        ax.plot(x, y1_div, label = "no interpolation")
    
    if specific == "point" or specific == "all":
        for ds in x:
            p = simulate_1p(file, sensor_pos, ds = ds, interpolation = "point", **kwargs)
            y2.append(p.time)
        y2_div = [y/cv for y in y2]
        ax.plot(x, y2_div, label = "point interpolation")
    
    if specific == "jump" or specific == "all":
        for ds in x:
            p = simulate_1p(file, sensor_pos, ds = ds, interpolation = "jump", **kwargs)
            y3.append(p.time)
        y3_div = [y/cv for y in y3]
        ax.plot(x, y3_div, label = "jump interpolation")
    
    if specific == "both" or specific == "all":
        for ds in x:
            p = simulate_1p(file, sensor_pos, ds = ds, interpolation = "both", **kwargs)
            y4.append(p.time)
        y4_div = [y/cv for y in y4]
        ax.plot(x, y4_div, label = "point and jump interpolation")

    if specific == "all":
        coords = []
        for i in range(len(x)):
            coords.append((x[i], y1_div[i], y2_div[i], y3_div[i], y4_div[i]))
        struc_array = np.array(coords, np.dtype([('x',np.float128),('none',np.float128),
                                                 ('point',np.float128),('jump',np.float128),
                                                 ('both',np.float128)]))

    ax.set_xscale("log")
    ax.set_ylim([0.98, 1.05])
    ax.legend()
    ax.grid()
    #ax.set_title('Drift time convergence',fontsize = 20)
    #ax.set_xlabel("step size in " r"$m$",fontsize = 20)
    #ax.set_ylabel("Drift time relative to convergence value",fontsize = 20)
    #ax.set_title('Drift time convergence')
    ax.set_xlabel("step size in " r"$m$")
    ax.set_ylabel("Relative drift time")
    
    if output_abspath_list != []:
        plt.show()
        for abspath in output_abspath_list:
            fig.savefig(abspath)
    print("computation time: ", round(time() - t, 3), "sec")
    if specific == "all":
        return struc_array
    else:
        return None






