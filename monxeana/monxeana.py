

# This Python3 library contains code focussing on the analysis of MonXe stuff.





# Contents

# Imports
# Generic Definitions
# Generic Helper Functions
# Raw Data
# Peak Data
# Activity Data
# Post Analysis





#######################################
### Imports
#######################################


import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import pprint
import math
import os
import re
import random
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import getpass
import json
from scipy import stats
from scipy.stats import chi2 # for "Fabian's calculation" of the Poissonian Error
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.stats import chi2
import subprocess
import copy

# including the 'monxe_software' libraries 'monxeana' and 'miscfig'
import sys
pathstring_miscfig = "/home/daniel/Desktop/arbeitsstuff/monxe/software/miscfig/"
sys.path.append(pathstring_miscfig)
import Miscellaneous_Figures as miscfig




#######################################
### Generic Definitions
#######################################


# username
username = getpass.getuser()


# paths
if username == "daniel":
    abspath_monxe = "/home/daniel/Desktop/arbeitsstuff/monxe/"
elif username == "monxe":
    abspath_monxe = "/home/monxe/Desktop/"
else:
    abspath_monxe = "./"
abspath_measurements = abspath_monxe +"measurements/"
relpath_data_compass = "./DAQ/run/RAW/" # this is the folder where CoMPASS stores the measurement data
abspath_monxe_measurements_external = "/media/daniel/2_intenso_256gb_stick/monxe_measurements/"
abspath_mastertalk_images = "/home/daniel/Desktop/arbeitsstuff/mastertalk/images/"
abspath_thesis = "/home/daniel/Desktop/arbeitsstuff/thesis/"
abspath_thesis_monxe_images =  abspath_thesis +"rec/"


# filenames
filename_data_csv = "DataR_CH0@DT5781A_840_run.csv" # timestamp (and eventually waveform) data
filename_data_csv_gemse = "DataR_CH0@DT5781A_793_run.csv" # timestamp (and eventually waveform) data
filename_data_txt = "CH0@DT5781A_840_EspectrumR_run.txt" # adc spectrum data
filename_histogram_png = "histogram" # histogram plot name
filename_measurement_data_dict = "measurement_data.json"
filename_raw_data_ndarray = "raw_data.npy"
filename_raw_data_png = "raw_data.png"
filename_peak_data_png = "peak_data.png"
filename_activity_data_png = "activity_data.png"
filename_icp_ms_complot = "icp_ms__complot.png"


# keys: 'measurement_data'
key_measurement_information = "measurement_information"
key_raw_data = "raw_data"
key_spectrum_data = "spectrum_data"
key_activity_data = "activity_data"


# format
color_uni_blue = '#004A9B'
color_uni_red = '#C1002A'
color_monxe_cyan = "#00E8E8" # the cyan-like color that was used within the MonXe logo
color_histogram = "black"
color_histogram_error = color_monxe_cyan
linewidth_histogram_std = 0.8 # the standard linewidth for a histogram plot


# miscellaneous
n_adc_channels = 16383 # from channel 0 to channel 16383
adc_channel_min = 0
adc_channel_max = 16383


### hardware
# detection vessel
r_dv_m = 0.077 # detection vessel radius in meters
h_dv_m = 0.0095 # detection vessel cylinder height in meters
w_dv_m = 0.022 # detection vessel top flange thickness in meters
r_dv_cf16_m = 0.016/2 # detection vessel top flange CF16 hole radius in meters (6x)
r_dv_cf35_m = 0.038/2 # detection vessel top flange CF35 hole radius in meters
v_hemi = (4/3 *r_dv_m**3 *math.pi *0.5)
v_cyl = (r_dv_m**2 *math.pi *h_dv_m)
v_cf16 = 6*(r_dv_cf16_m**2 *math.pi *w_dv_m)
v_cf35 = 1*(r_dv_cf35_m**2 *math.pi *w_dv_m)
v_dv_m3 = v_hemi +v_cyl +v_cf16 +v_cf35 # detection vessel volume in cubic meters, cylinder +hemisphere +6*CF16 +1*CF35
# emanation vessel
r_ev_m = 0.254/2 # emanation vessel radius in meters
l_ev_m = 0.400 # emanation vessel tube length in meters
w_ev_m = 0.026 # emanation vessel top flange thickness in meters
r_ev_cf16_m = 0.0175/2 # emanation vessel top flange CF16 hole radius in meters (3x)
r_ev_cf35_m = 0.0385/2 # emanation vessel top flange CF35 hole radius in meters (4x)
v_ev_m3 = (r_ev_m**2 *math.pi *l_ev_m) +3*(r_ev_cf16_m**2 *math.pi *w_ev_m) +4*(r_ev_cf35_m**2 *math.pi *w_ev_m)


# isotope data
# data taken from: http://www.lnhb.fr/nuclear-data/nuclear-data-table/
isotope_dict = {
    # radium
    "ra226" : {
        "isotope" : "ra226",
        "half_life_s" : 1600 *365 *24 *60 *60, # 1600 a in seconds
        "q_value_kev" : 4870.62,
        "alpha_energy_kev" : 4784.35,
        "decay_constant" : np.log(2)/(1600 *365 *24 *60 *60),
        "color" : "#00b822", # green
        "latex_label" : r"$^{226}\,\mathrm{Ra}$",
    },
    # radon
    "rn222" : {
        "isotope" : "rn222",
        "half_life_s" : 3.8232 *24 *60 *60, # 3.8232 d in seconds
        "q_value_kev" : 5590.3,
        "alpha_energy_kev" : 5489.48,
        "decay_constant" : np.log(2)/(3.8232 *24 *60 *60),
        #"color" : color_monxe_cyan,
        "color" : miscfig.colorstring_citirok1,
        "latex_label" : r"$^{222}\,\mathrm{Rn}$",
    },
    # polonium
    "po218" : {
        "isotope" : "po218",
        "half_life_s" : 3.071 *60, # 3.071 min
        "q_value_kev" : 6114.68,
        "alpha_energy_kev" : 6002.35,
        "decay_constant" : np.log(2)/(3.071 *60),
        "color" : "#ab00ff", # (https://colordesigner.io/gradient-generator)
        "latex_label" : r"$^{218}\,\mathrm{Po}$",
    },
    "po216" : {
        "isotope" : "po216",
        "half_life_S" : 0.148,
        "decay_constant" : np.log(2)/(0.148),
        "q_value_kev" : 6906.3,
        "alpha_energy_kev" : 6778.4,
        "color" : "#93009b",
        "latex_label" : r"$^{216}\,\mathrm{Po}$",
    },
    "po214" : {
        "isotope" : "po214",
        "half_life_s" : 162.3 *10**(-6), # 162.3 µs
        "decay_constant" : np.log(2)/(162.3 *10**(-6)),
        "q_value_kev" : 7833.46,
        "alpha_energy_kev" : 7686.82,
        "color" : "#f07000",
        "latex_label" : r"$^{214}\,\mathrm{Po}$",
    },
    "po212" : {
        "isotope" : "po212",
        "half_life_s" : 300 *10**(-9), # 300 nanoseconds
        "q_value_kev" : 8954.12,
        "alpha_energy_kev" : 8785.17,
        "decay_constant" : np.log(2)/(300 *10**(-9)),
        "color" : '#c40076',
        "latex_label" : r"$^{212}\,\mathrm{Po}$",
    },
    "po210" : {
        "isotope" : "po210",
        "half_life_s" : 138.3763 *24 *60 *60, # 138 days in seconds
        "q_value_kev" : 5407.45,
        "alpha_energy_kev" : 5304.33,
        "decay_constant" : np.log(2)/(138.3763 *24 *60 *60),
        #"color" : '#e30071',
        "color" : 'black',
        "latex_label" : r"$^{210}\,\mathrm{Po}$",
    },
    # lead
    "pb214" : {
        "isotope" : "pb214",
        "half_life_s" : 26.916 *60, # 26.916 min
        "decay_constant" : np.log(2)/(26.916 *60),
        "color" : color_uni_red, #"#ff1100", # red
        "latex_label" : r"$^{214}\,\mathrm{Pb}$",
    },
    # bismuth
    "bi214" : {
        "isotope" : "bi214",
        "half_life_s" : 19.8 *60, # 19.8 min
        "decay_constant" : np.log(2)/(19.8 *60),
        "color" : "#00b822", # green
        "latex_label" : r"$^{214}\,\mathrm{Bi}$",
    },
}


color_dict_ev = {
    "ev_cf250pp_1" : "yellow",
    "ev_cf35fn_1" : "orange",
}


# half lives
t_half_222rn = 3.8232 *24 *60 *60 # 3.8232 d
t_half_218po = 3.071 *60 # 3.071 min 
t_half_214pb = 26.916 *60 # 26.916 min
t_half_214bi = 19.8 *60 # 19.8 min
t_half_214po = 162.3 *10**(-6) # 162.3 µs
# decay constants
lambda_222rn = np.log(2)/t_half_222rn
lambda_218po = np.log(2)/t_half_218po
lambda_214pb = np.log(2)/t_half_214pb
lambda_214bi = np.log(2)/t_half_214bi
lambda_214po = np.log(2)/t_half_214po


# analysis
activity_interval_h = 3




#######################################
### Generic Helper Functions
#######################################


def detection_efficiency_correction(r_ema, detection_efficiency):
    return r_ema/detection_efficiency


# This function is used to convert a datestring (as I defined it, e.g. '20200731') into a format that can be handled by 'datetime'.
def mod_datetimestring(input_string):
    y = input_string[2:4]
    m = input_string[4:6]
    d = input_string[6:8]
    H = input_string[9:11]
    M = input_string[11:13]
    if len(input_string) == 15:
        S = input_string[13:15]
    else:
        S = r"00"

    return d +r"/" +m +r"/" +y +r" " +H +r":" +M +r":" +S


def error_propagation_for_one_dimensional_function(
    function,
    function_input_dict = {},
    function_parameter_dict = {},
    n_mc = 10**5,
    n_histogram_bins = 150,
    flag_verbose = False,
):
    """
    idea:
        This function is used to calculate the error propagation for a one dimensional real function (i.e., $\mathbb{R}^n \rightarrow \mathbb{R}$).
        Therefore MC data is generated according to the (potentially asymmetric) input distribution.
        This simulation dataset for each input variable is approximated by a Gaussian distribution with 'n_mc'/2-many events simulated for the lower and upper error intervals, respectively.
        Afterward, for every simulated input variable value the corresponding output function value is computed according to function 'function'.
        Finally, The left- and right-sided 34% interval of the y-data array is computed and returned as output.
    inputs:
        'function': function to compute the error propagation for
        'function_input_dict' : 
        'param_dict': dictionary, parameters passed on to above function, f(x,**param_dict)
        'n_mc': int divisible by 2, size of the MC population
        'n_histogram_bins': int, binning of the function output value histogram that will be used to determine the asymmetric width of the output distribution
    returns:
        function_output_mean: float, f(x)
        function_output_loweruncertainty: float, -34% error interval left of the function output mean value
        function_output_upperuncertainty: float, +34% error interval right of the function output mean value
    """

    # generating a dict containing all the error-containing keyword inputs that will be passed to the function
    function_input_list = list(set(["_".join(list(key.split("_"))[:-1]) for key in [*function_input_dict]])) # contains all keywords from 'function_input_dict' without the '_mean', '_loweruncertainty', and '_upperuncertainty' additions, those are the keyword inputs that 'function' requires.
    if flag_verbose == True: print(function_input_list)
    function_mc_input_dict = {}
    for k in function_input_list:
        function_mc_input_dict.update({k : []})
    if flag_verbose == True: print(f"epfodf(): 'function_input_list' = {function_input_list}")
    if flag_verbose == True: print(f"epfodf(): function parameters = {[*function_parameter_dict]}")

    # generating MC data by separately generating n_mc/2-many left- and right-sided Gaussian-distributed events
    rng = np.random.default_rng(seed=42)
    ctr = 0
    while ctr < n_mc/2: # generate only 'n_mc'-many MC data points, for every value of 'ctr' one value lower and one value higher than the current mean value are appended to 'function_mc_input_dict'
        for input_keyword in function_input_list:
            flag_is_lower_than_mean = False
            while flag_is_lower_than_mean == False:
                left_sided_data_point = rng.normal(function_input_dict[input_keyword +"_mean"], function_input_dict[input_keyword +"_loweruncertainty"], 1)[0]
                if left_sided_data_point <= function_input_dict[input_keyword +"_mean"]:
                    flag_is_lower_than_mean = True 
            function_mc_input_dict[input_keyword].append(left_sided_data_point)
            flag_is_higher_than_mean = False
            while flag_is_higher_than_mean == False:
                right_sided_data_point = rng.normal(function_input_dict[input_keyword +"_mean"], function_input_dict[input_keyword +"_upperuncertainty"], 1)[0]
                if right_sided_data_point >= function_input_dict[input_keyword +"_mean"]:
                    flag_is_higher_than_mean = True 
            function_mc_input_dict[input_keyword].append(right_sided_data_point)
        ctr += 1
        if flag_verbose == True and ctr % 500 == 0: print(f"epfodf(): generated {2*ctr} MC values to pass into 'function'")


    # shuffling the MC data lists of 'function_mc_input_dict' (otherwise function would feed 'n_mc'/2 fully left-fluctuation values and afterwards 'n_mc'/2 right-fluctuating values into function)
    for kw in [*function_mc_input_dict]:
        function_mc_input_dict[kw] = random.sample(function_mc_input_dict[kw], len(function_mc_input_dict[kw]))
    if flag_verbose == True:
        for kw in [*function_mc_input_dict]:
            print(f"epfodf(): distribution of '{kw}' function input values")
            plt.hist(x=function_mc_input_dict[kw], bins=n_histogram_bins)
            plt.show()
    
    # computing the mean function output
    function_input_mean_dict = {}
    for kw in function_input_list:
        function_input_mean_dict.update({kw : function_input_dict[kw +"_mean"]})
    function_output_mean = function(**function_input_mean_dict, **function_parameter_dict)

    # computing the 'function' MC output values
    function_output_mc_data = []
    for i in range(n_mc):
        function_mc_input_dict_temp = {}
        for kw in function_input_list:
            function_mc_input_dict_temp.update({kw : function_mc_input_dict[kw][i]})
        function_mc_val = function(**function_mc_input_dict_temp, **function_parameter_dict)
        function_output_mc_data.append(function_mc_val)
    if flag_verbose == True:
        print(f"epfodf(): distribution of function MC data")
        plt.hist(x=function_output_mc_data, bins=n_histogram_bins)
        plt.show()
        
    # determining the left- and right-sided widths of the distribution
    y_data_counts, y_data_bin_edges = np.histogram(a=function_output_mc_data, bins=n_histogram_bins)
    y_data_bin_centers = [edge +0.5*(y_data_bin_edges[2]-y_data_bin_edges[1]) for edge in y_data_bin_edges[:-1]]
    y_center, function_output_loweruncertainty, function_output_upperuncertainty = get_asymmetric_intervals_around_center_val_for_interpolated_discrete_distribution(
        distribution_bin_centers = y_data_bin_centers,
        distribution_counts = y_data_counts,
        distribution_center_value = function_output_mean,
        interval_width_lower_percent = 0.6827/2,
        interval_width_upper_percent = 0.6827/2,
        ctr_max = 10**6,
        granularity = 100,
        flag_verbose = flag_verbose)
    
    return function_output_mean, function_output_loweruncertainty, function_output_upperuncertainty


def calc_detection_efficiency(measurement_val, reference_val):
    detection_efficiency = measurement_val/reference_val
    return detection_efficiency


def calc_reduced_chi_square(
    y_data_obs, # array, observed values
    y_data_exp, # array, expected (i.e., fit) values
    y_data_err, # array, errors of observed values
    ddof, # integer, nu = n -m, n being len(y_data_obs) and m being the number of fitted parameters
    y_data_err_lower = [], # array, lower errors of observed values
    y_data_err_upper = [], # array, upper errors of observed values
):
    """
    https://www.astroml.org/book_figures/chapter4/fig_chi2_eval.html
    https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    """

    # calculating the mean errors if asymmetric ones are given
    if y_data_err == []:
        y_errors = [0.5*(y_data_err_lower[i] +y_data_err_upper[i]) for i in range(len(y_data_obs))]
    else:
        y_errors = y_data_err

    # calculating the reduced chi-square test statistic
    red_chi_square = 0
    for i, y_val in enumerate(y_data_obs):
        red_chi_square += ((y_data_obs[i]-y_data_exp[i])/(y_errors[i]))**2
    nu = len(y_data_obs)-ddof
    red_chi_square = red_chi_square/nu

    return red_chi_square


# This function is used to convert a datetime string (as defined by datetime, e.g. '31-07-20 15:31:25') into a datetime object.
def convert_string_to_datetime_object(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%d/%m/%y %H:%M:%S')
    return datetime_obj
#def convert_string_to_datetime_object(datetime_str):
#    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M')
#    return datetime_obj
#convert_string_to_datetime_object(datetime_str="20200731_1530")


# This function is used to convert a timestamp (or an iterable of timestamps) from one format to another.
# exemplary use:
#     i = [np.datetime64('today'), "20210531_1200", datetime.datetime.now()][0]
#     i = [[np.datetime64('today'),np.datetime64('today')], ["20210531_1200", "20210531_1200"], [datetime.datetime.now(),datetime.datetime.now()]][0]
#     print(f"i: {i}, type of i: {type(i)}")
#     f = timestamp_conversion(i, "str")
#     print(f"f: {f}, type of f: {type(f)}")
def timestamp_conversion(
    input_timestamp,
    flag_output_type = [
        "string", "str",
        "np.datetime64", "np.dt64",
        "datetime.datetime", "dt.dt"][4],
    flag_input_timestamp_string_format = "%Y%m%d_%H%M",
    flag_output_timestamp_string_format = "%Y%m%d_%H%M",):
    # figuring out the input data type whether it's an iterable or not
    if not hasattr(input_timestamp, "__len__") or type(input_timestamp)==str:
        input_data_type = type(input_timestamp)
        timestamp_i_list = [input_timestamp]
    else:
        input_data_type = type(input_timestamp[0])
        timestamp_i_list = input_timestamp
    # timestamp_i: conversion of input to a list of 'datetime.datetime' elements
    if input_data_type == str:
        timestamp_i_list = [datetime.datetime.strptime(timestamp_i, flag_input_timestamp_string_format) for timestamp_i in timestamp_i_list]
    elif input_data_type == datetime.datetime:
        timestamp_i_list = [timestamp_i for timestamp_i in timestamp_i_list]
    elif input_data_type == np.datetime64:
        timestamp_i_list = [timestamp_i.astype(datetime.datetime) for timestamp_i in timestamp_i_list]
    else:
        raise Exception(f"'timestamp_conversion()': unknown input timestamp format '{input_data_type}'")
    # timestamp_f: conversion of 'timestamp_i' to 'flag_output_type'
    if flag_output_type in ["string", "str"]:
        timestamp_f_list = [timestamp_i.strftime(flag_output_timestamp_string_format) for timestamp_i in timestamp_i_list]
    elif flag_output_type in ["np.datetime64", "np.dt64"]:
        timestamp_f_list = [np.array([timestamp_i]).astype(np.datetime64)[0] for timestamp_i in timestamp_i_list]
    elif flag_output_type in ["datetime.datetime", "dt.dt"]:
        timestamp_f_list = [timestamp_i for timestamp_i in timestamp_i_list]
    else:
        raise Exception(f"'timestamp_conversion()': unknown output timestamp format '{flag_output_type}'")
    # returning the output timestamp (list)
    if not hasattr(input_timestamp, "__len__") or type(input_timestamp)==str:
        return timestamp_f_list[0]
    else:
        return timestamp_f_list


# This function is used to calculate the mean and error of combined measurements of the same physical quantity for a list of measured values and corresponding errors according to Gaussian error propagation.
def calc_weighted_mean_with_error(
    measurements,
    errors):
    weighted_mean = np.sum([measurements[i]/errors[i]**2 for i in range(len(measurements))]) / np.sum([err**(-2) for err in errors])
    error = np.sqrt(1/(np.sum([err**(-2) for err in errors])))
    return [weighted_mean, error]


def calc_weighted_mean(**kwargs):
    """
    This function is used to calculate the weighted mean for arbitrary many measurements of the same quantity with (potentially) asymmetric errors.
    inputs:
        **kwargs: dict, dictionary that contains an arbitrary amount of measurements including their lower and upper uncertainties
                  The input dict needs to have the following syntax:
                  measurements = {
                      "a" : 4.2,
                      "a_loweruncertaintyparam" : 1.2,
                      "a_upperuncertaintyparam" : 2.2,
                      "b" : 4.3,
                      "b_loweruncertaintyparam" : 0.9,
                      "b_upperuncertaintyparam" : 2.1,
                      ...
                  }
                  Per measurement, there must be exactly three inputs: The measurement itself, plus its lower and upper uncertainties (denoted with e.g., "_loweruncertainty")
                  The reason for this is, that you can then just use the function 'error_propagation_for_one_dimensional_function' to calculate the (asymmetric) error of the weighted mean.
                  Search for '# calculating the combined value of n_rn222_t_meas_i (currently the weighted mean of the po218 and po214 numbers)' to view an example.
    returns:
        weighted_mean: float, the weighted mean calculated from all input measurements
    """
    
    # determing the input measurements
    meas_input_list = list(set([k for k in [*kwargs] if "param" not in k])) # contains all keywords from 'function_input_dict' without the '_mean', '_loweruncertainty', and '_upperuncertainty' additions, those are the keyword inputs that 'function' requires.
    means_list = []
    loweruncertainty_list = []
    upperuncertainty_list = []
    for k in meas_input_list:
        means_list.append(kwargs[k])
        loweruncertainty_list.append(kwargs[k +"_loweruncertaintyparam"])
        upperuncertainty_list.append(kwargs[k +"_upperuncertaintyparam"])
        
    # determining the weights
    weights_list = []
    weighted_means_list = []
    for i in range(len(means_list)):
        weight = 1/((0.5*(loweruncertainty_list[i] +upperuncertainty_list[i]))**2)
        weights_list.append(weight)
        weighted_means_list.append(weight*means_list[i])

    # calculating the weighted mean
    weighted_mean = np.sum(weighted_means_list)/np.sum(weights_list)

    return weighted_mean


# This function is used to calculate the ratio of following Gaussian error propagation.
def calc_ratio_with_error(numerator, denominator, err_num, err_denom):
    ratio_mean = numerator/denominator
    ratio_error = np.sqrt(((1/denominator)*err_num)**2 +((numerator/denominator**2)*err_denom)**2)
    return [ratio_mean, ratio_error]


# one possible energy-channel relation: linear
def function_linear_vec(x, m, t):
    if hasattr(x, "__len__"):
        return [m*xi +t for xi in x]
    else:
        return m*x +t


# This function is used to extrapolate the exponential decay/rise in radon activity.
# asymptotic exponential rise: a(t) = a_ema *(1-exp(-lambda_222rn*dt))
# exponential decay: a(t) = a_t_0 *exp(-lambda_222rn*dt)
# NOTE: 'time delta' (dt) refers to the time since t_0
def extrapolate_radon_activity(
    dt_extrapolation_s, # time delta at which the function chosen below is extrapolated
    known_activity_at_dt_known_bq, # known activity (in Becquerel) at ...
    known_dt_s, # ... known time delta (in seconds)
    #known_activity_error_at_dt_known_bq = 0, # error of known activity at known time
    flag_exp_rise_or_decay = ["rise", "decay"][0], # flag determining wheter to extrapolate exponential rise or decay
    lambda_222rn = isotope_dict["rn222"]["decay_constant"]): # 222rn decay constant
    # exponential decay
    if flag_exp_rise_or_decay == "decay":
        a_t_0 = known_activity_at_dt_known_bq *np.exp(lambda_222rn *known_activity_at_dt_known_bq)
        a_t_1 = a_t_0 *np.exp(-lambda_222rn *dt_extrapolation_s)
        #a_t_1_error = np.exp(lambda_222rn*known_dt_s) *np.exp(-lambda_222rn*dt_extrapolation_s) *known_activity_error_at_dt_known_bq
        return a_t_1
    # asymptotic exponential rise
    elif flag_exp_rise_or_decay == "rise":
        a_ema = known_activity_at_dt_known_bq/(1-np.exp(-lambda_222rn *known_dt_s))
        #a_ema_error = (1/(1-np.exp(-lambda_222rn*known_dt_s))) *known_activity_error_at_dt_known_bq
        if dt_extrapolation_s == "inf":
            return a_ema
        else:
            a_t1 = a_ema *(1-np.exp(-lambda_222rn *dt_extrapolation_s))
            #a_t1_error = a_ema_error *(1-np.exp(-lambda_222rn *dt_extrapolation_s))
            return a_t1


# This function is used to retrieve a Python3 dictionary stored as a .json file.
def get_dict_from_json(input_pathstring_json_file):
    with open(input_pathstring_json_file, "r") as json_input_file:
        ret_dict = json.load(json_input_file)
    return ret_dict


# This function is used to save a Python3 dictionary as a .json file.
def write_dict_to_json(output_pathstring_json_file, save_dict):
    with open(output_pathstring_json_file, "w") as json_output_file:
        json.dump(save_dict, json_output_file, indent=4)
    return


# This function is used to update and save the 'measurement_data' dictionary.
def update_and_save_measurement_data(
    abspath_measurement_data_dict,
    update_dict):

    # retrieving the 'measurement_data' dictionary
    try:
        measurement_data_dict = get_dict_from_json(input_pathstring_json_file=abspath_measurement_data_dict)
    except:
        measurement_data_dict = {}
        print(f"update_and_save_measurement_data(): generated new and empty 'measurement_data_dict'")

    # updating and saving the 'measurement_data' dict
    measurement_data_dict.update(update_dict)
    write_dict_to_json(abspath_measurement_data_dict,measurement_data_dict)
    
    # printing the 'update_dict' data
    print(f"update_and_save_measurement_data(): added the following dictionary to 'measurement_data.json'\n")
    print(json.dumps(update_dict, indent=4, sort_keys=True))
    
    # returning None
    return




#######################################
### Raw Data
#######################################


# This is the dtype list used to generate the MCA raw data ndarray
raw_data_dtype_list = [
    ("timestamp_ps", np.uint64), # timestamp in ps
    ("pulse_height_adc", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("flag_mca", np.unicode_, 16), # flag extracted from the mca list file
#    ("wfm_data", np.int16, n), # added if 'flag_get_wfm_data' is set to True in 'get_raw_data_from_mca_list_file()'
]


# This is the dtype used for the MCA raw data ndarray
raw_data_dtype = np.dtype(raw_data_dtype_list)


# This is the dtype used for raw data extracted from CoMPASS.
raw_data_dtype_custom_pha = np.dtype([
    ("timestamp_ps", np.uint64), # timestamp in ps
    ("pulse_height_adc", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("pulse_height_adc__fitheight", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("pulse_height_adc__fitasymptote", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("flag_mca", np.unicode_, 16), # flag extracted from the mca list file
    ("flag_pha", np.unicode_, 16), # flag added by the custom pulse height algorithm
])


# This function is used as the fit function to extract the pulse height with the 'compass_custom' algorithm.
def fitfunction__independent_exponential_rise_and_fall(x, y_baseline, x_rise, a_rise, lambda_rise, x_decay, a_decay, lambda_decay):
    if x >= x_rise:
        y_rise = a_rise*(1-np.exp(-lambda_rise*(x-x_rise)))
    else:
        y_rise = 0
    if x >= x_decay:
        y_decay = -a_decay*(1-np.exp(-lambda_decay*(x-x_decay)))
    else:
        y_decay = 0
    return y_baseline +y_rise +y_decay


# This function is used as the fit function to extract the pulse height with the 'compass_custom' algorithm in its vectorized form.
def fitfunction__independent_exponential_rise_and_fall__vec(x, y_baseline, x_rise, a_rise, lambda_rise, x_decay, a_decay, lambda_decay):
    y = np.zeros_like(x)
    for i in range(len(y)):
        y[i]=fitfunction__independent_exponential_rise_and_fall(x[i], y_baseline, x_rise, a_rise, lambda_rise, x_decay, a_decay, lambda_decay)
    return y


def get_raw_data_ndarray_from_mca_list_file(
    abspath_list_file, # abspath of the input MCA list file
    flag_abspath_output_ndarray = [], # list of abspaths at which the retrieved raw data ndarray should be saved
    flag_ctr = 10**10, # flag indicating the number of processed events
    flag_multiple_list_files = [False, True][0], # flag indicating whether data is retrieved from only one or many files matching the 'abspath_list_file' input string
    flag_get_waveform_data = [False, True][0], # flag indicating whether waveform data should be extracted as well, can be False, True, or a list with the abspath of the output and relevant time stamps ["/home/daniel/Desktop/wfm_data.npy", [178978375, 2485789275]]
    flag_ignored_list_files = []): # flag indicating the ignored files
    
    """
    This function is used to load the raw data from the MCA list file generated either by CoMPASS or MC2Analyzer.
    """

    ### initialization
    t_i = datetime.datetime.now()
    thisfunctionname = "get_raw_data_from_mca_list_file"

    ### retrieving the abspaths of the desired MCA list files
    print(f"{thisfunctionname}(): searching for\n '{abspath_list_file}'\n")
    # retrieving information from the 'abspath_list_file' input
    abspath_list_files_directory = "/".join(abspath_list_file.split("/")[:-1]) +"/"
    filename_list_file = abspath_list_file.split("/")[-1]
    abspath_list_files_list = [abspath_list_file]
    flag_daq = "CoMPASS" if filename_list_file.endswith(".csv") else "MC2"
    flag_daq_extension = ".csv" if filename_list_file.endswith(".csv") else ".txt"
    # searching for multiple files according to the 'flag_multiple_list_files' input
    if type(flag_multiple_list_files) == list:
        abspath_list_files_list = abspath_list_files_list +flag_multiple_list_files
    elif flag_multiple_list_files == True:
        candidate_filename_list = os.listdir(abspath_list_files_directory)
        for filename in candidate_filename_list:
            if os.path.isfile(abspath_list_files_directory +filename):
                if filename.endswith(flag_daq_extension):
                    if filename != filename_list_file:
                        filename_composition_list = re.split("\W+|_", filename) # https://stackoverflow.com/questions/1059559/split-strings-into-words-with-multiple-word-boundary-delimiters (accessed: 28th October 2021)
                        n_possible_matches = len(filename_composition_list)
                        n_unmatches = 3
                        n_matches = len([True for part in filename_composition_list if part in filename_list_file])
                        if n_matches >= n_possible_matches -n_unmatches:
                            abspath_list_files_list.append(abspath_list_files_directory +filename)
            else:
                pass
    elif flag_multiple_list_files == False:
        pass
    else:
        raise Exception("\tthe 'flag_multiple_list_files' input must be in [False, True, ['abspath/to/another/list_file']]")
    # printing the outcome of the file search
    print(f"{thisfunctionname}(): found the following {len(abspath_list_files_list)} MCA list files recorded with {flag_daq}:")
    for abspath in abspath_list_files_list:
        print(f"\t - {abspath.split('/')[-1]}")
    print("")

    ### looping over the list files and writing the data into 'timestamp_tuple_list'
    print(f"{thisfunctionname}(): starting data retrieval")
    timestamp_data_tuplelist = [] # this list will later be cast into a structured array
    ctr = 0 # counter tracking the number of processed entries
    for abspath_list_file in [pathstring for pathstring in abspath_list_files_list if all([True if ignored_filename not in pathstring else False for ignored_filename in flag_ignored_list_files])]:
        #truelist = [True if blacklistentry not in abspath_list_file else False for blacklistentry in input_filename_blacklist]
        with open(abspath_list_file) as input_file:
            print(f"\t - {abspath_list_file.split('/')[-1]}")
            for line in input_file:
                # CoMPASS
                if flag_daq in ["CoMPASS"]:
                    if line.startswith("BOA"):
                        continue
                    elif ctr <= flag_ctr:
                        line_list = list(line.split(";"))
                        board = np.uint64(line_list[0]) # irrelevant
                        channel = np.uint64(line_list[1]) # irrelevant
                        timestamp_ps = np.uint64(line_list[2]) # timestamp in picoseconds
                        pulse_height_adc = np.uint64(line_list[3]) # pulse height in adc determined via trapezoidal filter
                        flag_mca = line_list[4] # information flag provided by CoMPASS
                        wfm_data_list = [int(wfm_sample) for wfm_sample in line_list[5:]]
                # MC2
                elif flag_daq == "MC2":
                    if line.startswith("HEADER"):
                        continue
                    elif ctr <= flag_ctr:
                        line_list = list(line.split())
                        timestamp_ps = 10000*np.uint64(line_list[0]) # the MCA stores timestamps in clock cycle units (one clock cycle corresponds to 10ns, 10ns = 10000ps)
                        pulse_height_adc = np.int64(line_list[1])
                        flag_mca = line_list[2]
                # filling the 'timestamp_data_tuple' into 'timestamp_data_tuple_list'
                timestamp_data_tuple = (timestamp_ps, pulse_height_adc, flag_mca) if flag_daq in ["MC2"] or flag_get_waveform_data == False else (timestamp_ps, pulse_height_adc, flag_mca, wfm_data_list)# exported data corresponding to 'raw_data_dtype'
                if flag_get_waveform_data in [False, True]:
                    timestamp_data_tuplelist.append(timestamp_data_tuple)
                    ctr += 1
                else:
                    if timestamp_ps in flag_get_waveform_data:
                        timestamp_data_tuplelist.append(timestamp_data_tuple)
                        ctr += 1
                if ctr%1000==0:
                    print(f"\t----> {ctr} events processed")
    print("")

    ### storing the 'timestamp_data_tuple_list' in the numpy structured array 'retarray'
    # saving the array locally
    dt = np.dtype(raw_data_dtype_list) if flag_get_waveform_data == False else np.dtype(raw_data_dtype_list +[("wfm_data", np.int16, len(timestamp_data_tuplelist[0][-1]))])
    retarray = np.array(timestamp_data_tuplelist, dt)
    retarray = np.sort(retarray, order="timestamp_ps")
    for abspath_output in flag_abspath_output_ndarray:
        np.save(abspath_output, retarray)
    print(f"{thisfunctionname}(): saved '{flag_abspath_output_ndarray}'")
    print("")
    # ghjgh
    t_f = datetime.datetime.now()
    t_run = t_f -t_i
    print(f"{thisfunctionname}(): retrieval time: {t_run} h")
    # returning the raw data array
    return retarray


# This function is used to 
def get_raw_data_dict(raw_data_ndarray):

    # initializing the 'misc_meas_information_dict'
    raw_data_dict = {
        "measurement_duration_days" : get_measurement_duration(list_file_data=raw_data_ndarray, flag_unit='days'),
        "recorded_events" : {
            "total" : len(raw_data_ndarray),
            "thereof_in_ch0" : len(raw_data_ndarray[(raw_data_ndarray['pulse_height_adc'] == 0)]),
            "thereof_in_first_50_adcc" : len(raw_data_ndarray[(raw_data_ndarray['pulse_height_adc']<50)]),
            "thereof_in_negative_adcc" : len(raw_data_ndarray[(raw_data_ndarray['pulse_height_adc'] < 0)]),
            "thereof_above_max_adcc" : len(raw_data_ndarray[(raw_data_ndarray['pulse_height_adc'] > adc_channel_max)]),
        },
        "mca_flags" : {},
    }

    # adding mca flags
    mca_flag_list = []
    for i in range(len(raw_data_ndarray)):
        if raw_data_ndarray[i]["flag_mca"] not in mca_flag_list:
            mca_flag_list.append(raw_data_ndarray[i]["flag_mca"])
    for i in range(len(mca_flag_list)):
        raw_data_dict["mca_flags"].update({mca_flag_list[i] : len(raw_data_ndarray[(raw_data_ndarray['flag_mca'] == mca_flag_list[i])])})

    # return the 'misc_meas_dict'
    return raw_data_dict
    

# This function is used to infer the duration of a measurement. The timestamps listed by CoMPASS are given in picoseconds.
def get_measurement_duration(
    list_file_data,
    flag_unit = ["days", "minutes", "seconds"][0]
):
    conv_dict = {
        "days" : 24 *60* 60* 1000* 1000* 1000 *1000,
        "minutes" : 60* 1000* 1000* 1000 *1000,
        "seconds" : 1000* 1000* 1000 *1000,
    }
    t_ps = list_file_data[len(list_file_data)-1]["timestamp_ps"]
    return t_ps *(1/conv_dict[flag_unit])


# This function is used to plot a waveform from MCA raw data.
def plot_mca_waveform(
    wfm_x_data,
    wfm_y_data,
    plot_xlabel = r"record time  / $\mathrm{\mu s}$",
    plot_ylabel = r"$U_{\mathrm{sig}}^{\mathrm{MCA}}(t)$ / $\mathrm{adc\,\,channels}$",
    plot_ylabel_voltageaxis = r"$U_{\mathrm{sig}}^{\mathrm{MCA}}(t)$ / $\mathrm{V}$",
    plot_linewidth = 0.5,
    plot_linecolor = miscfig.uni_blue,
    plot_x_lim = [0, 1, "rel"], # x-axis boundaries, relative to 'wfm_x_data' min and max
    plot_y_lim = [-0.1, 1.1, "rel"], # y-axis boundaries, relative to wfm_y_data[0] and wfm_y_data[1]
    plot_labelfontsize = 12,
    plot_aspect_ratio = 9/16,
    plot_figsize_x_inch = miscfig.standard_figsize_x_inch,
    flag_output_abspath_list = False,
    flag_show_voltage_axis = False,
    flag_show = True,
    flag_comments_list = False,
    comments_textcolor = "black",
    comments_fontsize = 11,
    comments_linesep = 0.1,
    comments_textpos = [0.025, 0.9],
    ):
    # canvas
    fig, ax1 = plt.subplots(figsize=[plot_figsize_x_inch,plot_figsize_x_inch*plot_aspect_ratio], dpi=150, constrained_layout=True)
    # axes
    if plot_x_lim[2] == "rel":
        ax1.set_xlim([wfm_x_data[0] +plot_x_lim[0]*(wfm_x_data[-1]-wfm_x_data[0]), wfm_x_data[0] +plot_x_lim[1]*(wfm_x_data[-1]-wfm_x_data[0])])
    elif plot_x_lim[2] == "abs":
        ax1.set_xlim(plot_x_lim[0], plot_x_lim[1])
    if plot_y_lim[2] == "rel":
        ax1.set_ylim([np.min(wfm_y_data) +plot_y_lim[0]*(np.max(wfm_y_data)-np.min(wfm_y_data)), np.min(wfm_y_data) +plot_y_lim[1]*(np.max(wfm_y_data)-np.min(wfm_y_data))])
    elif plot_y_lim[2] == "abs":
        ax1.set_ylim(plot_y_lim[0], plot_y_lim[1])
    ax1.set_xlabel(plot_xlabel, fontsize=plot_labelfontsize)
    ax1.set_ylabel(plot_ylabel, fontsize=plot_labelfontsize)
    # plotting
    plt.plot(
        wfm_x_data,
        wfm_y_data,
        linewidth = plot_linewidth,
        color = plot_linecolor,
        linestyle = '-')
    # annotations
    if flag_comments_list != False:
        annotate_comments(
            comment_ax = ax1,
            comment_list = flag_comments_list,
            comment_textpos = comments_textpos,
            comment_textcolor = comments_textcolor,
            comment_linesep = comments_linesep,
            comment_fontsize = comments_fontsize,
            flag_alignment = ["top_to_bottom", "symmetric"])
    # voltage axis
    if flag_show_voltage_axis == True:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(plot_ylabel_voltageaxis, fontsize=plot_labelfontsize, labelpad=5)
        #divr_v = [0,1]
        #y2_lim = [divr_v[0]*plot_y_lim[], divr_v[1]*] if plot_y_lim[3]=="rel" else 
        ax2.set_ylim([0,1])
        #ax2.tick_params(axis='y', labelcolor=color)
    # saving
    if flag_show:
        plt.show()
    if flag_output_abspath_list != False:
        for output_abspath in flag_output_abspath_list:
            fig.savefig(output_abspath)
    return





#######################################
### Histogram Stuff
#######################################


# This is the dtype used for histogram data.
histogram_data_dtype = np.dtype([
    ("bin_centers", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("counts", np.uint64),
    ("counts_errors_lower", np.uint64),
    ("counts_errors_upper", np.uint64)
])


# This function is used to calculate the Poissonian error of a number of counts.
def calc_poissonian_error(
    number_of_counts,
    flag_mode = ["sqrt", "poissonian", "poissonian_symmetrical"][1]
):
    # for a large number of entries
    if flag_mode == "sqrt":
        if number_of_counts == 0:
            poissonian_error = 1
        else:
            poissonian_error = np.sqrt(number_of_counts)
        return poissonian_error, poissonian_error
    # asymmetrical error; use "poissonian_symmetrical" for curve_fit
    elif flag_mode in ["poissonian", "poissonian_symmetrical"]:
        alpha = 0.318
        low, high = (chi2.ppf(alpha/2, 2*number_of_counts) / 2, chi2.ppf(1-alpha/2, 2*number_of_counts + 2) / 2)
        if number_of_counts == 0:
            low = 0.0
        low_interval = number_of_counts - low
        high_interval = high - number_of_counts
        if flag_mode == "poissonian":
            return low_interval, high_interval
        elif flag_mode == "poissonian_symmetrical":
            return max(low_interval, high_interval), max(low_interval, high_interval)
    # catching exceptions
    else:
        raise Exception("Invalid input: 'flag_mode'.")


# This function is used to convert raw timestamp data into histogram data
def get_histogram_data_from_timestamp_data(
    timestamp_data, # the timestamp data retrieved by 'get_timestamp_data'
    histval = "pulse_height_adc",
    number_of_bins = n_adc_channels, # the number of bins, per default every adc channel counts as one bin
    flag_errors = ["sqrt", "poissonian", "poissonian_symmetrical"][1],
):

    # calculating binwidth, bin centers and histogram data
    binwidth = (adc_channel_max-adc_channel_min)/(number_of_bins-1)
    data_histogram_adc_channels = np.arange(adc_channel_min, adc_channel_max +binwidth, binwidth)
    data_histogram_counts = np.histogram(
        a=timestamp_data[histval],
        bins=number_of_bins,
        range=(adc_channel_min -0.5*binwidth,adc_channel_max +0.5*binwidth)
    )[0]

    # casting the rebinned date into an ndarray
    histogram_data_tuplelist = []
    for i in range(len(data_histogram_adc_channels)):
        histogram_data_tuplelist.append((
            data_histogram_adc_channels[i],
            data_histogram_counts[i],
            calc_poissonian_error(data_histogram_counts[i], flag_mode=flag_errors)[0], # lower errors
            calc_poissonian_error(data_histogram_counts[i], flag_mode=flag_errors)[1], # upper errors
        ))
    histogram_data = np.array(histogram_data_tuplelist, histogram_data_dtype)
    return histogram_data


# This function is used to stepize arbitrary histogram data.
# I.e. it takes two list-like objects representing both the bin centers and also the corresponding counts and calculates two new lists containing both the left and right edges of the bins and two instances of the counts.
def stepize_histogram_data(
    bincenters,
    counts,
    counts_errors_lower = [],
    counts_errors_upper = [],
    flag_addfirstandlaststep = True):

    # calculating the binwidth and initializing the lists
    binwidth = bincenters[1] -bincenters[0]
    bincenters_stepized = np.zeros(2*len(bincenters))
    counts_stepized = np.zeros(2*len(counts))
    counts_errors_lower_stepized = np.zeros(2*len(counts_errors_lower))
    counts_errors_upper_stepized = np.zeros(2*len(counts_errors_upper))

    # stepizing the data
    for i in range(len(bincenters)):
        bincenters_stepized[2*i] = bincenters[i] -0.5*binwidth
        bincenters_stepized[(2*i)+1] = bincenters[i] +0.5*binwidth
        counts_stepized[2*i] = counts[i]
        counts_stepized[2*i+1] = counts[i]
    for i in range(len(counts_errors_lower)):
        counts_errors_lower_stepized[2*i] = counts_errors_lower[i]
        counts_errors_lower_stepized[2*i+1] = counts_errors_lower[i]
    for i in range(len(counts_errors_upper)):
        counts_errors_upper_stepized[2*i] = counts_errors_upper[i]
        counts_errors_upper_stepized[2*i+1] = counts_errors_upper[i]

    # appending a zero to both the beginning and end so the histogram can be plotted even nicer
    bin_centers_stepized_mod = [bincenters_stepized[0]] +list(bincenters_stepized) +[bincenters_stepized[len(bincenters_stepized)-1]]
    counts_stepized_mod = [0] +list(counts_stepized) +[0]

    # returning
    if flag_addfirstandlaststep==False:
        return bincenters_stepized, counts_stepized, counts_errors_lower_stepized, counts_errors_upper_stepized
    else:
        return bincenters_stepized, counts_stepized, counts_errors_lower_stepized, counts_errors_upper_stepized, bin_centers_stepized_mod, counts_stepized_mod


# This function is used to load the 'documentation.json' file and plot the respective comments (i.e. keys) onto a histogram plot
def annotate_documentation_json(
    annotate_ax, # ax object to be annotated
    filestring_documentation_json = "./documentation.json", # filestring determining which 'documentation.json' file to load
    text_fontsize = 11, # font size of the annotated text
    text_color = "black", # color of the annotated text
    text_x_i = 0.03, # x coordinate of the first text line (relative to the x axis)
    text_y_i = 0.75, # y coordinate of the first text line (relative to the y axis)
    text_parskip = 0.09, # text parskip
    flag_keys = "", # flag determining whether and in which order the keys are to be printed, default is printing all via ""
    flag_addduration = True, # flag determining whether the duration of the measurement is calculated
    flag_print_comment=False, # flag determining whether the 'comment' key should be printed or not
    flag_orientation="left" # flag determining whether the text is printed flushed right or left
):
    ### loading the data from the 'documentation.json' file
    with open(filestring_documentation_json) as json_file:
        doc_data_dict = json.load(json_file)
    ### preparing the text annotation
    ctr_textpos = 0
    # calculating the duration of the measurement
    if flag_addduration == True:
        t_i = datetime.datetime.strptime(doc_data_dict["start"], '%y/%m/%d %H:%M')
        t_f = datetime.datetime.strptime(doc_data_dict["end"], '%y/%m/%d %H:%M')
        t_delta = t_f -t_i
        doc_data_dict.update({"duration" : str(t_delta)})
    
    # determining which keys from 'documentation.json' are being printed onto the plot
    if flag_keys == "":
        keys_iterlist = sorted([*doc_data_dict])
    else:
        keys_iterlist = flag_keys
    ### annotating the comment keys retrieved from the .json file
    for key in keys_iterlist:
        if (key != "comment") or (key == "comment" and flag_print_comment == True):
            plt.text(
                x=text_x_i,
                y=text_y_i -ctr_textpos*text_parskip,
                s=r"\textbf{"+ key +r"}: " +doc_data_dict[key].replace("_","\_"),
                fontsize=text_fontsize,
                color=text_color,
                rotation=0,
                horizontalalignment=flag_orientation,
                verticalalignment='center',
                transform=annotate_ax.transAxes
            )
            ctr_textpos += 1
    return





#######################################
### Peak Data
#######################################


# Function to define a gaussian curve with amplitude "A", mean "mu" and sigma "sigma".
def function_gauss(x,A,mu,sigma):
    return A/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))


# Function to define a Crystall Ball curve.
# See Stefan Bruenner's Thesis (p.30) for more details.
def function_crystal_ball_one(x, mu, sigma, alpha, n, N) -> float: # evtl. N als Parameter
    A = (n/abs(alpha))**n *np.exp(-((abs(alpha)**2)/(2)))
    B = (n/abs(alpha)) -abs(alpha)
    if sigma == 0:
        comp_val = 12
    else:
        comp_val = (float(x)-float(mu))/float(sigma)
    #C = (n/abs(alpha)) *(1/(n-1)) *np.exp(-((abs(alpha)**2)/(2)))
    #D = np.sqrt(math.pi/2) *(1 +erf(abs(alpha)/np.sqrt(2)))
    #N = 1/(sigma*(C+D))
    if comp_val > (-1)*alpha:
        return N * np.exp(-(((x-mu)**2)/(2*sigma**2)))
    if comp_val <= (-1)*alpha:
        return N * A* (B - ((x-mu)/(sigma)))**(-n)

# curve_fit() has problems fitting piecewise defined functions, such as crystal_ball_function().
# Online I found, that one has to vectorize the function in order for curve_fit() to be able to fit it properly.
# Here's the corresponding link I found (accessed 12th March 2019): https://stackoverflow.com/questions/11129812/fitting-piecewise-function-in-python
def function_crystal_ball_one_vec(x, mu, sigma, alpha, n, N):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=function_crystal_ball_one(x[i], mu, sigma, alpha, n, N)
    return y


#Function to define a double Crystal Ball curve utilizing the (single) Crystal Ball Curve defined above
def function_crystal_ball_two(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1):
    return function_crystal_ball_one(x, mu_0, sigma_0, alpha_0, n_0, N_0) + function_crystal_ball_one(x, mu_1, sigma_1, alpha_1, n_1, N_1)


# curve_fit() has problems fitting piecewise defined functions, such as quad_crystal_ball_function().
# Online I found, that one has to vectorize the function in order for curve_fit() to be able to fit it properly.
# Here's the corresponding link I found (accessed 12th March 2019): https://stackoverflow.com/questions/11129812/fitting-piecewise-function-in-python
def function_crystal_ball_two_vec(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=function_crystal_ball_two(x[i], mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1)
    return y


# Function to define a triple Crystal Ball curve utilizing the (single) Crystal Ball Curve defined above.
def function_crystal_ball_three(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2):
    return function_crystal_ball_one(x, mu_0, sigma_0, alpha_0, n_0, N_0) +function_crystal_ball_one(x, mu_1, sigma_1, alpha_1, n_1, N_1) +function_crystal_ball_one(x, mu_2, sigma_2, alpha_2, n_2, N_2)


# curve_fit() has problems fitting piecewise defined functions, such as tri_crystal_ball_function().
# Online I found, that one has to vectorize the function in order for curve_fit() to be able to fit it properly.
# Here's the corresponding link I found (accessed 12th March 2019): https://stackoverflow.com/questions/11129812/fitting-piecewise-function-in-python
def function_crystal_ball_three_vec(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=function_crystal_ball_three(x[i], mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2)
    return y


# Function to define a quadruple Crystal Ball curve utilizing the (single) Crystal Ball Curve defined above.
def function_crystal_ball_four(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3):
    return function_crystal_ball_one(x, mu_0, sigma_0, alpha_0, n_0, N_0) +function_crystal_ball_one(x, mu_1, sigma_1, alpha_1, n_1, N_1) +function_crystal_ball_one(x, mu_2, sigma_2, alpha_2, n_2, N_2) +function_crystal_ball_one(x, mu_3, sigma_3, alpha_3, n_3, N_3)


# curve_fit() has problems fitting piecewise defined functions, such as quad_crystal_ball_function().
# Online I found, that one has to vectorize the function in order for curve_fit() to be able to fit it properly.
# Here's the corresponding link I found (accessed 12th March 2019): https://stackoverflow.com/questions/11129812/fitting-piecewise-function-in-python
def function_crystal_ball_four_vec(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=function_crystal_ball_four(x[i], mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3)
    return y


# Function to define a quadruple Crystal Ball curve utilizing the (single) Crystal Ball Curve defined above.
def function_crystal_ball_five(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3, mu_4, sigma_4, alpha_4, n_4, N_4):
    return function_crystal_ball_one(x, mu_0, sigma_0, alpha_0, n_0, N_0) +function_crystal_ball_one(x, mu_1, sigma_1, alpha_1, n_1, N_1) +function_crystal_ball_one(x, mu_2, sigma_2, alpha_2, n_2, N_2) +function_crystal_ball_one(x, mu_3, sigma_3, alpha_3, n_3, N_3) +function_crystal_ball_one(x, mu_4, sigma_4, alpha_4, n_4, N_4)


# Vectorization of quint_crystal_ball_function()
def function_crystal_ball_five_vec(x, mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3, mu_4, sigma_4, alpha_4, n_4, N_4):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i]=function_crystal_ball_five(x[i], mu_0, sigma_0, alpha_0, n_0, N_0, mu_1, sigma_1, alpha_1, n_1, N_1, mu_2, sigma_2, alpha_2, n_2, N_2, mu_3, sigma_3, alpha_3, n_3, N_3, mu_4, sigma_4, alpha_4, n_4, N_4)
    return y


# This function is used to fit a sum of n Crystal Ball functions to a MonXe histogram.
# The output is then a dictionary containing the determined fit parameters for each peak along with the respective errors.
def fit_range_mult_crystal_ball(
    histogram_data, # a ndarray (with columns '', '' and ''; as generated with XXXX) the Crystal Ball fit is applied to
    n = 2, # number of Crystal Ball peaks (n=2 corresponds to two Crystal Ball peaks)
    fit_range = "", # interval when applying the fit to just an interval of the x data (i.e. bin centers)
    **kwargs # see arguments for scipy.curve_fit (e.g. 'p0' and 'bounds'
):

    ### processing the input
    # restricting the fit to a certain range of bin center values
    if fit_range != "":
        fit_data = histogram_data[(histogram_data["bin_centers"] >= fit_range[0]) & (histogram_data["bin_centers"] <= fit_range[1])]
    else:
        fit_data = histogram_data
    # selecting the corresponding fit function
    if n == 1:
        fit_function = function_crystal_ball_one_vec
    elif n==2:
        fit_function = function_crystal_ball_two_vec
    elif n==3:
        fit_function = function_crystal_ball_three_vec
    elif n==4:
        fit_function = function_crystal_ball_four_vec
    elif n==5:
        fit_function = function_crystal_ball_five_vec
    else:
        print("The current implementation of 'fit_range_mult_crystal_ball' only supports a maximum of five Crystal Ball peaks.")
        return

    ### fitting 'fit_data' with 'n' Crystal Ball functions
    # curve_fit output: 
    p_opt, p_cov = curve_fit(
        f = fit_function,
        xdata = fit_data["bin_centers"],
        ydata = fit_data["counts"],
        sigma = fit_data["counts_errors_upper"],
        absolute_sigma = True,
        method='lm', # "lm" cannot handle covariance matrices with deficient rank
        **kwargs
    )
    # calculating the errors of the fit parameters
    p_err = np.sqrt(np.diag(p_cov))

    ### filling the output dictionary with the fit parameters
    fit_parameter_dictionary = {}
    name_parameter = ["mu", "sigma", "alpha", "n", "N"]
    for i in range(n):
        fit_parameter_dictionary.update({str(i) : {}})
        fit_parameter_dictionary[str(i)].update({"fit_data" : {}})
        fit_parameter_dictionary[str(i)].update({"fit_data_errors" : {}})
        for j in range(5):
            fit_parameter_dictionary[str(i)]["fit_data"].update({name_parameter[j] : p_opt[(i*5)+j]})
            fit_parameter_dictionary[str(i)]["fit_data_errors"].update({name_parameter[j] : p_err[(i*5)+j]})
    return fit_parameter_dictionary



# This function is used to generate a ndarray (with columns 'x' and 'y') which can be used to plot a 2D plot
def get_function_values_for_plotting(
    function, # function which is used to calculate the y values
    x_min, # minimum x value
    x_max, # maximum x value
    n_samples, # number of samples
    **kwargs # keyword arguments which are passed on to the function call (e.g. parameters for Crystal Ball functions)
):
    # defining the ndarray dtype
    gnampf_dtype = np.dtype([
        ("x", np.float64),
        ("y", np.float64)
    ])
    # generating data and saving the ndarray
    tuple_list = [(x, function(x, **kwargs)) for x in np.linspace(start=x_min, stop=x_max, num=n_samples, endpoint=True)]
    data = np.array(tuple_list, gnampf_dtype)
    return data
    

# This function is used to calculate the resolution from the fit parameters for one specific Crystal Ball fit    
def get_resolution(
    single_cb_fit_param_dict,
    single_cb_fit_param_error_dict,
    flag_percent = True # flag determining whether the output is given in percent or in absolute numbers
):
    if flag_percent == True:
        fac = 100
    else:
        fac = 1
    f = 2*np.sqrt(2*np.log(2)) # constant conversion factor for the conversion from a gaussian sigma to the FWHM
    resolution = fac *f *single_cb_fit_param_dict["sigma"]/single_cb_fit_param_dict["mu"]
    resolution_error = fac *np.sqrt( (f *(1/single_cb_fit_param_dict["mu"]) *single_cb_fit_param_error_dict["sigma"])**2  +  (f *single_cb_fit_param_dict["sigma"] *(1/single_cb_fit_param_dict["mu"]**2) *single_cb_fit_param_error_dict["mu"])**2 )
    return resolution, resolution_error


# This function is used to add plottable graph data to the 'fit_parameter_dictionary' generated by 'fit_range_mult_crystal_ball'.
def add_graph_data_to_fpd(fit_parameter_dictionary):
    # looping over the peak numbers and adding graph data by calling 'get_function_values_for_plotting' 
    for key in fit_parameter_dictionary:
        fit_parameter_dictionary[key].update({"graph_data" : get_function_values_for_plotting(function=function_crystal_ball_one, x_min=0, x_max=adc_channel_max, n_samples=4000, **fit_parameter_dictionary[key]["fit_data"])})
    return


# This function is used to automatically add peak specific data to the 'fit_parameter_dictionary' generated by 'fit_range_mult_crystal_ball'.
def calc_peak_data(
    peak_data_dictionary,
    timestamp_data_ndarray,
    n_sigma_left = 5,
    n_sigma_right = 3):

    # energy resolution
    for key in peak_data_dictionary:
        peak_data_dictionary[key].update({"resolution" : {}})
    for key in peak_data_dictionary:
        res, res_err = get_resolution(
            single_cb_fit_param_dict = peak_data_dictionary[key]["fit_data"],
            single_cb_fit_param_error_dict = peak_data_dictionary[key]["fit_data_errors"])
        peak_data_dictionary[key]["resolution"].update({"resolution_in_percent" : res})
        peak_data_dictionary[key]["resolution"].update({"resolution_error" : res_err})

    # counts
    for key in peak_data_dictionary:
        peak_data_dictionary[key].update({"counts" : {}})
    for key in peak_data_dictionary:
        left_border_adc = peak_data_dictionary[key]["fit_data"]["mu"] -n_sigma_left*peak_data_dictionary[key]["fit_data"]["sigma"]
        right_border_adc = peak_data_dictionary[key]["fit_data"]["mu"] +n_sigma_right*peak_data_dictionary[key]["fit_data"]["sigma"]
        peak_data_dictionary[key]["counts"].update({
            "left_border_adc" : left_border_adc,
            "right_border_adc" : right_border_adc,
            "counts" : len(timestamp_data_ndarray[(timestamp_data_ndarray["pulse_height_adc"]>=left_border_adc) & (timestamp_data_ndarray["pulse_height_adc"]<=right_border_adc)])
        })

    # end of function
    return


# This function is used to extract the 'peak_data_dict' from 'raw_data' and 'measurement_data'.
def get_spectrum_data_output_dict(
    measurement_data_dict,
    raw_data_ndarray,
):

    ### profile: 'crystal_ball_fit_with_linear_energy_channel_relation'
    if measurement_data_dict["spectrum_data_input"]["flag_spectrum_data"] == "crystal_ball_fit_with_linear_energy_channel_relation":

        # generating rebinned histogram data
        data_histogram_rebinned = get_histogram_data_from_timestamp_data(
            timestamp_data = raw_data_ndarray,
            number_of_bins = int(n_adc_channels/measurement_data_dict["spectrum_data_input"]["rebin"]))

        # fitting the histogram with Crystal Ball functions
        crystal_ball_peak_fit_data_dict = fit_range_mult_crystal_ball(
            n = int(len(measurement_data_dict["spectrum_data_input"]["p_opt_guess"])/5),
            histogram_data = data_histogram_rebinned,
            fit_range = measurement_data_dict["spectrum_data_input"]["fit_range"],
            p0 = measurement_data_dict["spectrum_data_input"]["p_opt_guess"])

        # fitting a the energy-channel relation with a linear function
        adcc_list = []
        adcc_error_list = []
        e_mev_list = []
        for peaknum in measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]:
            adcc_list.append(crystal_ball_peak_fit_data_dict[peaknum]["fit_data"]["mu"])
            adcc_error_list.append(crystal_ball_peak_fit_data_dict[peaknum]["fit_data_errors"]["mu"])
            e_mev_list.append(isotope_dict[measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][peaknum]]["alpha_energy_kev"]/1000)
        p_opt, p_cov = curve_fit(
            f = function_linear_vec,
            xdata = adcc_list,
            ydata = e_mev_list,
            #sigma = fit_data["counts_errors"],
            #absolute_sigma = True,
            method = 'lm' # "lm" cannot handle covariance matrices with deficient rank
        )
        p_err = np.sqrt(np.diag(p_cov))

        # peak data calculations (i.e. determine resolution, number of counts, etc...)
        #calc_peak_data(
        #    peak_data_dictionary = peak_data_dict,
        #    timestamp_data_ndarray = data_raw,
        #    n_sigma_left = 10,
        #    n_sigma_right = 4)

        # filling the 'spectrum_data_output_dict'
        spectrum_data_output_dict = {
            "histogram" : {
                "bin_centers_adc" : [str(entry) for entry in data_histogram_rebinned["bin_centers"]],
                "counts" : [str(entry) for entry in data_histogram_rebinned["counts"]],
                "counts_errors_lower" : [str(entry) for entry in data_histogram_rebinned["counts_errors_lower"]],
                "counts_errors_upper" : [str(entry) for entry in data_histogram_rebinned["counts_errors_upper"]],
            },
            "peak_fits" : crystal_ball_peak_fit_data_dict,
            "energy_channel_relation_fit" : {
                "fit_data": {
                    "m" : p_opt[0],
                    "t" : p_opt[1]},
                "fit_data_errors" : {
                    "m" : p_err[0],
                    "t" : p_err[1]},
            },
        }

    return spectrum_data_output_dict


# This function is used to plot a monxe spectrum.
# So far only works with measurement_data_dict['spectrum_data_input']['flag_spectrum_data'] == "crystal_ball_fit_with_linear_energy_channel_relation"
def plot_mca_spectrum(
    raw_data_ndarray,
    measurement_data_dict = {},
    # plot stuff
    plot_aspect_ratio = 9/16,
    plot_figsize_x_inch = miscfig.standard_figsize_x_inch,
    plot_xlabel = ["", "jfk"][0],
    plot_ylabel = ["", "jfk"][0], # y-axis label, if "" then automatically derived from binwidth and axis scale
    plot_x_lim = [0, 1, "rel"],
    plot_y_lim = [0.0, 1.1, "rel"],
    plot_linewidth = 0.5,
    plot_linecolor = "black",
    plot_labelfontsize = 11,
    plot_annotate_comments_dict = {},
    # flags
    flag_comments = [],
    flag_x_axis_units = ["adc_channels", "mev"][0],
    flag_plot_histogram_data_from = ["raw_data_ndarray", "spectrum_data"][0],
    flag_setylogscale = False,
    flag_output_abspath_list = [False, ["~/jfk.png"]][0],
    flag_show = True,
    flag_errors = [False, "poissonian"][0],
    flag_plot_fits = [], # list of fits to be plotted, given in peak numbers
    flag_preliminary = [False, True][0],
    flag_show_peak_labels = [False, True][0],
    flag_show_isotope_windows = [False,True][0],
):

    # canvas
    fig, ax1 = plt.subplots(figsize=[plot_figsize_x_inch,plot_figsize_x_inch*plot_aspect_ratio], dpi=150, constrained_layout=True)

    # retrieving the histogram data to be plotted
    data_histogram = get_histogram_data_from_timestamp_data(timestamp_data=raw_data_ndarray, histval="pulse_height_adc")
    x_data = data_histogram["bin_centers"]
    y_data = data_histogram["counts"]
    if flag_plot_histogram_data_from == "spectrum_data":
        x_data_adc = [float(entry) for entry in measurement_data_dict["spectrum_data_output"]["histogram"]["bin_centers_adc"]]
        x_data = [float(entry) for entry in measurement_data_dict["spectrum_data_output"]["histogram"]["bin_centers_adc"]]
        y_data = [float(entry) for entry in measurement_data_dict["spectrum_data_output"]["histogram"]["counts"]]
    binwidth = x_data[2] -x_data[1]
    ylabel = r"entries per " +f"{binwidth:.1f}" +r" adc channels"
    xlabel = r"alpha energy / adc channels"
    if flag_x_axis_units == "mev":
        x_data = function_linear_vec(x=x_data, m=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["m"], t=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["t"])
        binwidth = float(x_data[2]) -float(x_data[1])
        ylabel = r"entries per " +f"{binwidth*1000:.2f}" +r" $\mathrm{keV}$"
        xlabel = r"alpha energy / $\mathrm{MeV}$"

    # axes
    if plot_x_lim[2] == "rel":
        xlim = [x_data[0] +plot_x_lim[0]*(x_data[-1]-x_data[0]), x_data[0] +plot_x_lim[1]*(x_data[-1]-x_data[0])]
    elif plot_x_lim[2] == "abs":
        xlim = [plot_x_lim[0], plot_x_lim[1]]
    if plot_y_lim[2] == "rel":
        ylim = [min(y_data) +plot_y_lim[0]*(max(y_data)-min(y_data)), min(y_data) +plot_y_lim[1]*(max(y_data)-min(y_data))]
    elif plot_y_lim[2] == "abs":
        ylim = [plot_y_lim[0], plot_y_lim[1]]
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    xlabel = plot_xlabel if plot_xlabel != "" else xlabel
    ylabel = plot_ylabel if plot_ylabel != "" else ylabel
    ax1.set_xlabel(xlabel, fontsize=plot_labelfontsize)
    ax1.set_ylabel(ylabel, fontsize=plot_labelfontsize)
    if flag_setylogscale:
        ax1.set_yscale('log')

    # plotting the stepized histogram
    bin_centers, counts, counts_errors_lower, counts_errors_upper, bin_centers_mod, counts_mod = stepize_histogram_data(
        bincenters = x_data,
        counts = y_data,
        flag_addfirstandlaststep = True)
    plt.plot(
        bin_centers_mod,
        counts_mod,
        linewidth = plot_linewidth,
        color = plot_linecolor,
        linestyle = '-',
        zorder = 1,
        label = "jfk")

    # plotting the fits
    if flag_plot_fits != []:
        #fit_x_data = np.linspace(start=xlim[0], stop=xlim[1], endpoint=True, num=500)
        for peaknum in [str(peaknum) for peaknum in flag_plot_fits]:
            plt.plot(
                x_data,
                [function_crystal_ball_one(x_data_adc_val, **measurement_data_dict["spectrum_data_output"]["peak_fits"][peaknum]["fit_data"]) for x_data_adc_val in x_data_adc],
                linewidth = 1.2,
                color = isotope_dict[measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][peaknum]]["color"] if peaknum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] else "red",
                linestyle = '-',
                zorder = 1,
                label = "fit") #r"$" +peak_data_dict[key]["isotope_data"]["latex_label"] +r"$")

    # plotting the Poissonian errors
#    if flag_plot_errors:
#        plt.fill_between(
#            bin_centers if flag_x_axis_units=="adc_channels" else [energy_channel_relation(xi, *p_opt) for xi in bin_centers],
#            counts-counts_errors_lower,
#            counts+counts_errors_upper,
#            color = color_histogram_error,
#            alpha = 1,
#            zorder = 0,
#            interpolate = True)

    ### annotations
#    all_peaks = [*measurement_data_dict["spectrum_data_output"]]
#    known_peaks = [peaknum for peaknum in all_peaks if "a_priori" in [*measurement_data_dict["spectrum_data_input"][peaknum]]]
#    # annotationg the MonXe logo
#    #miscfig.image_onto_plot(
#    #    filestring = "monxe_logo__transparent_bkg.png",
#    #    ax=ax1,
#    #    position=(x_lim[0]+0.90*(x_lim[1]-x_lim[0]),y_lim[0]+0.87*(y_lim[1]-y_lim[0])),
#    #    pathstring = pathstring_miscellaneous_figures +"monxe_logo/",
#    #    zoom=0.02,
#    #    zorder=0)
#    # peak labels
    if flag_show_peak_labels == True:
        for peaknum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]]:
            plt.text(
                x = measurement_data_dict["spectrum_data_output"]["peak_fits"][peaknum]["fit_data"]["mu"] if flag_x_axis_units=="adc_channels" else function_linear_vec(x=measurement_data_dict["spectrum_data_output"]["peak_fits"][peaknum]["fit_data"]["mu"], **measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]),
                y = 1.02*function_crystal_ball_one(x=measurement_data_dict["spectrum_data_output"]["peak_fits"][peaknum]["fit_data"]["mu"], **measurement_data_dict["spectrum_data_output"]["peak_fits"][peaknum]["fit_data"]),
                #transform = ax1.transAxes,
                s = r"$" +isotope_dict[measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][peaknum]]["latex_label"] +r"$",
                color = isotope_dict[measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][peaknum]]["color"],
                fontsize = 11,
                verticalalignment = 'bottom',
                horizontalalignment = 'right')
#    # shading the fit region
#    #ax1.axvspan(
#    #    fitrange[0] if flag_x_axis_units=="adc_channels" else energy_channel_relation(fitrange[0], *p_opt),
#    #    fitrange[1] if flag_x_axis_units=="adc_channels" else energy_channel_relation(fitrange[1], *p_opt),
#    #    alpha = 0.5,
#    #    linewidth = 0,
#    #    color = 'grey',
#    #    zorder = -50)
#    # shading the isotope windows (i.e., the extracted counts)
    if "activity_data_output" in [*measurement_data_dict] and flag_show_isotope_windows == True:
        for peak in ["po218", "po214"]:
            isotope_window_adcc = measurement_data_dict["activity_data_output"]["event_extraction"][peak +"_adcc_window"]
            ax1.axvspan(
                isotope_window_adcc[0] if flag_x_axis_units == "adc_channels" else function_linear_vec(x=[isotope_window_adcc[0]], m=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["m"], t=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["t"])[0],
                isotope_window_adcc[1] if flag_x_axis_units == "adc_channels" else function_linear_vec(x=[isotope_window_adcc[1]], m=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["m"], t=measurement_data_dict["spectrum_data_output"]["energy_channel_relation_fit"]["fit_data"]["t"])[0],
                alpha = 0.3,
                linewidth = 0,
                color = isotope_dict[peak]["color"],
                zorder = -50)

    # annotating comments
    if flag_comments != []:
        comment_list = []
        if "delta_t_ema" in flag_comments:
            comment_list.append(r"$\Delta t_{\mathrm{ema}}=" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['delta_t_ema_s']/(60*60*24):.1f}" +r"\,\mathrm{d}$")
        if "delta_t_trans" in flag_comments:
            comment_list.append(r"$\Delta t_{\mathrm{trans}}=" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['delta_t_trans_s']/(60*60):.1f}" +r"\,\mathrm{h}$")
        if "delta_t_meas" in flag_comments:
            comment_list.append(r"$\Delta t_{\mathrm{meas}}=" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['delta_t_meas_s']/(60*60*24):.1f}" +r"\,\mathrm{d}$")
        if "delta_t_meas_eff" in flag_comments and "activity_data_input" in [*measurement_data_dict]:
            if measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] == "t_meas_f":
                delta_t_meas_eff = measurement_data_dict['activity_data_output']['activity_extrapolation']['delta_t_meas_s']
            elif type(measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1]) in [int,float]:
                delta_t_meas_eff = measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] -measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][0]
            comment_list.append(r"$\Delta t_{\mathrm{meas}}=" +f"{delta_t_meas_eff/(60*60*24):.1f}" +r"\,\mathrm{d}$")
        if "n_meas" in flag_comments:
            comment_list.append(r"$n^{\mathrm{meas}}_{^{218}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous']['n_meas_po218']:.1f}" +r",\,n^{\mathrm{meas}}_{^{214}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous'][n_meas_po214]:.1f}" +r"$")            
        if "n_bkg_expected" in flag_comments:
            comment_list.append(r"$\bar{n}^{\mathrm{bkg}}_{^{218}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous']['n_bkg_expected_po218']:.1f}" +r",\,\bar{n}^{\mathrm{bkg}}_{^{214}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous']['n_bkg_expected_po214']:.1f}" +r"$")            
        if "n_sig" in flag_comments:
            comment_list.append(r"$\bar{n}^{\mathrm{sig}}_{^{218}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous']['n_sig_po218']:.1f}" +r",\,\bar{n}^{\mathrm{sig}}_{^{214}\mathrm{Po}}=" +f"{measurement_data_dict['activity_data_output']['miscellaneous']['n_sig_po214']:.1f}" +r"$")            
        if "detection_efficiency" in flag_comments:
            comment_list.append(r"$\varepsilon^{\mathrm{det}}=(" +f"{measurement_data_dict['activity_data_input']['detection_efficiency_mean']*100:.2f}" +r"_{-" +f"{measurement_data_dict['activity_data_input']['detection_efficiency_loweruncertainty']*100:.2f}" +r"}^{+" +f"{measurement_data_dict['activity_data_input']['detection_efficiency_upperuncertainty']*100:.2f}" +r"})\,\mathrm{mBq}$")            
        if "resolution_po214" in flag_comments:
            resolution_po214 = 2*np.sqrt(2*np.log(2)) *measurement_data_dict['spectrum_data_output']['peak_fits']["1"]["fit_data"]["sigma"]/measurement_data_dict['spectrum_data_output']['peak_fits']["1"]["fit_data"]["mu"]
            comment_list.append(r"$\mathrm{res}^{\mathrm{FWHM}}_{^{214}\mathrm{Po}}=" +f"{100*resolution_po214:.1f}" +r"\,\%$")
        if "bc_rema_result" in flag_comments and "activity_data_output" in [*measurement_data_dict]:
            comment_list.append(r"${^{\mathrm{BC}}R}^{\mathrm{ema}}_{^{222}\mathrm{Rn}}=(" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_mean_bq']*1000:.1f}" +r"_{-" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_loweruncertainty_bq']*1000:.1f}" +r"}^{+" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_upperuncertainty_bq']*1000:.1f}" +r"})\,\mathrm{mBq}$")
        if "amf_rema_result" in flag_comments and "activity_data_output" in [*measurement_data_dict]:
            comment_list.append(r"${^{\mathrm{AMF}}R}^{\mathrm{ema}}_{^{222}\mathrm{Rn}}=(" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_mean_bq']*1000:.1f}" +r"_{-" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_loweruncertainty_bq']*1000:.1f}" +r"}^{+" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_upperuncertainty_bq']*1000:.1f}" +r"})\,\mathrm{mBq}$")
        annotate_comments(
            comment_ax = ax1,
            comment_list = comment_list,
            **plot_annotate_comments_dict,
        )

#    # measurement comments
#    #monxeana.annotate_comments(
#    #    comment_ax = ax1,
#    #    comment_list = [
#    #        r"calibration activity: $(19.6\pm 2.1)\,\mathrm{mBq}$",
#    #        r"entries total: " +f"{len(data_raw)}",
#    #        r"measurement duration: " +f"{monxeana.get_measurement_duration(list_file_data=data_raw, flag_unit='days'):.3f} days",
#    #    ],
#    #    comment_textpos = [0.025, 0.9],
#    #    comment_textcolor = "black",
#    #    comment_linesep = 0.1,
#    #    comment_fontsize = 11)
#    # annotating 
#    #plt.text(
#    #    x = peak_data_dict["2"]["fit_data"]["mu"] -0.02*x_width,
#    #    y = 1.02*monxeana.function_crystal_ball_one(x=peak_data_dict["2"]["fit_data"]["mu"], **peak_data_dict["2"]["fit_data"]) -0.06*y_width,
#    #    #transform = ax1.transAxes,
#    #    s = r"$R=" +f"{peak_data_dict['2']['resolution']['resolution_in_percent']:.1f}" +r"\,\%$",
#    #    color = "black",
#    #    fontsize = 11,
#    #    verticalalignment = 'center',
#    #    horizontalalignment = 'right')
    # marking as 'preliminary'
    if flag_preliminary == True:
        plt.text(
            x = 0.97,
            y = 0.95,
            transform = ax1.transAxes,
            s = "preliminary",
            color = "red",
            fontsize = 13,
            verticalalignment = 'center',
            horizontalalignment = 'right')
    # legend
    #plt.legend()

    # saving
    if flag_show:
        plt.show()
    if flag_output_abspath_list != []:
        for output_abspath in flag_output_abspath_list:
            fig.savefig(output_abspath)
    return





#######################################
### Activity Data
#######################################


# This function is corresponding to the exponential rise of the 218Po and 214Po activities.
def exp_rise(t, a, lambd):
    return a*(1-np.exp(-lambd*t))


# This function is fitted with curve_fit() in order to extract a better measurement of the initial radon activity.
def integral_function_po218(
        t, # time at which the integral of the integrand_function over 'integration_interval_width' is evaluated
        n222rn0,
        n218po0,
        n214pb0,
        n214bi0,
        r):

    # case a: array-input (e.g. if function is fitted with curve_fit() )
    if hasattr(t, '__len__'):
        return [integrate.quad(
                func = po218,
                a = ts-0.5*activity_interval_h*60*60,
                b = ts+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, n214pb0, n214bi0, r)
            )[0] for ts in t]

    # case b: scalar-like (e.g. for explicit calculations)
    else:
        return integrate.quad(
                func = po218,
                a = t-0.5*activity_interval_h*60*60,
                b = t+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, n214pb0, n214bi0, r)
            )[0]


# This function is fitted with curve_fit() in order to extract a better measurement of the initial radon activity.
# It utilizes the 'bi214()' function as the 'po214()' function is not existing due to computational limitations.
def integral_function_po214(
        t, # time at which the integral of the integrand_function over 'integration_interval_width' is evaluated
        n222rn0,
        n218po0,
        n214pb0,
        n214bi0,
        r):

    # case a: array-input (e.g. if function is fitted with curve_fit() )
    if hasattr(t, '__len__'):
        return [integrate.quad(
                func = bi214,
                a = ts-0.5*activity_interval_h*60*60,
                b = ts+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, n214pb0, n214bi0, r)
            )[0] for ts in t]

    # case b: scalar-like (e.g. for explicit calculations)
    else:
        return integrate.quad(
                func = bi214,
                a = t-0.5*activity_interval_h*60*60,
                b = t+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, n214pb0, n214bi0, r)
            )[0]


# This function is used to annotate multiline comments onto a plot.
def annotate_comments(
    comment_ax,
    comment_list,
    comment_textpos = [0.025, 0.9],
    comment_textcolor = "black",
    comment_linesep = 0.1,
    comment_fontsize = 11,
    flag_alignment = ["top_to_bottom", "symmetric"]):

    ctr_textpos = 0
    for i in range(len(comment_list)):
        plt.text(
            x = comment_textpos[0],
            y = comment_textpos[1] -ctr_textpos*comment_linesep,
            s = comment_list[i],
            fontsize = comment_fontsize,
            color = comment_textcolor if type(comment_textcolor)==str else comment_textcolor[i],
            rotation = 0,
            horizontalalignment = "left" if comment_textpos[0] < 0.5 else "right",
            verticalalignment = "top",
            transform = comment_ax.transAxes
        )
        ctr_textpos += 1

    return


# This function is used to extract the 'activity_data_output_dict' from 'raw_data' and 'measurement_data'.
def get_activity_data_output_dict(
    measurement_data_dict,
    raw_data_ndarray,
):

    ### profile: 'polonium_activity_fit'
    if measurement_data_dict["activity_data_input"]["flag_activity_data"] == "activity_model_fit":

        # determining the measurement time window to be considered for the activity model fit
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the measurement time window")
        if measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] == "t_meas_f":
            t_max_ps = max(raw_data_ndarray["timestamp_ps"])
        elif type(measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1]) in [int,float]:
            t_max_ps = measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] *(10**12)
        if measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][0] == 0:
            t_min_ps = 0
        elif type(measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][0]) in [int,float]:
            t_min_ps = measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][0] *(10**12)

        # calculating the time edges during which the detected events are counted
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): calculating the time edges during which the detected events are counted")
        timestamp_edges_ps = [t_min_ps]
        timestamp_ctr = 1
        while timestamp_edges_ps[len(timestamp_edges_ps)-1] +measurement_data_dict["activity_data_input"]["activity_interval_ps"] < t_max_ps:
            timestamp_edges_ps.append(measurement_data_dict["activity_data_input"]["activity_interval_ps"]*timestamp_ctr +t_min_ps)
            timestamp_ctr += 1
        timestamp_centers_ps = [i +0.5*measurement_data_dict["activity_data_input"]["activity_interval_ps"] for i in timestamp_edges_ps[:-1]]
        timestamp_centers_seconds = [i/(1000**4) for i in timestamp_centers_ps]

        # extracting the detected counts per time bin
        decays_per_activity_interval_po218 = []
        decays_per_activity_interval_po214 = []
        decays_per_activity_interval_po218_errors_lower = []
        decays_per_activity_interval_po214_errors_lower = []
        decays_per_activity_interval_po218_errors_upper = []
        decays_per_activity_interval_po214_errors_upper = []

        # determining the adcc selection windows for the individual peaks
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the adcc selection windows")
        if measurement_data_dict["activity_data_input"]["flag_calibration"] not in ["self_absolute_adcc", "self_relative_adcc", "self_relative_sigma"]:
            print(f"include calibration by external file here")
        else:
            po218_peaknum = [keynum for keynum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] if measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][keynum]=="po218"][0]
            po214_peaknum = [keynum for keynum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] if measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][keynum]=="po214"][0]
            if measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_absolute_adcc":
                adcc_selection_window_po218_left = measurement_data_dict["activity_data_input"]["po218_window"][0]
                adcc_selection_window_po218_right = measurement_data_dict["activity_data_input"]["po218_window"][1]
                adcc_selection_window_po214_left = measurement_data_dict["activity_data_input"]["po214_window"][0]
                adcc_selection_window_po214_right = measurement_data_dict["activity_data_input"]["po214_window"][1]
            elif measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_relative_adcc":
                adcc_selection_window_po218_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] -measurement_data_dict["activity_data_input"]["po218_window"][0]
                adcc_selection_window_po218_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] +measurement_data_dict["activity_data_input"]["po218_window"][1]
                adcc_selection_window_po214_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] -measurement_data_dict["activity_data_input"]["po214_window"][0]
                adcc_selection_window_po214_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] +measurement_data_dict["activity_data_input"]["po214_window"][1]
            elif measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_relative_sigma":
                adcc_selection_window_po218_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] -(measurement_data_dict["activity_data_input"]["po218_window"][0] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po218_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] +(measurement_data_dict["activity_data_input"]["po218_window"][1] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po214_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] -(measurement_data_dict["activity_data_input"]["po214_window"][0] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po214_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] +(measurement_data_dict["activity_data_input"]["po214_window"][1] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["sigma"])

        # determining the detected po218 and po214 decays per 'activity_interval_ps'
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the detected po218 and po214 decays per 'activity_interval_ps'")
        for i in range(len(timestamp_edges_ps)-1):
            decays_per_activity_interval_po218.append(len(raw_data_ndarray[
                (raw_data_ndarray["timestamp_ps"] >= timestamp_edges_ps[i]) &
                (raw_data_ndarray["timestamp_ps"] <= timestamp_edges_ps[i+1]) &
                (raw_data_ndarray["pulse_height_adc"] >= adcc_selection_window_po218_left) &
                (raw_data_ndarray["pulse_height_adc"] <= adcc_selection_window_po218_right)]))
            decays_per_activity_interval_po214.append(len(raw_data_ndarray[
                (raw_data_ndarray["timestamp_ps"] >= timestamp_edges_ps[i]) &
                (raw_data_ndarray["timestamp_ps"] <= timestamp_edges_ps[i+1]) &
                (raw_data_ndarray["pulse_height_adc"] >= adcc_selection_window_po214_left) &
                (raw_data_ndarray["pulse_height_adc"] <= adcc_selection_window_po214_right)]))
        for i in range(len(decays_per_activity_interval_po218)):
            decays_per_activity_interval_po218_errors_lower.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po218[i], flag_mode=measurement_data_dict["activity_data_input"]["flag_errors"])[0])
            decays_per_activity_interval_po218_errors_upper.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po218[i], flag_mode=measurement_data_dict["activity_data_input"]["flag_errors"])[1])
            decays_per_activity_interval_po214_errors_lower.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po214[i], flag_mode=measurement_data_dict["activity_data_input"]["flag_errors"])[0])
            decays_per_activity_interval_po214_errors_upper.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po214[i], flag_mode=measurement_data_dict["activity_data_input"]["flag_errors"])[1])
            decays_per_activity_interval_po218_mean_errors = [0.5*(decays_per_activity_interval_po218_errors_lower[i] +decays_per_activity_interval_po218_errors_upper[i]) for i in range(len(decays_per_activity_interval_po218_errors_lower))]
            decays_per_activity_interval_po214_mean_errors = [0.5*(decays_per_activity_interval_po214_errors_lower[i] +decays_per_activity_interval_po214_errors_upper[i]) for i in range(len(decays_per_activity_interval_po214_errors_lower))]

        # activity model fit
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): activity model fit")
        if measurement_data_dict["activity_data_input"]["flag_model_fit"] == "fit_po218_and_po214_independently":
            print(f"This still needs to be implemented")
        elif measurement_data_dict["activity_data_input"]["flag_model_fit"] == "fit_po218_and_po214_simultaneously":
            # defining a function that allows to simultaneously fit both the po218 and po214 activities
            def constrained_fit_function(double_bin_centers, n222rn0, n218po0, n214pb0, n214bi0, r):
                l = len(double_bin_centers)
                result_1 = integral_function_po218(double_bin_centers[:l], n222rn0, n218po0, n214pb0, n214bi0, r)
                result_2 = integral_function_po214(double_bin_centers[l:], n222rn0, n218po0, n214pb0, n214bi0, r)
                return result_1 +result_2
            # fitting the measured activities
            p_opt, p_cov = curve_fit(
                f =  constrained_fit_function,
                xdata = timestamp_centers_seconds +timestamp_centers_seconds,
                ydata = decays_per_activity_interval_po218 +decays_per_activity_interval_po214,
                sigma = decays_per_activity_interval_po218_mean_errors +decays_per_activity_interval_po214_mean_errors, # NOTE: at some point one would probably like to implement a fit function (instead of curve_fit) that also respects asymmetric errors, so far I am using the artihmetic mean of both errors
                absolute_sigma = True,
                bounds = measurement_data_dict["activity_data_input"]["p_opt_bounds"],
                p0 = measurement_data_dict["activity_data_input"]["p_opt_guess"])
                #method = 'lm', # "lm" cannot handle covariance matrices with deficient rank
            p_err = np.sqrt(np.diag(p_cov))
            po218_n222rn0 = p_opt[0]
            po218_n218po0 = p_opt[1]
            po218_n214pb0 = p_opt[2]
            po218_n214bi0 = p_opt[3]
            po218_r = p_opt[4]
            po218_n222rn0_error = p_err[0]
            po218_n218po0_error = p_err[1]
            po218_n214pb0_error = p_err[2]
            po218_n214bi0_error = p_err[3]
            po218_r_error = p_err[4]
            po214_n222rn0 = p_opt[0]
            po214_n218po0 = p_opt[1]
            po214_n214pb0 = p_opt[2]
            po214_n214bi0 = p_opt[3]
            po214_r = p_opt[4]
            po214_n222rn0_error = p_err[0]
            po214_n218po0_error = p_err[1]
            po214_n214pb0_error = p_err[2]
            po214_n214bi0_error = p_err[3]
            po214_r_error = p_err[4]

        # extrapolating the equilibrium emanation activity
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): extrapolating the equilibrium emanation activity")
        if measurement_data_dict["activity_data_input"]["flag_activity_extrapolation"] == False:
            dt_trans = "NA"
            dt_ema = "NA"
            a_ema = "NA"
            #a_ema_error ="NA" 
        elif measurement_data_dict["activity_data_input"]["flag_activity_extrapolation"] in ["default"]:
            # retrieving the relevant times from the ELOG timestamps
            t_ema_i = measurement_data_dict[key_measurement_information]["t_ema_i"]
            t_trans_i = measurement_data_dict[key_measurement_information]["t_trans_i"]
            t_meas_i = measurement_data_dict[key_measurement_information]["t_meas_i"]
            t_meas_f = measurement_data_dict[key_measurement_information]["t_meas_f"]
            # calculating the relevant time deltas
            dt_meas = (timestamp_conversion(t_meas_f)-timestamp_conversion(t_meas_i)).total_seconds()
            dt_ema = (timestamp_conversion(t_trans_i)-timestamp_conversion(t_ema_i)).total_seconds()
            dt_trans = (timestamp_conversion(t_meas_i)-timestamp_conversion(t_trans_i)).total_seconds()
            # calculating the activities
            a_t_meas_i = po214_n222rn0 *isotope_dict["rn222"]["decay_constant"]
            a_t_meas_i_uncertainty = po214_n222rn0_error *isotope_dict["rn222"]["decay_constant"]

            # extrapolating the radon activity from 't_meas_i' to 't_trans_i'
            a_t_trans_i_mean, a_t_trans_i_loweruncertainty, a_t_trans_i_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = extrapolate_radon_activity,
                function_input_dict = {
                    "known_activity_at_dt_known_bq_mean" : a_t_meas_i,
                    "known_activity_at_dt_known_bq_loweruncertainty" : a_t_meas_i_uncertainty,
                    "known_activity_at_dt_known_bq_upperuncertainty" : a_t_meas_i_uncertainty,
                },
                function_parameter_dict = {
                    "flag_exp_rise_or_decay" : ["rise", "decay"][1],
                    "lambda_222rn" : isotope_dict["rn222"]["decay_constant"],
                    "known_dt_s" : dt_trans,
                    "dt_extrapolation_s" : 0,
                },
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

            # extrapolating the radon activity from 't_trans_i' to 'inf'
            a_ema_withoutdeteffcorr_mean, a_ema_withoutdeteffcorr_loweruncertainty, a_ema_withoutdeteffcorr_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = extrapolate_radon_activity,
                function_input_dict = {
                    "known_activity_at_dt_known_bq_mean" : a_t_trans_i_mean,
                    "known_activity_at_dt_known_bq_loweruncertainty" : a_t_trans_i_loweruncertainty,
                    "known_activity_at_dt_known_bq_upperuncertainty" : a_t_trans_i_upperuncertainty,
                },
                function_parameter_dict = {
                    "flag_exp_rise_or_decay" : ["rise", "decay"][0],
                    "lambda_222rn" : isotope_dict["rn222"]["decay_constant"],
                    "known_dt_s" : dt_ema,
                    "dt_extrapolation_s" : "inf",
                },
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

        # detection efficiency correction
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): detection efficiency correction")
        if measurement_data_dict["activity_data_input"]["detection_efficiency_mean"] != 1:
            r_ema_mean, r_ema_loweruncertainty, r_ema_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = detection_efficiency_correction,
                function_input_dict = {
                    "r_ema_mean" : a_ema_withoutdeteffcorr_mean,
                    "r_ema_loweruncertainty" : a_ema_withoutdeteffcorr_loweruncertainty,
                    "r_ema_upperuncertainty" : a_ema_withoutdeteffcorr_upperuncertainty,
                    "detection_efficiency_mean" : measurement_data_dict["activity_data_input"]["detection_efficiency_mean"],
                    "detection_efficiency_loweruncertainty" : measurement_data_dict["activity_data_input"]["detection_efficiency_loweruncertainty"],
                    "detection_efficiency_upperuncertainty" : measurement_data_dict["activity_data_input"]["detection_efficiency_upperuncertainty"],
                },
                function_parameter_dict = {},
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])
        else:
            r_ema_mean = a_ema_withoutdeteffcorr_mean
            r_ema_loweruncertainty = a_ema_withoutdeteffcorr_loweruncertainty
            r_ema_upperuncertainty = a_ema_withoutdeteffcorr_upperuncertainty


        # chi square test
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): chi square test")
        decays_per_activity_interval_po218_expected = integral_function_po218(timestamp_centers_seconds, po218_n222rn0, po218_n218po0, po218_n214pb0, po218_n214bi0, po218_r)
        decays_per_activity_interval_po214_expected = integral_function_po214(timestamp_centers_seconds, po214_n222rn0, po214_n218po0, po214_n214pb0, po214_n214bi0, po214_r)
        if measurement_data_dict["activity_data_input"]["flag_calculate_chi_square"] == True:
            chi_square, chi_square_p_value = stats.chisquare(
                f_obs = decays_per_activity_interval_po218 +decays_per_activity_interval_po214,
                f_exp = decays_per_activity_interval_po218_expected +decays_per_activity_interval_po214_expected,
                ddof = 1)
            chi_square_reduced = calc_reduced_chi_square(
                y_data_obs = decays_per_activity_interval_po218 +decays_per_activity_interval_po214,
                y_data_exp = decays_per_activity_interval_po218_expected +decays_per_activity_interval_po214_expected,
                y_data_err = [],
                ddof = 1,
                y_data_err_lower = decays_per_activity_interval_po218_errors_lower +decays_per_activity_interval_po214_errors_lower,
                y_data_err_upper = decays_per_activity_interval_po218_errors_upper +decays_per_activity_interval_po214_errors_upper,
            )


        # writing and returning the 'activity_data_dict'
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): writing the 'activity_data_dict'")
        activity_data_output_dict = {
            "event_extraction" : {
                "po218_adcc_window" : [adcc_selection_window_po218_left, adcc_selection_window_po218_right],
                "po214_adcc_window" : [adcc_selection_window_po214_left, adcc_selection_window_po214_right],
            },
            "activity_model_fit_results" : {
                "chi_square" : chi_square,
                "chi_square_p_value" : chi_square_p_value,
                "reduced_chi_square" : chi_square_reduced,
                "po218" : {
                    "timestamp_centers_seconds" : timestamp_centers_seconds,
                    "decays_per_activity_interval" : decays_per_activity_interval_po218,
                    "decays_per_activity_interval_errors_lower" : decays_per_activity_interval_po218_errors_lower,
                    "decays_per_activity_interval_errors_upper" : decays_per_activity_interval_po218_errors_upper,
                    "decays_per_activity_interval_expected" : decays_per_activity_interval_po218_expected,
                    "n222rn0" : po218_n222rn0,
                    "n218po0" : po218_n218po0,
                    "n214pb0" : po218_n214pb0,
                    "n214bi0" : po218_n214bi0,
                    "r" : po218_r,
                    "n222rn0_error" : po218_n222rn0_error,
                    "n218po0_error" : po218_n218po0_error,
                    "n214pb0_error" : po218_n214pb0_error,
                    "n214bi0_error" : po218_n214bi0_error,
                    "r_error" : po218_r_error
                },
                "po214" : {
                    "timestamp_centers_seconds" : timestamp_centers_seconds,
                    "decays_per_activity_interval" : decays_per_activity_interval_po214,
                    "decays_per_activity_interval_errors_lower" : decays_per_activity_interval_po214_errors_lower,
                    "decays_per_activity_interval_errors_upper" : decays_per_activity_interval_po214_errors_upper,
                    "decays_per_activity_interval_expected" : decays_per_activity_interval_po214_expected,
                    "n222rn0" : po214_n222rn0,
                    "n218po0" : po214_n218po0,
                    "n214pb0" : po214_n214pb0,
                    "n214bi0" : po214_n214bi0,
                    "r" : po214_r,
                    "n222rn0_error" : po214_n222rn0_error,
                    "n218po0_error" : po214_n218po0_error,
                    "n214pb0_error" : po214_n214pb0_error,
                    "n214bi0_error" : po214_n214bi0_error,
                    "r_error" : po214_r_error
                }
            },
            "activity_extrapolation" : {
                "delta_t_trans_s" : dt_trans,
                "delta_t_ema_s" : dt_ema,
                "delta_t_meas_s" : dt_meas,
                "r_ema_mean_bq" : r_ema_mean,
                "r_ema_loweruncertainty_bq" :  r_ema_loweruncertainty,
                "r_ema_upperuncertainty_bq" :  r_ema_upperuncertainty,
            },
        }

    ### profile: 'polonium_activity_fit'
    elif measurement_data_dict["activity_data_input"]["flag_activity_data"] == "box_counting":

        # determining the adcc selection windows for the individual peaks
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the adcc selection windows for the individual peaks")
        if measurement_data_dict["activity_data_input"]["flag_calibration"] not in ["self_absolute_adcc", "self_relative_adcc", "self_relative_sigma"]:
            print(f"include calibration by external file here")
        else:
            po218_peaknum = [keynum for keynum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] if measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][keynum]=="po218"][0]
            po214_peaknum = [keynum for keynum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] if measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][keynum]=="po214"][0]
            if measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_absolute_adcc":
                adcc_selection_window_po218_left = measurement_data_dict["activity_data_input"]["po218_window"][0]
                adcc_selection_window_po218_right = measurement_data_dict["activity_data_input"]["po218_window"][1]
                adcc_selection_window_po214_left = measurement_data_dict["activity_data_input"]["po214_window"][0]
                adcc_selection_window_po214_right = measurement_data_dict["activity_data_input"]["po214_window"][1]
            elif measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_relative_adcc":
                adcc_selection_window_po218_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] -measurement_data_dict["activity_data_input"]["po218_window"][0]
                adcc_selection_window_po218_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] +measurement_data_dict["activity_data_input"]["po218_window"][1]
                adcc_selection_window_po214_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] -measurement_data_dict["activity_data_input"]["po214_window"][0]
                adcc_selection_window_po214_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] +measurement_data_dict["activity_data_input"]["po214_window"][1]
            elif measurement_data_dict["activity_data_input"]["flag_calibration"] == "self_relative_sigma":
                adcc_selection_window_po218_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] -(measurement_data_dict["activity_data_input"]["po218_window"][0] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po218_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["mu"] +(measurement_data_dict["activity_data_input"]["po218_window"][1] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po218_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po214_left = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] -(measurement_data_dict["activity_data_input"]["po214_window"][0] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["sigma"])
                adcc_selection_window_po214_right = measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["mu"] +(measurement_data_dict["activity_data_input"]["po214_window"][1] *measurement_data_dict["spectrum_data_output"]["peak_fits"][po214_peaknum]["fit_data"]["sigma"])

        # determining the measurement time window to be considered for the activity model fit
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the measurement time window")
        if measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] == "t_meas_f":
            t_max_ps = max(raw_data_ndarray["timestamp_ps"])
        elif type(measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1]) in [int,float]:
            t_max_ps = measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"][1] *(10**12)

        # determining the detected po218 and po214 decays within the measurement time
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): determining the detected po218 and po214 decays within the measurement time")
        time_window_ps = [0,t_max_ps]
        n_meas_po218 = len(raw_data_ndarray[
            (raw_data_ndarray["timestamp_ps"] >= time_window_ps[0]) &
            (raw_data_ndarray["timestamp_ps"] <= time_window_ps[1]) &
            (raw_data_ndarray["pulse_height_adc"] >= adcc_selection_window_po218_left) &
            (raw_data_ndarray["pulse_height_adc"] <= adcc_selection_window_po218_right)])
        n_meas_po214 = len(raw_data_ndarray[
            (raw_data_ndarray["timestamp_ps"] >= time_window_ps[0]) &
            (raw_data_ndarray["timestamp_ps"] <= time_window_ps[1]) &
            (raw_data_ndarray["pulse_height_adc"] >= adcc_selection_window_po214_left) &
            (raw_data_ndarray["pulse_height_adc"] <= adcc_selection_window_po214_right)])

        # calculating the number of background events
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): calculating the number of background events")
        n_bkg_expected_po218 = calc_number_of_expected_background_events(
            input_t_meas_f_ps = time_window_ps[1],
            adc_window = [adcc_selection_window_po218_left, adcc_selection_window_po218_right],
            background_measurements_abspath_list = measurement_data_dict["activity_data_input"]["background_measurements_list"])
        n_bkg_expected_po214 = calc_number_of_expected_background_events(
            input_t_meas_f_ps = time_window_ps[1],
            adc_window = [adcc_selection_window_po214_left, adcc_selection_window_po214_right],
            background_measurements_abspath_list = measurement_data_dict["activity_data_input"]["background_measurements_list"])

        # calculating the number of actual signal events        
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): calculating the number of actual signal events")
        n_sig_po218_mean, n_sig_po218_loweruncertainty, n_sig_po218_upperuncertainty = calc_number_of_signal_events(n_meas=n_meas_po218, n_bkg_expected=n_bkg_expected_po218, flag_verbose=measurement_data_dict["activity_data_input"]["flag_verbose"])
        n_sig_po214_mean, n_sig_po214_loweruncertainty, n_sig_po214_upperuncertainty = calc_number_of_signal_events(n_meas=n_meas_po214, n_bkg_expected=n_bkg_expected_po214, flag_verbose=measurement_data_dict["activity_data_input"]["flag_verbose"])

        # calculating the number of rn222 atoms present in the detection vessel at t_meas_i
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): calculating the number of rn222 atoms present at 't_meas_i'")
        n_rn222_t_meas_i_po218_mean, n_rn222_t_meas_i_po218_loweruncertainty, n_rn222_t_meas_i_po218_upperuncertainty = error_propagation_for_one_dimensional_function(
            function = get_n222rn0_from_detected_218po_decays,
            function_input_dict = {
                "N_mean" : n_sig_po218_mean,
                "N_loweruncertainty" : n_sig_po218_loweruncertainty,
                "N_upperuncertainty" : n_sig_po218_upperuncertainty,
            },
            function_parameter_dict = {
                "tf" : time_window_ps[0]/(10**12),
                "ti" : time_window_ps[1]/(10**12),
                "R" : 0,
                "n218po0" : 0},
            n_mc = 10**5,
            n_histogram_bins = 150,
            flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])
        n_rn222_t_meas_i_po214_mean, n_rn222_t_meas_i_po214_loweruncertainty, n_rn222_t_meas_i_po214_upperuncertainty = error_propagation_for_one_dimensional_function(
            function = get_n222rn0_from_detected_214bi_decays,
            function_input_dict = {
                "N_mean" : n_sig_po214_mean,
                "N_loweruncertainty" : n_sig_po214_loweruncertainty,
                "N_upperuncertainty" : n_sig_po214_upperuncertainty,
            },
            function_parameter_dict = {
                "tf" : time_window_ps[0]/(10**12),
                "ti" : time_window_ps[1]/(10**12),
                "R" : 0,
                "n218po0" : 0,
                "n218po0" : 0,
                "n214pb0" : 0,
                "n214bi0" : 0},
            n_mc = 10**5,
            n_histogram_bins = 150,
            flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

        # for some unknown reason 'get_n222rn0_from_detected_218po_decays' and 'get_n222rn0_from_detected_214bi_decays' sometimes output negative values
        n_rn222_t_meas_i_po218_mean = float(np.sqrt(n_rn222_t_meas_i_po218_mean**2))
        n_rn222_t_meas_i_po218_loweruncertainty = float(np.sqrt(n_rn222_t_meas_i_po218_loweruncertainty**2))
        n_rn222_t_meas_i_po218_upperuncertainty = float(np.sqrt(n_rn222_t_meas_i_po218_upperuncertainty**2))
        n_rn222_t_meas_i_po214_mean = float(np.sqrt(n_rn222_t_meas_i_po214_mean**2))
        n_rn222_t_meas_i_po214_loweruncertainty = float(np.sqrt(n_rn222_t_meas_i_po214_loweruncertainty**2))
        n_rn222_t_meas_i_po214_upperuncertainty = float(np.sqrt(n_rn222_t_meas_i_po214_upperuncertainty**2))

        # calculating the combined value of n_rn222_t_meas_i (currently the weighted mean of the po218 and po214 numbers)
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): combining the rn222 determined for both po218 and po214")
        n_rn222_t_meas_i_combined_mean, n_rn222_t_meas_i_combined_loweruncertainty, n_rn222_t_meas_i_combined_upperuncertainty= error_propagation_for_one_dimensional_function(
            function = calc_weighted_mean,
            function_input_dict = {
                "n_rn222_0_po218_mean" : n_rn222_t_meas_i_po218_mean,
                "n_rn222_0_po218_loweruncertainty" : n_rn222_t_meas_i_po218_loweruncertainty,
                "n_rn222_0_po218_upperuncertainty" : n_rn222_t_meas_i_po218_upperuncertainty,
                "n_rn222_0_po214_mean" : n_rn222_t_meas_i_po214_mean,
                "n_rn222_0_po214_loweruncertainty" : n_rn222_t_meas_i_po214_loweruncertainty,
                "n_rn222_0_po214_upperuncertainty" : n_rn222_t_meas_i_po214_upperuncertainty,
            },
            function_parameter_dict = {
                "n_rn222_0_po218_loweruncertaintyparam" : n_rn222_t_meas_i_po218_loweruncertainty,
                "n_rn222_0_po218_upperuncertaintyparam" : n_rn222_t_meas_i_po218_upperuncertainty,
                "n_rn222_0_po214_loweruncertaintyparam" : n_rn222_t_meas_i_po214_loweruncertainty,
                "n_rn222_0_po214_upperuncertaintyparam" : n_rn222_t_meas_i_po214_upperuncertainty,
            },
            n_mc = 10**5,
            n_histogram_bins = 150,
            flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

        #n_rn222_t_meas_i_combined_mean = 0.5*(n_rn222_t_meas_i_po218_mean +n_rn222_t_meas_i_po214_mean)
        #n_rn222_t_meas_i_combined_loweruncertainty = 0.5*(n_rn222_t_meas_i_po218_loweruncertainty +n_rn222_t_meas_i_po214_loweruncertainty)
        #n_rn222_t_meas_i_combined_upperuncertainty = 0.5*(n_rn222_t_meas_i_po218_upperuncertainty +n_rn222_t_meas_i_po214_upperuncertainty)

        # deciding whether to use the number obtained from po218, po214, or both (based on whether the po218 and po214 values are compatible with one another)
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): deciding whether to use the po218 or the po214 value")
        d = np.sqrt((n_rn222_t_meas_i_po218_mean -n_rn222_t_meas_i_po214_mean)**2)
        sd = np.sqrt((0.5*(n_rn222_t_meas_i_po218_loweruncertainty+n_rn222_t_meas_i_po218_upperuncertainty))**2 +(n_rn222_t_meas_i_po214_loweruncertainty+n_rn222_t_meas_i_po214_upperuncertainty)**2)
        if d/sd < 2:
            n_rn222_at_t_meas_i_mean = n_rn222_t_meas_i_combined_mean
            n_rn222_at_t_meas_i_loweruncertainty = n_rn222_t_meas_i_combined_loweruncertainty
            n_rn222_at_t_meas_i_upperuncertainty = n_rn222_t_meas_i_combined_upperuncertainty
        else:
            n_rn222_at_t_meas_i_mean = n_rn222_t_meas_i_po214_mean
            n_rn222_at_t_meas_i_loweruncertainty = n_rn222_t_meas_i_po214_loweruncertainty
            n_rn222_at_t_meas_i_upperuncertainty = n_rn222_t_meas_i_po214_upperuncertainty

        # extrapolating the equilibrium emanation activity
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): extrapolating the equilibrium emanation activity")
        if measurement_data_dict["activity_data_input"]["flag_activity_extrapolation"] == False:
            dt_trans = "NA"
            dt_ema = "NA"
            a_ema = "NA"
            #a_ema_error ="NA" 
        elif measurement_data_dict["activity_data_input"]["flag_activity_extrapolation"] in ["default"]:
            # retrieving the relevant times from the ELOG timestamps
            t_ema_i = measurement_data_dict[key_measurement_information]["t_ema_i"]
            t_trans_i = measurement_data_dict[key_measurement_information]["t_trans_i"]
            t_meas_i = measurement_data_dict[key_measurement_information]["t_meas_i"]
            t_meas_f = measurement_data_dict[key_measurement_information]["t_meas_f"]
            # calculating the relevant time deltas
            dt_meas = (timestamp_conversion(t_meas_f)-timestamp_conversion(t_meas_i)).total_seconds() # only used to be written to output
            dt_ema = (timestamp_conversion(t_trans_i)-timestamp_conversion(t_ema_i)).total_seconds()
            dt_trans = (timestamp_conversion(t_meas_i)-timestamp_conversion(t_trans_i)).total_seconds()
            # calculating the activities
            a_t_meas_i_mean = n_rn222_at_t_meas_i_mean *isotope_dict["rn222"]["decay_constant"]
            a_t_meas_i_loweruncertainty = n_rn222_at_t_meas_i_loweruncertainty *isotope_dict["rn222"]["decay_constant"]
            a_t_meas_i_upperuncertainty = n_rn222_at_t_meas_i_upperuncertainty *isotope_dict["rn222"]["decay_constant"]

            # extrapolating the radon activity from 't_meas_i' to 't_trans_i'
            a_t_trans_i_mean, a_t_trans_i_loweruncertainty, a_t_trans_i_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = extrapolate_radon_activity,
                function_input_dict = {
                    "known_activity_at_dt_known_bq_mean" : a_t_meas_i_mean,
                    "known_activity_at_dt_known_bq_loweruncertainty" : a_t_meas_i_loweruncertainty,
                    "known_activity_at_dt_known_bq_upperuncertainty" : a_t_meas_i_upperuncertainty,
                },
                function_parameter_dict = {
                    "flag_exp_rise_or_decay" : ["rise", "decay"][1],
                    "lambda_222rn" : isotope_dict["rn222"]["decay_constant"],
                    "known_dt_s" : dt_trans,
                    "dt_extrapolation_s" : 0,
                },
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

            # extrapolating the radon activity from 't_trans_i' to 'inf'
            a_ema_withoutdeteffcorr_mean, a_ema_withoutdeteffcorr_loweruncertainty, a_ema_withoutdeteffcorr_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = extrapolate_radon_activity,
                function_input_dict = {
                    "known_activity_at_dt_known_bq_mean" : a_t_trans_i_mean,
                    "known_activity_at_dt_known_bq_loweruncertainty" : a_t_trans_i_loweruncertainty,
                    "known_activity_at_dt_known_bq_upperuncertainty" : a_t_trans_i_upperuncertainty,
                },
                function_parameter_dict = {
                    "flag_exp_rise_or_decay" : ["rise", "decay"][0],
                    "lambda_222rn" : isotope_dict["rn222"]["decay_constant"],
                    "known_dt_s" : dt_ema,
                    "dt_extrapolation_s" : "inf",
                },
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])

        # detection efficiency correction
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): detection efficiency correction")
        if measurement_data_dict["activity_data_input"]["detection_efficiency_mean"] != 1:
            r_ema_mean, r_ema_loweruncertainty, r_ema_upperuncertainty = error_propagation_for_one_dimensional_function(
                function = detection_efficiency_correction,
                function_input_dict = {
                    "r_ema_mean" : a_ema_withoutdeteffcorr_mean,
                    "r_ema_loweruncertainty" : a_ema_withoutdeteffcorr_loweruncertainty,
                    "r_ema_upperuncertainty" : a_ema_withoutdeteffcorr_upperuncertainty,
                    "detection_efficiency_mean" : measurement_data_dict["activity_data_input"]["detection_efficiency_mean"],
                    "detection_efficiency_loweruncertainty" : measurement_data_dict["activity_data_input"]["detection_efficiency_loweruncertainty"],
                    "detection_efficiency_upperuncertainty" : measurement_data_dict["activity_data_input"]["detection_efficiency_upperuncertainty"],
                },
                function_parameter_dict = {},
                n_mc = 10**5,
                n_histogram_bins = 150,
                flag_verbose = measurement_data_dict["activity_data_input"]["flag_verbose"])
        else:
            r_ema_mean = a_ema_withoutdeteffcorr_mean
            r_ema_loweruncertainty = a_ema_withoutdeteffcorr_loweruncertainty
            r_ema_upperuncertainty = a_ema_withoutdeteffcorr_upperuncertainty

        # filling the 'activity_data_output_dict'
        if measurement_data_dict["activity_data_input"]["flag_verbose"] == True: print(f"get_activity_data_output_dict(): filling the 'activity_data_output_dict'")
        activity_data_output_dict = {
            "event_extraction" : {
                "po218_adcc_window" : [adcc_selection_window_po218_left, adcc_selection_window_po218_right],
                "po214_adcc_window" : [adcc_selection_window_po214_left, adcc_selection_window_po214_right],
            },
            "miscellaneous" : {
                "n_meas_po218" : n_meas_po218,
                "n_sig_po218" : n_sig_po218_mean,
                "n_bkg_expected_po218" : n_bkg_expected_po218,
                "n_meas_po214" : n_meas_po214,
                "n_sig_po214" : n_sig_po214_mean,
                "n_bkg_expected_po214" : n_bkg_expected_po214,
                "n_rn222_t_meas_i_po218_mean" : n_rn222_t_meas_i_po218_mean,
                "n_rn222_t_meas_i_po218_loweruncertainty" : n_rn222_t_meas_i_po218_loweruncertainty,
                "n_rn222_t_meas_i_po218_upperuncertainty" : n_rn222_t_meas_i_po218_upperuncertainty,
                "n_rn222_t_meas_i_po214_mean" : n_rn222_t_meas_i_po214_mean,
                "n_rn222_t_meas_i_po214_loweruncertainty" : n_rn222_t_meas_i_po214_loweruncertainty,
                "n_rn222_t_meas_i_po214_upperuncertainty" : n_rn222_t_meas_i_po214_upperuncertainty,
                "n_rn222_t_meas_i_combined_mean" : n_rn222_t_meas_i_combined_mean,
                "n_rn222_t_meas_i_combined_loweruncertainty" : n_rn222_t_meas_i_combined_loweruncertainty,
                "n_rn222_t_meas_i_combined_upperuncertainty" : n_rn222_t_meas_i_combined_upperuncertainty,
            },
            "activity_extrapolation" : {
                "delta_t_trans_s" : dt_trans,
                "delta_t_ema_s" : dt_ema,
                "delta_t_meas_s" : dt_meas,
                "r_ema_mean_bq" : r_ema_mean,
                "r_ema_loweruncertainty_bq" :  r_ema_loweruncertainty,
                "r_ema_upperuncertainty_bq" :  r_ema_upperuncertainty,
            },
        }
        print(activity_data_output_dict)


    return activity_data_output_dict


# This function is used to calculate the number of signal events from the measured events and calculated background events.
# The reason the write a dedicated function for this seemingly trivial task is, that I might expand this later to also account for the issues raised in the Feldman and Cousins paper.
# Also I was lazy and just used the Gaussian approximation... I should probably update that at some point in the future.
def calc_number_of_signal_events(
    n_meas,
    n_bkg_expected,
    flag_verbose):

    # calculating the Poissonian errors of the input measurements
    n_meas_loweruncertainty, n_meas_upperuncertainty = calc_poissonian_error(
        number_of_counts = n_meas,
        flag_mode = ["sqrt", "poissonian", "poissonian_symmetrical"][1])
    n_bkg_expected_loweruncertainty, n_bkg_expected_upperuncertainty = calc_poissonian_error(
        number_of_counts = n_bkg_expected,
        flag_mode = ["sqrt", "poissonian", "poissonian_symmetrical"][1])

    # calculating the error propagation on the number of signal events
    def sig_is_meas_minus_bkg(n_meas, n_bkg):
        n_sig = n_meas -n_bkg
        return n_sig
    n_sig_mean = sig_is_meas_minus_bkg(n_meas=n_meas, n_bkg=n_bkg_expected)
    y_mean, n_sig_loweruncertainty, n_sig_upperuncertainty = error_propagation_for_one_dimensional_function(
        function = sig_is_meas_minus_bkg,
        function_input_dict = {
            "n_meas_mean" : n_meas,
            "n_meas_loweruncertainty" : n_meas_loweruncertainty,
            "n_meas_upperuncertainty" : n_meas_upperuncertainty,
            "n_bkg_mean" : n_bkg_expected,
            "n_bkg_loweruncertainty" : n_bkg_expected_loweruncertainty,
            "n_bkg_upperuncertainty" : n_bkg_expected_upperuncertainty,
        },
        function_parameter_dict = {},
        n_mc = 10**5,
        n_histogram_bins = 150,
        flag_verbose = flag_verbose,
    )
    return n_sig_mean, n_sig_loweruncertainty, n_sig_upperuncertainty


# This function is used to calculate the expected number of background events, based on the given background data 'raw_data.npy'.
def calc_number_of_expected_background_events(
    input_t_meas_f_ps,
    adc_window,
    background_measurements_abspath_list):
    
    # calculating the respective 't_meas_f_ps' timestamps and storing timestamps and bkg arrays in dictionary
    background_arrays_list = [np.load(abspath_bkg_data) for abspath_bkg_data in background_measurements_abspath_list]
    t_meas_f_ps_dict = {}
    for bkg_data in background_arrays_list:
        t_meas_f_ps = np.max(bkg_data["timestamp_ps"])
        t_meas_f_ps_dict.update({str(t_meas_f_ps) : bkg_data})
    t_meas_f_ps_list = [int(key) for key in [*t_meas_f_ps_dict]]
    t_meas_f_ps_list.sort()
    #print(f"your timestamp: {input_t_meas_f_ps/(24*60*60*(10**12)):.4f} days")
    #for i, tmstp in enumerate(t_meas_f_ps_list):
    #    print(f"{i}th file: {tmstp/(24*60*60*(10**12)):.4f} days")

    # looping over all values of 't_meas_f_ps' (in increasing order).
    # In each iteration all counts detected therein are added and divided by the number of contributing files.
    # The new value is then added to 'expected_number_of_background_events'
    expected_number_of_background_events = 0
    list_ctr = 0
    tstmp_ps_current_low = 0
    tstmp_ps_current_high = t_meas_f_ps_list[list_ctr] #  
    # adding all the bkg arrays whose measurement duration is smaller than 'input_t_meas_f_ps'
    while tstmp_ps_current_high < input_t_meas_f_ps:
        n_add = 0
        for i in range(list_ctr, len(t_meas_f_ps_list), 1):
            bkg_array = t_meas_f_ps_dict[str(t_meas_f_ps_list[i])]
            n_add += len(bkg_array[
                (bkg_array["timestamp_ps"]>tstmp_ps_current_low) &
                (bkg_array["timestamp_ps"]<tstmp_ps_current_high) &
                (bkg_array["pulse_height_adc"]>adc_window[0]) &
                (bkg_array["pulse_height_adc"]<adc_window[1])  ])
        n_add = n_add/(len(t_meas_f_ps_list)-list_ctr)
        expected_number_of_background_events += n_add
        tstmp_ps_current_low = t_meas_f_ps_list[list_ctr]
        tstmp_ps_current_high = t_meas_f_ps_list[list_ctr+1]
        list_ctr += 1
    # adding all the bkg arrays whose measurement duration is bigger than 'input_t_meas_f_ps'
    tstmp_ps_current_high = input_t_meas_f_ps
    n_add = 0
    for i in range(list_ctr, len(t_meas_f_ps_list), 1):
        #print(f"this is fine, i={i}")
        bkg_array = t_meas_f_ps_dict[str(t_meas_f_ps_list[i])]
        #print(f"this as well, i={i}")
        n_add += len(bkg_array[
            (bkg_array["timestamp_ps"]>tstmp_ps_current_low) &
            (bkg_array["timestamp_ps"]<tstmp_ps_current_high) &
            (bkg_array["pulse_height_adc"]>adc_window[0]) &
            (bkg_array["pulse_height_adc"]<adc_window[1])  ])
        #print(f"just like this, i={i}")
    n_add = n_add/(len(t_meas_f_ps_list)-list_ctr)
    expected_number_of_background_events += n_add

    return expected_number_of_background_events

#######################################
### Decay Chain Model
#######################################


### Rn222


# This function returns an analytical expression for the activity of Rn222 nuclei (retrieved from the linear Rn222 decay chain model with additional emanation source term).
def rn222(
    t, # time in s
    n222rn0, # number of initial rn222 nuclei at t_i
    r): # 222rn emanation rate in Bq
    
    if t<= 0:
        return 0
    else:
        return isotope_dict["rn222"]["decay_constant"] *((isotope_dict["rn222"]["decay_constant"]*n222rn0 + (-1 + np.exp(isotope_dict["rn222"]["decay_constant"]*t))*r)/(np.exp(isotope_dict["rn222"]["decay_constant"]*t)*isotope_dict["rn222"]["decay_constant"]))


# This function is used to calculate the expected number of rn222 decays by integrating the analytical rn222 decay model over time.
def get_number_of_expected_rn222_decays_between_ti_and_tf(
    R, # rn222 equilibrium emanation activity (in 1/s)
    tf, # time at which the data acquisition was started (in seconds after t=0) 
    ti = 0, # time at which the data acquisition was stopped (in seconds after t=0) 
    n222rn0 = 0,  # number of rn222 atoms present at time t=0
):
    return ((n222rn0 - R/isotope_dict["rn222"]["decay_constant"]) / np.exp(isotope_dict["rn222"]["decay_constant"]*ti)
+ (-n222rn0 + R/isotope_dict["rn222"]["decay_constant"]) / np.exp(isotope_dict["rn222"]["decay_constant"]*tf)
+ R*(tf - ti))


# This function is used to calculate the rn222 emanation activity by solving the analytical rn222 decay model for 'R'.
def get_r_from_detected_222rn_decays(
    N, # number of detected rn222 decays
    tf, # time in seconds at which the data acquisition was stopped, measured after t=0
    ti = 0, # time in seconds at which the data acquisition was stopped, measured after t=0
    n222rn0 = 0, # number of rn222 atoms present at t=0
):
    return ((isotope_dict["rn222"]["decay_constant"]*(np.exp(isotope_dict["rn222"]["decay_constant"]*(tf + ti))*N
-np.exp(isotope_dict["rn222"]["decay_constant"]*tf)*n222rn0
+np.exp(isotope_dict["rn222"]["decay_constant"]*ti)*n222rn0))/(-np.exp(isotope_dict["rn222"]["decay_constant"]*tf)
+np.exp(isotope_dict["rn222"]["decay_constant"]*ti)
+np.exp(isotope_dict["rn222"]["decay_constant"]*(tf + ti))*isotope_dict["rn222"]["decay_constant"]*(tf - ti)))


# This function is used to calculate the initial number of radon atoms 'n222rn0' by solving the analytical rn222 decay model for 'n222rn0'.
def get_n222rn0_from_detected_222rn_decays(
    N, # number of detected rn222 decays
    tf, # time in seconds at which the data acquisition was stopped, measured after t=0
    ti = 0, # time in seconds at which the data acquisition was stopped, measured after t=0
    R = 0, # radon equilibrium emanation activity
):
    return ((np.exp(isotope_dict["rn222"]["decay_constant"]*tf)*R - np.exp(isotope_dict["rn222"]["decay_constant"]*ti)*R
+np.exp(isotope_dict["rn222"]["decay_constant"]*(tf + ti))
*isotope_dict["rn222"]["decay_constant"]*(N + R*(-tf + ti)))/((np.exp(isotope_dict["rn222"]["decay_constant"]*tf)
-np.exp(isotope_dict["rn222"]["decay_constant"]*ti))*isotope_dict["rn222"]["decay_constant"]))


### Po218


# This function returns an analytical expression for the activity of Po218 nuclei (retrieved from the linear Rn222 decay chain model with additional emanation source term).
def po218(
    t, # time in s
    n222rn0, # number of initial rn222 nuclei at t_i
    n218po0, # number of initial rn222 nuclei at t_i
    n214pb0, # number of initial pb214 nuclei at t_i, the function does not depend on this input, nevertheless it is needed for the combined fit
    n214bi0, # number of initial bi214 nuclei at t_i, the function does not depend on this input, nevertheless it is needed for the combined fit
    r): # 222rn emanation rate in Bq

    if t<= 0:
        return 0
    else:
        return isotope_dict["po218"]["decay_constant"] *((isotope_dict["po218"]["decay_constant"]**2*n218po0 - (-1 + np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*t)))*isotope_dict["rn222"]["decay_constant"]*r - isotope_dict["po218"]["decay_constant"]*(isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) + (-np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r))/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*t))*(isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"]))))


# This function is used to calculate the expected number of po218 decays by integrating the analytical po218 decay model over time.
def get_number_of_expected_po218_decays_between_ti_and_tf(
    R, # rn222 equilibrium emanation activity (in 1/s)
    tf, # time at which the data acquisition was started (in seconds after t=0) 
    ti = 0, # time at which the data acquisition was stopped (in seconds after t=0) 
    n222rn0 = 0,  # number of rn222 atoms present at time t=0
    n218po0 = 0,  # number of po218 atoms present at time t=0
):
    return ((np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*(tf + ti)))*
isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - R) - 
np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*ti + isotope_dict["po218"]["decay_constant"]*(tf + ti)))*
isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - R) + 
np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))*
isotope_dict["rn222"]["decay_constant"]*((-isotope_dict["po218"]["decay_constant"]**2)*n218po0 + 
isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0) - 
isotope_dict["rn222"]["decay_constant"]*R) + 
np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))*
isotope_dict["rn222"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]**2*n218po0 - 
isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0) + 
isotope_dict["rn222"]["decay_constant"]*R) + 
np.exp(np.float128((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf + ti)))*
isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]*R*(tf - ti))/
np.exp(np.float128((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf + ti)))/
(isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]))


# This function is used to calculate the initial number of radon atoms 'n222rn0' by solving the analytical po218 decay model for 'n222rn0'.
def get_n222rn0_from_detected_218po_decays(
    N, # number of detected po218 decays
    tf, # time in seconds at which the data acquisition was stopped, measured after t=0
    ti = 0, # time in seconds at which the data acquisition was stopped, measured after t=0
    R = 0, # radon equilibrium emanation activity
    n218po0 = 0, # initial number of po218 atoms
):
    return ((np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["po218"]["decay_constant"]**2*R
-np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*ti + isotope_dict["po218"]["decay_constant"]*(tf + ti)))*isotope_dict["po218"]["decay_constant"]**2*R
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))
*isotope_dict["rn222"]["decay_constant"]*((-isotope_dict["po218"]["decay_constant"]**2)*n218po0
+isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n218po0 - isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))
*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]**2*n218po0
-isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n218po0 + isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"]))*(tf + ti))
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(N + R*(-tf + ti)))
/(isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["po218"]["decay_constant"]
-np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*ti + isotope_dict["po218"]["decay_constant"]*(tf + ti)))*isotope_dict["po218"]["decay_constant"]
-np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))*isotope_dict["rn222"]["decay_constant"]
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*(tf + ti)))*isotope_dict["rn222"]["decay_constant"])))


### Pb214


# This function returns an analytical expression for the activity of Pb214 nuclei (retrieved from the linear Rn222 decay chain model with additional emanation source term).
def pb214(
    t, # time in s
    n222rn0, # number of initial rn222 nuclei at t_i
    n218po0, # number of initial po218 nuclei at t_i
    n214pb0, # number of initial pb214 nuclei at t_i
    r): # 222rn emanation rate in Bq
    
    if t<= 0:
        return 0
    else:
        return isotope_dict["pb214"]["decay_constant"] *((isotope_dict["pb214"]["decay_constant"]**3*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*n214pb0 + (-1 + np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*t)))*isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]*r + isotope_dict["pb214"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]**2*n214pb0 - isotope_dict["po218"]["decay_constant"]**2*(n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0) + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n218po0 - np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0 + (-np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)) + np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*n222rn0) + (np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]*r + (-np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*r) + isotope_dict["pb214"]["decay_constant"]*((-isotope_dict["po218"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]**2*(n214pb0 - (-1 + np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) + (np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*t)) - np.exp((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*isotope_dict["rn222"]["decay_constant"]**2*r + isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*(n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) + (-np.exp(isotope_dict["pb214"]["decay_constant"]*t) + np.exp(np.float128((isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r)))/np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*t))/(isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))


### Bi214


# This function returns an analytical expression for the activity of Bi214 nuclei (retrieved from the linear Rn222 decay chain model with additional emanation source term).
def bi214(
    t, # time in s
    n222rn0, # number of initial rn222 nuclei at t_i
    n218po0, # number of initial po218 nuclei at t_i
    n214pb0, # number of initial pb214 nuclei at t_i
    n214bi0, # number of initial bi214 nuclei at t_i
    r): # 222rn emanation rate in Bq

    if t<= 0:
        return 0
    else:
        return (isotope_dict["bi214"]["decay_constant"] *((isotope_dict["bi214"]["decay_constant"]**4*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*n214bi0 
+ (-1 + np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)))*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*r 
+ isotope_dict["bi214"]["decay_constant"]**2*((-isotope_dict["pb214"]["decay_constant"]**2)*(isotope_dict["po218"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*(isotope_dict["rn222"]["decay_constant"]**2*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + isotope_dict["po218"]["decay_constant"]**2*(n214bi0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t)))*(n214pb0 + n218po0)) + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 
- (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*(2*n214pb0 + n218po0))) 
+ isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]
*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]*r 
+ (-np.exp(isotope_dict["bi214"]["decay_constant"]*t) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))
*isotope_dict["rn222"]["decay_constant"]*r) + isotope_dict["pb214"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*isotope_dict["rn222"]["decay_constant"]**2*n218po0 
+ isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**3*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 + (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) 
- np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) 
- (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**3*r + isotope_dict["po218"]["decay_constant"]**3*(isotope_dict["rn222"]["decay_constant"]
*(n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 
- np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n222rn0 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) 
+ (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r)) 
+ isotope_dict["pb214"]["decay_constant"]**3*(isotope_dict["po218"]["decay_constant"]**2*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0) 
- isotope_dict["rn222"]["decay_constant"]*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*r) + isotope_dict["po218"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n218po0 + (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n222rn0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r))) + isotope_dict["bi214"]["decay_constant"]
*((-isotope_dict["pb214"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]**2*(n214bi0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))
*(n214pb0 + n218po0 + n222rn0)) + (np.exp(isotope_dict["bi214"]["decay_constant"]*t) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]**2*r + isotope_dict["pb214"]["decay_constant"]**2
*((np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*isotope_dict["rn222"]["decay_constant"]**2*n218po0 - isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]**3*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 
- (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) 
+ (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**3*r + isotope_dict["po218"]["decay_constant"]**3
*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))
*n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) 
+ (-np.exp(isotope_dict["bi214"]["decay_constant"]*t) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*t)))*r)) + isotope_dict["pb214"]["decay_constant"]**3*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**2
*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 
- (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) 
- (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**2*r - isotope_dict["po218"]["decay_constant"]**2
*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))
*n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) 
+ (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*t)))*r))) + isotope_dict["bi214"]["decay_constant"]**3*((-isotope_dict["pb214"]["decay_constant"]**3)
*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["po218"]["decay_constant"] 
+ isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]*n214bi0 + isotope_dict["rn222"]["decay_constant"]*n214bi0 
+ (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*r) 
+ isotope_dict["pb214"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"]**2*(n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + (-np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*n218po0) + isotope_dict["po218"]["decay_constant"]
*(np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*isotope_dict["rn222"]["decay_constant"]*n218po0 
- np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - r) 
+ np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t))*r) + isotope_dict["rn222"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n214pb0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*r)) + isotope_dict["pb214"]["decay_constant"]
*(isotope_dict["po218"]["decay_constant"]**3*n214bi0 - isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**2
*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 
+ (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) - isotope_dict["rn222"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n214bi0 
+ (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*r) 
+ isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0 
- np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["pb214"]["decay_constant"])*t))*(n218po0 + n222rn0)) + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) 
+ np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] 
- isotope_dict["rn222"]["decay_constant"])*t)))*r))))/np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t))/(isotope_dict["bi214"]["decay_constant"]
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"]))))


# This function is used to calculate the expected number of bi214 decays by integrating the analytical bi214 decay model over time.
def get_number_of_expected_bi214_decays_between_ti_and_tf(
    R, # rn222 equilibrium emanation activity (in 1/s)
    tf, # time at which the data acquisition was started (in seconds after t=0) 
    ti = 0, # time at which the data acquisition was stopped (in seconds after t=0) 
    n222rn0 = 0,  # number of rn222 atoms present at time t=0
    n218po0 = 0,  # number of po218 atoms present at time t=0
    n214pb0 = 0,  # number of pb214 atoms present at time t=0
    n214bi0 = 0,  # number of bi214 atoms present at time t=0
):
    return ((np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]**2
*isotope_dict["po218"]["decay_constant"]**2
*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - R)
-np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]**2
*isotope_dict["po218"]["decay_constant"]**2
*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - R)
+np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]**2
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*((-isotope_dict["po218"]["decay_constant"]**2)*n218po0
+isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0)
-isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]**2
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]**2*n218po0
-isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0)
+isotope_dict["rn222"]["decay_constant"]*R)
-np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["pb214"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"]**3*n214pb0
-isotope_dict["pb214"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n214pb0
+isotope_dict["po218"]["decay_constant"]*(n214pb0 + n218po0))
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(n214pb0 + n218po0 + n222rn0)
-isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*tf
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + (isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf
+ti)))*isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"]**3*n214pb0
-isotope_dict["pb214"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n214pb0
+isotope_dict["po218"]["decay_constant"]*(n214pb0 + n218po0))
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(n214pb0 + n218po0 + n222rn0)
-isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["bi214"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["pb214"]["decay_constant"]*(tf + ti) + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*((-isotope_dict["bi214"]["decay_constant"]**4)*n214bi0
+isotope_dict["bi214"]["decay_constant"]**3*((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*n214bi0
+isotope_dict["pb214"]["decay_constant"]*(n214bi0 + n214pb0))
-isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0)
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0))
+isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0 + n222rn0)
-isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf + (isotope_dict["pb214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf + ti)))
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"]**4*n214bi0
-isotope_dict["bi214"]["decay_constant"]**3*((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*n214bi0
+isotope_dict["pb214"]["decay_constant"]*(n214bi0 + n214pb0))
+isotope_dict["bi214"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0)
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0))
-isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0 + n222rn0)
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*R)
+np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]
*R*(tf - ti))
/np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf
+ti)))/(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]))


# This function is used to calculate the initial number of radon atoms 'n222rn0' by solving the analytical po218 decay model for 'n222rn0'.
def get_n222rn0_from_detected_214bi_decays(
    N, # number of detected rn222 decays
    tf, # time in seconds at which the data acquisition was stopped, measured after t=0
    ti = 0, # time in seconds at which the data acquisition was stopped, measured after t=0
    R = 0, # radon equilibrium emanation activity
    n218po0 = 0, # initial number of po218 atoms
    n214pb0 = 0, # initial number of pb214 atoms
    n214bi0 = 0, # initial number of bi214 atoms
):
    return ((np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])*(tf + ti)))*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*(-N + (isotope_dict["bi214"]["decay_constant"]**3
*n214bi0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214bi0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
-(isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214bi0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]**3
*n214bi0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]**2
*n214pb0)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*tf))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
-(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]**2
*n214pb0)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214pb0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
-(isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214pb0)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214pb0)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*tf))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
-(isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n214pb0)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]**2*((isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*n214bi0
+isotope_dict["pb214"]["decay_constant"]*(n214bi0 + n214pb0)))/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])
*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) - (isotope_dict["bi214"]["decay_constant"]**2*((isotope_dict["po218"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])*n214bi0
+isotope_dict["pb214"]["decay_constant"]*(n214bi0 + n214pb0)))
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) - (isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*tf))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) - (isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*tf))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"]))) - (isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*n218po0)
/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["rn222"]["decay_constant"]*n214pb0
+isotope_dict["po218"]["decay_constant"]*(n214pb0 + n218po0)))/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])
*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["bi214"]["decay_constant"]
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["rn222"]["decay_constant"]*n214pb0
+isotope_dict["po218"]["decay_constant"]*(n214pb0 + n218po0)))
/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*ti))*((-isotope_dict["bi214"]["decay_constant"] + isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0)
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0)))
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0)
+isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*(n214bi0 + n214pb0 + n218po0)))
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"]))) + (isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)
/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*ti))*(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*ti))*(isotope_dict["pb214"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*ti))*(isotope_dict["po218"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf))*(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]
*tf))*(isotope_dict["pb214"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["pb214"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"]
-isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf))*((isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["po218"]["decay_constant"]
*tf))*(isotope_dict["po218"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])*(-isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])))
+(isotope_dict["bi214"]["decay_constant"]*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*R)/(np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]
*ti))*(isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])))
+R*(tf - ti)))/((-np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*tf
+isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["bi214"]["decay_constant"]*(tf + ti)
+isotope_dict["pb214"]["decay_constant"]*(tf + ti))))
*isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])
*isotope_dict["pb214"]["decay_constant"]
*isotope_dict["po218"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]*isotope_dict["po218"]["decay_constant"]
*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["po218"]["decay_constant"])
-np.exp(np.float128(isotope_dict["pb214"]["decay_constant"]*tf
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + (isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf
+ti)))*isotope_dict["bi214"]["decay_constant"]
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]
-isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]
+np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]
*tf + (isotope_dict["pb214"]["decay_constant"] + isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(tf + ti)))
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])
*isotope_dict["po218"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])
*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]
+np.exp(np.float128(isotope_dict["po218"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))*isotope_dict["bi214"]["decay_constant"]
*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])
-np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["po218"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["pb214"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*isotope_dict["pb214"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["pb214"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])
+np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["pb214"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["bi214"]["decay_constant"]*(tf + ti) + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]
*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["bi214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["po218"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])
-np.exp(np.float128(isotope_dict["rn222"]["decay_constant"]*tf + isotope_dict["bi214"]["decay_constant"]*ti + isotope_dict["rn222"]["decay_constant"]*ti
+isotope_dict["pb214"]["decay_constant"]*(tf + ti) + isotope_dict["po218"]["decay_constant"]*(tf + ti)))
*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]
*(-isotope_dict["pb214"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(-isotope_dict["po218"]["decay_constant"]
+isotope_dict["rn222"]["decay_constant"])))




#    # looping over all viable candidates from analysis list
#    #for key in [*peak_data_dict]:
#    if False:
#        if "isotope_data" in peak_data_dict[key]:
#            if peak_data_dict[key]["isotope_data"]["label"] in analysis_list:
#            
#                # preparations depending on the peak to be analyzed
#                analysis_list_keys.append(key)
#                if peak_data_dict[key]["isotope_data"]["label"]=="po218":
#                    func = monxeana.integral_function_po218
#                    p_opt_bounds = ([0,0,0],[+np.inf,10,+np.inf])
#                    p_opt_guess = [0,0,0.02]
#                elif peak_data_dict[key]["isotope_data"]["label"]=="po214":
#                    func = monxeana.integral_function_po214
#                    p_opt_bounds = ([0,0,0,0,0],[+np.inf,10,10,10,+np.inf])
#                    p_opt_guess = [0,0,0,0,0.02]

#                # extracting the detected counts per time bin
#                isotope_activity_data = []
#                for i in range(len(timestamp_edges_ps)-1):
#                    isotope_activity_data.append(len(data_raw[
#                        (data_raw["timestamp_ps"] >= timestamp_edges_ps[i]) &
#                        (data_raw["timestamp_ps"] <= timestamp_edges_ps[i+1]) &
#                        (data_raw["pulse_height_adc"] >= peak_data_dict[key]["fit_data"]["mu"] -adc_selection_window_left) &
#                        (data_raw["pulse_height_adc"] <= peak_data_dict[key]["fit_data"]["mu"] +adc_selection_window_right)]))
#                isotope_activity_data_errors = []
#                for i in range(len(isotope_activity_data)):
#                    isotope_activity_data_errors.append(np.sqrt(isotope_activity_data[i]))

#                # fitting the measured activities
#                p_opt, p_cov = curve_fit(
#                    f =  func,
#                    xdata = timestamp_centers_seconds,
#                    ydata = isotope_activity_data,
#                    sigma = isotope_activity_data_errors,
#                    absolute_sigma = True,
#                    bounds = p_opt_bounds,
#                    #method = 'lm', # "lm" cannot handle covariance matrices with deficient rank
#                    p0 = p_opt_guess)
#                print(p_opt)
#                emanation_rates.append(p_opt[-1])
#                p_err = np.sqrt(np.diag(p_cov))
#                emanation_rates_errors.append(p_err[-1])
#                ylim = [0,1.05*max([isotope_activity_data[i]+isotope_activity_data_errors[i] for i in range(len(timestamp_centers_seconds))])]
#                ax1.set_ylim(ylim)

#                # plotting the fit
#                plt.plot(
#                    plot_x_vals,
#                    [func(ts*monxeana.isotope_dict["rn222"]["half_life"], *p_opt) for ts in plot_x_vals],
#                    linewidth = 2,
#                    linestyle = "-",
#                    color = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["color"],
#                    alpha = 1,
#                    label = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["latex_label"] +" (fit)",
#                    zorder = 30)

#                # plotting the activity data
#                plt.plot(
#                    [ts/monxeana.isotope_dict["rn222"]["half_life"] for ts in timestamp_centers_seconds],
#                    isotope_activity_data,
#                    linewidth = 1,
#                    marker = "o",
#                    markersize = 3.8,
#                    markerfacecolor = "white",
#                    markeredgewidth = 0.5,
#                    markeredgecolor = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["color"],
#                    linestyle = "",
#                    alpha = 1,
#                    label = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["latex_label"] +" (data)",
#                    zorder = 1)
#                plt.errorbar(
#                    marker = "", # plotting just the errorbars
#                    linestyle = "",
#                    fmt = '',
#                    x = [ts/monxeana.isotope_dict["rn222"]["half_life"] for ts in timestamp_centers_seconds],
#                    y = isotope_activity_data,
#                    yerr = isotope_activity_data_errors,
#                    xerr = [0.5*activity_interval_ps/(10**12*monxeana.isotope_dict["rn222"]["half_life"]) for ts in timestamp_centers_seconds],
#                    ecolor = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["color"],
#                    elinewidth = 0.5,
#                    capsize = 1.2,
#                    barsabove = True,
#                    capthick = 0.5)


# This function is used to plot the activity data.
def plot_activity_model_fit(
    measurement_data_dict,
    raw_data_ndarray,
    # plot stuff
    plot_aspect_ratio = 9/16,
    plot_figsize_x_inch = miscfig.standard_figsize_x_inch,
    plot_x_lim_s = ["max", [0, 12*24*60*60]][0],
    plot_annotate_comments_dict = {},
    plot_legend_dict = {},
    # flags
    flag_plot_fits = True,
    flag_x_axis_units = ["seconds", "radon_half_lives"][1],
    flag_output_abspath_list = [],
    flag_comments = [],
    flag_show = True,
    flag_errors = [False, "poissonian"][1],
    #flag_plot_fits = [0,1,2],
    flag_preliminary = [False, True][0],
):

    # figure formatting
    fig, ax1 = plt.subplots(figsize=[plot_figsize_x_inch,plot_figsize_x_inch*plot_aspect_ratio], dpi=150, constrained_layout=True)
#    if plot_x_lim_s == "max" and measurement_data_dict['activity_data_input']["flag_t_meas_ana_s"] == "delta_t_meas":
#        xlim = [0,max(raw_data_ndarray["timestamp_ps"])/(10**12)]
#    elif plot_x_lim_s == "max" and measurement_data_dict['activity_data_input']["flag_t_meas_ana_s"] != "delta_t_meas":
#        xlim = [0,measurement_data_dict['activity_data_input']["flag_t_meas_ana_s"]]
#    else:
#        xlim = plot_x_lim_s
    if plot_x_lim_s == []:
        xlim = measurement_data_dict["activity_data_input"]["delta_t_meas_window_s"]
        if xlim[1] == "t_meas_f":
            xlim[1] = max(raw_data_ndarray["timestamp_ps"])/(10**12)
    else:
        xlim = plot_x_lim_s
    if flag_x_axis_units == "seconds":
        latex_time_unit_string = r"$\mathrm{s}$"
        time_unit_conversion_factor = 1
    elif flag_x_axis_units == "radon_half_lives":
        time_unit_conversion_factor = 1/(isotope_dict["rn222"]["half_life_s"])
        latex_time_unit_string = r"$T^{\mathrm{half}}_{^{222}\mathrm{Rn}}$"
    ax1.set_xlabel(r"time since $t_{\mathrm{meas}}^{\mathrm{i}}$ / " +latex_time_unit_string)
    ax1.set_ylabel(r"decays detected within $" +f"{measurement_data_dict['activity_data_input']['activity_interval_ps']/(10**12*60*60):.1f}" +"\,\mathrm{h}$")
    fit_plot_x_vals_s = np.linspace(start=xlim[0], stop=xlim[1], endpoint=True, num=500)
    xlim = [xlim[0]*time_unit_conversion_factor, xlim[1]*time_unit_conversion_factor]
    ax1.set_xlim(xlim)

    # plotting the activity data
    for isotope in ["po218", "po214"]:
#        plt.plot(
#            [ts*time_unit_conversion_factor for ts in measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["timestamp_centers_seconds"]],
#            measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["decays_per_activity_interval"],
#            linewidth = 1,
#            marker = "o",
#            markersize = 3.8,
#            markerfacecolor = "white",
#            markeredgewidth = 0.5,
#            markeredgecolor = isotope_dict[isotope]["color"],
#            linestyle = "",
#            alpha = 1,
#            label = isotope_dict[isotope]["latex_label"] +" (data)",
#            zorder = 1)
        plt.errorbar(
            marker = "o", # plotting just the errorbars
            markersize = 3.8,
            markerfacecolor = "white",
            markeredgewidth = 0.5,
            markeredgecolor = isotope_dict[isotope]["color"],
            linestyle = "",
            fmt = '',
            x = [ts*time_unit_conversion_factor for ts in measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["timestamp_centers_seconds"]],
            y = measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["decays_per_activity_interval"],
            yerr = measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["decays_per_activity_interval_errors_lower"],
            xerr = [0.5*measurement_data_dict["activity_data_input"]["activity_interval_ps"]*(1/10**12)*time_unit_conversion_factor for ts in measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["timestamp_centers_seconds"]],
            ecolor = isotope_dict[isotope]["color"],
            elinewidth = 0.5,
            capsize = 1.2,
            barsabove = True,
            label = isotope_dict[isotope]["latex_label"] +" (data)",
            capthick = 0.5)

    # plotting the fits
    if flag_plot_fits == True:
        for isotope in ["po218", "po214"]:
            peaknum = [keynum for keynum in [*measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"]] if measurement_data_dict["spectrum_data_input"]["a_priori_knowledge"][keynum]==isotope][0]

            fit_function = integral_function_po218 if isotope == "po218" else integral_function_po214
            x_vals = [ts*time_unit_conversion_factor for ts in fit_plot_x_vals_s]
            y_vals = fit_function(
                fit_plot_x_vals_s,
                measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["n222rn0"],
                measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["n218po0"],
                measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["n214pb0"],
                measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["n214bi0"],
                measurement_data_dict["activity_data_output"]["activity_model_fit_results"][isotope]["r"]
                )
            plt.plot(
                x_vals,
                y_vals,
                linewidth = 2,#measurement_data_dict["peak_data"],
                linestyle = "-",
                color = isotope_dict[isotope]["color"],
                alpha = 1,
                label = isotope_dict[isotope]["latex_label"] +" (fit)",
                zorder = 30)

    # marking as 'preliminary'
    if flag_preliminary == True:
        plt.text(
            x = 0.97,
            y = 0.95,
            transform = ax1.transAxes,
            s = "preliminary",
            color = "red",
            fontsize = 13,
            verticalalignment = 'center',
            horizontalalignment = 'right')


    # annotating comments
    if flag_comments != []:
        comment_list = []
        if "red_chi_square" in flag_comments:
            comment_list.append(r"$\chi^2_{\mathrm{red}}=" +f"{measurement_data_dict['activity_data_output']['activity_model_fit_results']['reduced_chi_square']:.2f}" +r"$")
        if "amf_rema_result" in flag_comments:
            comment_list.append(r"${^{\mathrm{AMF}}R}^{\mathrm{ema}}_{^{222}\mathrm{Rn}}=(" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_mean_bq']*1000:.1f}" +r"_{-" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_loweruncertainty_bq']*1000:.1f}" +r"}^{+" +f"{measurement_data_dict['activity_data_output']['activity_extrapolation']['r_ema_upperuncertainty_bq']*1000:.1f}" +r"})\,\mathrm{mBq}$")
        annotate_comments(
            comment_ax = ax1,
            comment_list = comment_list,
            **plot_annotate_comments_dict,
        )


    # legend
    if plot_legend_dict != {}:
        plt.legend(**plot_legend_dict)


    # saving the output plot
    if flag_show == True:
        plt.show()
    if flag_output_abspath_list != []:
        for output_abspath in flag_output_abspath_list:
            fig.savefig(output_abspath)

    return





#######################################
### Sensitivity Stuff
#######################################


# This function is utilized to calculate the upper limit 's_up' at C.L. 'cl' for a counting experiment in which 'n_obs' events have been measured with an expected background of 'b' events.
# For reference have a look at Glen Cowan's CERN talk 'statistics for particle physicists', slide 6
def calculate_upper_limit_for_counting_experiment(
    n_obs, # number of observed events, typically an integer
    b, # mean number of expected background events, can also be float
    cl = 0.9, # C.L. corresponding to significance alpha
):
    s_up = 0.5*chi2.ppf(cl, 2*(n_obs+1), loc=0, scale=1) -b
    return s_up


# This function is use to calculate the expected limit, i.e., the sensitivity, of a counting experiment.
def calculate_expected_upper_limit_in_n_for_expected_background(
    n_bkg_expected,
    confidence_level = 0.90,
    n_samples = 10**4,
    random_seed = None,
):
    rng = np.random.default_rng(seed=random_seed)
    poisson_data = rng.poisson(n_bkg_expected, n_samples)
    upper_limit_data = [calculate_upper_limit_for_counting_experiment(cl=confidence_level, n_obs=pd, b=n_bkg_expected) for pd in poisson_data]
    mean, lower, upper = get_asymmetric_intervals_around_center_val_for_interpolated_discrete_distribution(
        distribution_bin_centers = [],
        distribution_counts = [],
        distribution_data = upper_limit_data,
        distribution_center_value = None,
        interval_width_lower_percent = 0.6827/2,
        interval_width_upper_percent = 0.6827/2,
        flag_verbose = False)
    return mean, lower, upper


# This function is use to calculate the expected limit or the emanation rate, i.e., the sensitivity, of the MonXe radon emanation chamber.
def calculate_expected_upper_limit_in_r_for_expected_background_and_measurement_duration(
    n_bkg_expected,
    t_meas_d,
    confidence_level = 0.90,
    n_samples = 10**4,
    random_seed = None,
):
    rng = np.random.default_rng(seed=random_seed)
    poisson_data = rng.poisson(n_bkg_expected, n_samples)
    upper_limit_data = [
        isotope_dict["rn222"]["decay_constant"]*get_n222rn0_from_detected_214bi_decays(
            N = calculate_upper_limit_for_counting_experiment(cl=confidence_level, n_obs=pd, b=n_bkg_expected),
            tf = t_meas_d*24*60*60,
            ti = 0,
            R = 0,
            n218po0 = 0,
            n214pb0 = 0,
            n214bi0 = 0)
        for pd in poisson_data]
    mean, lower, upper = get_asymmetric_intervals_around_center_val_for_interpolated_discrete_distribution(
        distribution_bin_centers = [],
        distribution_counts = [],
        distribution_data = upper_limit_data,
        distribution_center_value = None,
        interval_width_lower_percent = 0.6827/2,
        interval_width_upper_percent = 0.6827/2,
        flag_verbose = False)

    return mean, lower, upper


# This function is used to calculate the interval widths around a specified center value of a asymmetrical, discrete, and linearly extrapolated distribution.
def get_asymmetric_intervals_around_center_val_for_interpolated_discrete_distribution(
    distribution_bin_centers = [], # bin centers of the input distribution
    distribution_counts = [], # counts of the input distributions
    distribution_data = [], # distribution data, give if bin centers and counts are not yet calculated, overwriteds 'distribution_bin_centers' and 'distribution_counts', NOTE: can only contain integer values!
    distribution_center_value = None, # value around which the intervals are calculated, if None the mean is calculated
    interval_width_lower_percent = 0.6827/2, # percentage of the data contained within the lower interval width output, default of 0.6827 corresponds to one sigma interval of normal distribution
    interval_width_upper_percent = 0.6827/2, # percentage of the data contained within the upper interval width output, default of 0.6827 corresponds to one sigma interval of normal distribution
    ctr_max = 10**6, # maximum number of the safety counter that prevents the program from running in an infinite loop
    granularity = 100, # granularity of the numeric precision with which the intervals are calculated
    flag_verbose = False, # flag indicating whether or not text output is given
):

    # calculating 'distribution_bin_centers' and 'distribution_bin_counts'
    if distribution_data != []:
        if flag_verbose: print(f"gaiacvfidd(): 'distribution_data' given")
        if flag_verbose: print(f"gaiacvfidd(): calculating 'distribution_bin_centers' and 'distribution_bin_counts'")
        distribution_bin_centers = sorted(list(set(distribution_data)))
        distribution_counts = []
        for bin_center in distribution_bin_centers:
            distribution_counts.append(len([val for val in distribution_data if val==bin_center]))
        print(f"now distribution bin centers and counts")
        #print(distribution_bin_centers)
        print(distribution_counts)
    # calculating 'distribution_data'
    else:
        if flag_verbose: print(f"gaiacvfidd(): 'distribution_data' not given")
        if flag_verbose: print(f"gaiacvfidd(): calculating 'distribution_data'")
        distribution_data = []
        for bc, cts in zip(distribution_bin_centers, distribution_counts):
            for i in range(cts):
                distribution_data.append(bc)

    # determining the center value
    center_val = np.mean(distribution_data) if distribution_center_value == None else distribution_center_value
    if flag_verbose:
        print(f"gaiacvfidd(): 'center_val' = {center_val:.4f}")

    # correcting the distribution data for the bin center edges
    leftmost_bin_center = distribution_bin_centers[0] -1.0*(distribution_bin_centers[1] -distribution_bin_centers[0])
    rightmost_bin_center = distribution_bin_centers[-1] +1.0*(distribution_bin_centers[-1] -distribution_bin_centers[-2])
    bin_centers = [leftmost_bin_center] +list(distribution_bin_centers) +[rightmost_bin_center]
    counts = [0] +list(distribution_counts) +[0]

    # defining the linearly interpolated function
    def linearly_interpolated_distribution_function(x):
        f = np.interp(
            x = x,
            xp = bin_centers,
            fp = counts)
        return f

    # calculating the integral over the interpolated function
    norm_val = quad(func=linearly_interpolated_distribution_function, a=bin_centers[0], b=bin_centers[-1])[0]
    if flag_verbose: print(f"gaiacvfidd(): 'norm_val' = {norm_val:.4f}") 

    # calculating the lower interval width
    if flag_verbose: print(f"gaiacvfidd(): determining lower interval width")
    ctr = 0
    interval_width_lower_val = bin_centers[0]
    interval_width_lower_val_neg = center_val
    diff = 1
    # while not not yet precise enough, test 'interval_width_lower_val' values
    while np.sqrt(diff**2) >= 0.00001 and ctr <= ctr_max:
        # loop over 'granularity'-many values between 'interval_width_lower_val' and 'interval_width_lower_val_neg' until the difference 'diff' between 'interval_width_lower_percent' and 'integral' becomes smaller than 'precision'
        # afterwards, set new values for 'interval_width_lower_val' and 'interval_width_lower_val_neg' and evaluate 'diff'
        if flag_verbose: print(f"\tnow testing interval [{interval_width_lower_val:.12f}, {interval_width_lower_val_neg:.12f}]")
        for current_interval_width_lower_val in list(np.linspace(start=interval_width_lower_val, stop=interval_width_lower_val_neg, num=granularity, endpoint=True)):
            integral = quad(
                func = linearly_interpolated_distribution_function,
                a = current_interval_width_lower_val,
                b = center_val,
                limit = 100)[0]
            diff = integral -interval_width_lower_percent*norm_val
            ctr += 1
            if diff >= 0:
                interval_width_lower_val = current_interval_width_lower_val
            else:
                interval_width_lower_val_neg = current_interval_width_lower_val
                break
        if flag_verbose: print(f"\texited 'for' loop with 'ctr' = {ctr}.")
    # setting the lower interval width according to the calculations above
    interval_width_lower = center_val -interval_width_lower_val
    if flag_verbose: print(f"gaiacvfidd(): 'interval_width_lower' = {interval_width_lower:.4f}")

    # calculating the upper interval width
    if flag_verbose: print(f"gaiacvfidd(): determining upper interval width")
    ctr = 0
    interval_width_upper_val = bin_centers[-1]
    interval_width_upper_val_neg = center_val
    diff = 1
    # while not not yet precise enough, test 'interval_width_lower_val' values
    while np.sqrt(diff**2) >= 0.00001 and ctr <= ctr_max:
        # loop over 'granularity'-many values between 'interval_width_lower_val' and 'interval_width_lower_val_neg' until the difference 'diff' between 'interval_width_lower_percent' and 'integral' becomes smaller than 'precision'
        # afterwards, set new values for 'interval_width_lower_val' and 'interval_width_lower_val_neg' and evaluate 'diff'
        if flag_verbose: print(f"\tnow testing interval [{interval_width_upper_val_neg:.8f}, {interval_width_upper_val:.8f}]")
        for current_interval_width_upper_val in list(np.linspace(start=interval_width_upper_val, stop=interval_width_upper_val_neg, num=granularity, endpoint=True)):
            integral = quad(
                func = linearly_interpolated_distribution_function,
                a = center_val,
                b = current_interval_width_upper_val,
                limit = 100)[0]
            diff = integral -interval_width_lower_percent*norm_val
            ctr += 1
            if diff >= 0:
                interval_width_upper_val = current_interval_width_upper_val
            else:
                interval_width_upper_val_neg = current_interval_width_upper_val
                break
        if flag_verbose: print(f"\texited 'for' loop with 'ctr' = {ctr}.")
    # setting the lower interval width according to the calculations above
    interval_width_upper = interval_width_upper_val -center_val
    if flag_verbose: print(f"gaiacvfidd(): 'interval_width_upper' = {interval_width_upper:.4f}")

    # returning
    return center_val, interval_width_lower, interval_width_upper




#######################################
### Post Analysis
#######################################


# This function is used to retrieve measurement data, i.e., output .json files, inquired by the 'input_measurement_id_list'.
def get_measurement_data_for_id_list(input_measurement_id_list):

    # initializing
    print(f"get_measurement_data_for_id_list({input_measurement_id_list}):")
    measurement_dict = {}
    measurement_folder_list = [filename for filename in os.listdir(abspath_measurements) if os.path.isdir(abspath_measurements +filename)]
    for measurement_id in input_measurement_id_list:
        relpath_measurement_folder_list = [directory_name for directory_name in measurement_folder_list if "__" +measurement_id +"__" in directory_name]

        # writing the acquired data to the 'measurement_dict'
        if len(relpath_measurement_folder_list) == 1:
            relpath_measurement_folder = "./" +relpath_measurement_folder_list[0] +"/"
            measurement_dict.update({
                list(relpath_measurement_folder_list[0].split("__"))[1] : {
                    "timestamp" : list(relpath_measurement_folder_list[0].split("__"))[0],
                    "name" : list(relpath_measurement_folder_list[0].split("__"))[2],
                    "data" : get_dict_from_json(abspath_measurements +relpath_measurement_folder +filename_measurement_data_dict)}
                })
            print(f"\tretrieved '{relpath_measurement_folder +filename_measurement_data_dict +'.json'}'")
        else:
            print(f"\tERROR: Could not find data corresponding to measurement id '{measurement_id}' (candidate directories: {relpath_measurement_folder_list}).")

    # printing the data retrieval result
    diff_list = [measurement_id for measurement_id in input_measurement_id_list if measurement_id not in [*measurement_dict]]
    if len(diff_list) != 0:
        raise Exception(f"\tERROR Could not retrieve the requested data for the following measurement ids: {diff_list}.")
    else:
        print(f"\tSuccessfully retrieved the requested data.")
                  
    # returning the 'measurement_dict' filled with the requested data.
    return measurement_dict

