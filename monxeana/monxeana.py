
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
import getpass
import json
from scipy.stats import chi2 # for "Fabian's calculation" of the Poissonian Error
import scipy.integrate as integrate
import subprocess

# including the 'monxe_software' libraries 'monxeana' and 'miscfig'
import sys
pathstring_miscfig = "/home/daniel/Desktop/arbeitsstuff/20180705__monxe/monxe_software/miscfig/"
sys.path.append(pathstring_miscfig)
import Miscellaneous_Figures as miscfig




#######################################
### Generic Definitions
#######################################


# username
username = getpass.getuser()


# paths
if username == "daniel":
    abspath_monxe = "/home/daniel/Desktop/arbeitsstuff/20180705__monxe/"
elif username == "monxe":
    abspath_monxe = "/home/monxe/Desktop/"
else:
    abspath_monxe = "./"
abspath_measurements = abspath_monxe +"monxe_measurements/"
relpath_data_compass = "./data/DAQ/run/RAW/" # this is the folder where CoMPASS stores the measurement data


# filenames
filename_data_csv = "DataR_CH0@DT5781A_840_run.csv" # timestamp (and eventually waveform) data
filename_data_txt = "CH0@DT5781A_840_EspectrumR_run.txt" # adc spectrum data
filename_histogram_png = "histogram" # histogram plot name
filename_measurement_data_dict = "measurement_data"
filename_raw_data_ndarray = "raw_data.npy"
filename_raw_data_png = "raw_data.png"
filename_peak_data_png = "peak_data.png"
filename_activity_data_png = "activity_data.png"



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
        "color" : color_monxe_cyan,
        "latex_label" : r"$^{222}\,\mathrm{Rn}$",
    },
    # polonium
    "po218" : {
        "isotope" : "po218",
        "half_life_s" : 3.071 *60, # 3.071 min
        "q_value_kev" : 6114.68,
        "alpha_energy_kev" : 6002.35,
        "decay_constant" : np.log(2)/(3.071 *60),
        "color" : "#6700ad", # 1/5 in color gradient (https://colordesigner.io/gradient-generator) from "#6700ad" (purple) to "d11166" (wine red)
        "latex_label" : r"$^{218}\,\mathrm{Po}$",
    },
    "po216" : {
        "isotope" : "po216",
        "half_life_S" : 0.148,
        "decay_constant" : np.log(2)/(0.148),
        "q_value_kev" : 6906.3,
        "alpha_energy_kev" : 6778.4,
        "color" : "#93009b", # 2/5 in color gradient (https://colordesigner.io/gradient-generator) from "#6700ad" (purple) to "d11166" (wine red)
        "latex_label" : r"$^{216}\,\mathrm{Po}$",
    },
    "po214" : {
        "isotope" : "po214",
        "half_life_s" : 162.3 *10**(-6), # 162.3 µs
        "decay_constant" : np.log(2)/(162.3 *10**(-6)),
        "q_value_kev" : 7833.46,
        "alpha_energy_kev" : 7686.82,
        "color" : "#b10088", # 3/5 in color gradient (https://colordesigner.io/gradient-generator) from "#6700ad" (purple) to "d11166" (wine red)
        "latex_label" : r"$^{214}\,\mathrm{Po}$",
    },
    "po212" : {
        "isotope" : "po212",
        "half_life_s" : 300 *10**(-9), # 300 nanoseconds
        "q_value_kev" : 8954.12,
        "alpha_energy_kev" : 8785.17,
        "decay_constant" : np.log(2)/(300 *10**(-9)),
        "color" : '#c40076', # 4/5 in color gradient (https://colordesigner.io/gradient-generator) from "#6700ad" (purple) to "d11166" (wine red)
        "latex_label" : r"$^{212}\,\mathrm{Po}$",
    },
    "po210" : {
        "isotope" : "po210",
        "half_life_s" : 138.3763 *24 *60 *60, # 138 days in seconds
        "q_value_kev" : 5407.45,
        "alpha_energy_kev" : 5304.33,
        "decay_constant" : np.log(2)/(138.3763 *24 *60 *60),
        "color" : '#d11166', # 5/5 in color gradient (https://colordesigner.io/gradient-generator) from "#6700ad" (purple) to "d11166" (wine red)
        "latex_label" : r"$^{226}\,\mathrm{Ra}$",
    },
    # lead
    "pb214" : {
        "isotope" : "pb214",
        "half_life_s" : 26.916 *60, # 26.916 min
        "decay_constant" : np.log(2)/(26.916 *60),
        "color" : "#ff1100", # red
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
### Generic Functions
#######################################


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


# This function is used to convert a datetime string (as defined by datetime, e.g. '31-07-20 15:31:25') into a datetime object.
def convert_string_to_datetime_object(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%d/%m/%y %H:%M:%S')
    return datetime_obj
#def convert_string_to_datetime_object(datetime_str):
#    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M')
#    return datetime_obj
#convert_string_to_datetime_object(datetime_str="20200731_1530")


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
    measurement_data_dict,
    update_dict,
    measurement_data_output_pathstring
):

    # updating and saving the 'measurement_data' dict
    measurement_data_dict.update(update_dict)
    write_dict_to_json(measurement_data_output_pathstring,measurement_data_dict)
    
    # printing the 'update_dict' data
    print(f"'update_and_save_measurement_data()':")
    print(json.dumps(update_dict, indent=4, sort_keys=True))
    
    # returning None
    return




#######################################
### Raw Data
#######################################


# This is the dtype used for raw data extracted from CoMPASS.
raw_data_dtype = np.dtype([
    ("timestamp_ps", np.uint64), # timestamp in ps
    ("pulse_height_adc", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("flag_mca", np.unicode_, 16), # flag extracted from the mca list file
])


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


# This function is used to load the raw data from the MCA list file generated either by MC2Analyzer or CoMPASS.
def get_raw_data_from_list_file(
    pathstring_input_data, # pathstring at which the MCA list file can be found
    pathstring_output, # pathstring according to which the extracted data should be saved as a ndarray
    flag_ctr = 10**10, # counter determining the number of processed events
    flag_daq = ["mc2analyzer", "compass_auto", "compass_custom"][1], # flag indicating the method of data extraction
    flag_debug = ["no_debugging", "debug", "debug_storewfms"][1]): # flag indicating the debugging method

    # automatically retrieving the data if split into multiple files
    t_i = datetime.datetime.now()
    print(f"get_raw_data_from_list_file(): searching for\n '{pathstring_input_data}'\n")
    if flag_daq in ["compass_auto", "compass_custom"]:
        path_searchfolder = "/".join(pathstring_input_data.split("/")[:-1]) +"/" # this is the directory of the input list file
        measurement_files_pathstrings = [path_searchfolder +filename for filename in os.listdir(path_searchfolder) if filename_data_csv[:-4] in filename]
    elif flag_daq == "mc2analyzer":
        measurement_files_pathstrings = [pathstring_input_data]
    else:     # catching undefined 'flag_daq' values
        raise Exception(f"get_raw_data_from_list_file(): wrong input 'flag_daq': {flag_daq}")
    print(f"get_raw_data_from_list_file(): found the following {len(measurement_files_pathstrings)} raw data files:")
    for i in range(len(measurement_files_pathstrings)):
        print("\t - " +measurement_files_pathstrings[i].split("/")[-1])
    print("")

    # looping over the list files and writing the data into 'timestamp_tuple_list'
    print(f"get_raw_data_from_list_file(): starting data retrieval")
    timestamp_data_tuplelist = [] # this list will later be cast into a structured array
    ctr = 0 # counter tracking the number of processed waveforms
    if flag_debug == "debug_storewfms":
        subprocess.call("rm -r " +relpath_rawdatadebugging +"*", shell=True)
    for pathstring_measurement_data in measurement_files_pathstrings:
        with open(pathstring_measurement_data) as input_file:

            # retrieving raw data from CoMPASS list file
            if flag_daq in ["compass_auto", "compass_custom"]:
                for line in input_file:
                    if line.startswith("BOA"):
                        continue
                    elif ctr <= flag_ctr:
                    #elif not line.startswith("BOA"):
                        line_list = list(line.split(";"))
                        board = np.uint64(line_list[0]) # irrelevant
                        channel = np.uint64(line_list[1]) # irrelevant
                        timestamp_ps = np.uint64(line_list[2]) # timestamp in picoseconds
                        pulse_height_adc = np.uint64(line_list[3]) # pulse height in adc determined via trapezoidal filter
                        flag_mca = line_list[4] # information flag provided by CoMPASS
                        wfm_data_ndarray = np.array([(int(i)) for i in line_list[5:]], np.dtype([("wfm_data_adc", np.int16)])) # waveform data in adc channels
                        t_ns = [i for i in range(len(wfm_data_ndarray["wfm_data_adc"]))] # sampling times in 10ns (corresponding to MCA sampling rate of 100MS/s)
                        timestamp_data_tuple = (timestamp_ps, pulse_height_adc, flag_mca) # exported data corresponding to 'raw_data_dtype'

                        # determining the energy with a custom PHA instead of utilizing the CoMPASS trapezoidal filter algorithm
                        if flag_daq == "compass_custom":
                            try:
                                p0_i = [ # this initial guess is empirically tuned to amp_v5.1a and amp_v5.1b utilizing CoMPASS settings
                                    np.mean(wfm_data[:1000]), # baseline
                                    1750, # x_rise
                                    1.05*(np.amax(wfm_data)-np.mean(wfm_data[:20])), # a_rise
                                    0.011, # lambda_rise
                                    1750, # x_decay
                                    1.16*(np.amax(wfm_data)-np.amin(wfm_data)), # a_deca
                                    0.00015] # lambda_decay
                                popt, pcov = curve_fit(
                                    f = fitfunction__independent_exponential_rise_and_fall__vec,
                                    xdata = t_ns[:15000],
                                    ydata = wfm_data[:15000],
                                    p0 = p0_i,
                                    sigma = None,
                                    absolute_sigma = False,
                                    method = [None, "lm", "trf", "dogbox"][0],
                                    maxfev = 50000)
                                perr = np.sqrt(np.diag(pcov))
                                fitvals = fitfunction__independent_exponential_rise_and_fall__vec(
                                    x = t_ns,
                                    y_baseline = popt[0],
                                    x_rise = popt[1],
                                    a_rise = popt[2],
                                    lambda_rise = popt[3],
                                    x_decay = popt[4],
                                    a_decay = popt[5],
                                    lambda_decay = popt[6])
                                pulse_height_adc__compass = np.uint64(line_list[3])
                                pulse_height_adc__fitheight = np.amax(fitvals) -popt[0]
                                pulse_height_adc__fitasymptote = popt[2]
                                flag_pha = "fit_successful"
                                print(f"fitted wfm No.: {ctr}")
                            except:
                                pulse_height_adc__compass = pulse_height_adc,
                                pulse_height_adc__fitheight = pulse_height_adc,
                                pulse_height_adc__fitasymptote = pulse_height_adc,
                                flag_pha = "fit_failed"
                            timestamp_data_tuple = (
                                timestamp_ps,
                                pulse_height_adc__compass,
                                pulse_height_adc__fitheight,
                                pulse_height_adc__fitasymptote,
                                flag_mca,
                                flag_pha)

                        # debugging
                        if flag_debug != "no_debugging":
                            print(f"\n\n\n###############################################\n\n\n")
                            print(f"wfm No.: {ctr}")
                            fig, ax1 = plt.subplots(figsize=(5.670, 3.189), dpi=110, constrained_layout=True)
                            plt.plot(
                                t_ns,
                                wfm_data,
                                color = "black")
                            ax1.set_xlabel(r"Time / $10\,\mathrm{ns}$")
                            ax1.set_ylabel(r"Voltage / $\mathrm{adc\,\,channels}$")
                            ax1.set_xlim(left=0, right=max(t_ns))
                            if flag_pha == "fit_successful":
                                print("parameters found:")
                                print(f"\t\tpopt[0] = {popt[0]} whereas p0(baseline) = {p0_i[0]}")
                                print(f"\t\tpopt[1] = {popt[1]} whereas p0(x_rise) = {p0_i[1]}")
                                print(f"\t\tpopt[2] = {popt[2]} whereas p0(a_rise) = {p0_i[2]}")
                                print(f"\t\tpopt[3] = {popt[3]} whereas p0(lambda_rise) = {p0_i[3]}")
                                print(f"\t\tpopt[4] = {popt[4]} whereas p0(x_decay) = {p0_i[4]}")
                                print(f"\t\tpopt[5] = {popt[5]} whereas p0(a_decay) = {p0_i[5]}")
                                print(f"\t\tpopt[6] = {popt[6]} whereas p0(lambda_decay) = {p0_i[6]}")
                                plt.plot(
                                    t_ns,
                                    fitfunction__independent_exponential_rise_and_fall__vec(
                                        x = t_ns,
                                        y_baseline = popt[0],
                                        x_rise = popt[1],
                                        a_rise = popt[2],
                                        lambda_rise = popt[3],
                                        x_decay = popt[4],
                                        a_decay = popt[5],
                                        lambda_decay = popt[6]),
                                    color = color_monxe_cyan)
                            elif flag_pha == "fit_failed":
                                print("fit failed")
                            else:
                                print(f"something strange happened: 'flag_pha'={flag_pha}")
                            plt.show()
                            if flag_debug == "debug_storewfms":
                                np.save(relpath_rawdatadebugging +"wfm_" +str(ctr) +".npy", wfm_data_ndarray)
                                fig.savefig(relpath_rawdatadebugging +"wfm_" +str(ctr) +".png")

                        # filling the individual wfm data into the 'timestamp_data_tuplelist'
                        timestamp_data_tuplelist.append(timestamp_data_tuple)
                        if ctr%1000==0:
                            print(f"\t\t{ctr} events processed")
                        ctr += 1

            # retrieving raw data from MC2Analyzer list file
            elif flag_daq == "mc2analyzer":
                for line in input_file:
                    line_list = list(line.split())
                    if not line.startswith("HEADER") and ctr < flag_ctr:
                        timestamp_ps = 10000*np.uint64(line_list[0]) # the MCA stores timestamps in clock cycle units (one clock cycle corresponds to 10ns, 10ns = 10000ps)
                        pulse_height_adc = np.int64(line_list[1])
                        extra = np.int64(line_list[2])
                        timestamp_data_tuplelist.append((
                            timestamp_ps,
                            pulse_height_adc,
                            extra))
                        ctr += 1
                    elif line.startswith("HEADER") and ctr < flag_ctr:
                        print(f"\theader line: {line[:-1]}")
                    elif ctr < flag_ctr:
                        print(f"\tunprocessed line: {line}")
                    else:
                        continue

    # storing the extracted data in a numpy structured array
    #retarray = np.sort(np.array(timestamp_data_tuplelist, raw_data_dtype), order="timestamp_ps")
    retarray = np.sort(
        np.array(
            timestamp_data_tuplelist,
            raw_data_dtype if flag_daq != "compass_custom" else raw_data_dtype_custom_pha),
        order = "timestamp_ps")
    if pathstring_output != "none":
        np.save(pathstring_output, retarray)
    t_f = datetime.datetime.now()
    t_run = t_f -t_i
    print(f"\nget_raw_data_from_list_file(): retrieval time: {t_run} h")

    # returning the raw data array
    return retarray


# This function is used to print miscellaneous informations regarding the raw data retrieved from a MCA measurement
def print_misc_meas_information(meas_ndarray):

    print(f"\nprint_misc_meas_information(): measurement information:")

    # general information
    print(f"\tmeasurement duration: {get_measurement_duration(list_file_data=meas_ndarray, flag_unit='days'):.3f} days")
    print(f"\trecorded events: {len(meas_ndarray)}")

    # event groups
    print(f"\t\tthereof in ch0: {len(meas_ndarray[(meas_ndarray['pulse_height_adc'] == 0)])}")
    print(f"\t\tthereof within the first 50 adc channels: {len(meas_ndarray[(meas_ndarray['pulse_height_adc']<50)])}")
    print(f"\t\tthereof in negative: {len(meas_ndarray[(meas_ndarray['pulse_height_adc'] < 0)])}")
    print(f"\t\tthereof above max adc channel ({adc_channel_max}): {len(meas_ndarray[(meas_ndarray['pulse_height_adc'] > adc_channel_max)])}")
    # fit (only for CoMPASS DAQ)
    if "flag_daq" in meas_ndarray.dtype.names:
        print(f"\t\tthereof correctly fitted: {len(meas_ndarray[(meas_ndarray['flag_pha'] == 'fit_successful')])}")
        print(f"\t\tthereof not fitted: {len(meas_ndarray[(meas_ndarray['flag_pha'] == 'fit_failed')])}")
    # mca flags
    mca_flag_list = []
    for i in range(len(meas_ndarray)):
        if meas_ndarray[i]["flag_mca"] not in mca_flag_list:
            mca_flag_list.append(meas_ndarray[i]["flag_mca"])
    for i in range(len(mca_flag_list)):
        print(f"\t\tthereof with mca_flag '{mca_flag_list[i]}': {len(meas_ndarray[(meas_ndarray['flag_mca'] == mca_flag_list[i])])}")

    # return
    retlist = [
        f"entries: {len(meas_ndarray)} (within first 50 adc channels: {len(meas_ndarray[(meas_ndarray['pulse_height_adc']<50)])})",
        f"measurement duration: {get_measurement_duration(list_file_data=meas_ndarray, flag_unit='days'):.1f}" +r"$\,\mathrm{d}$"]
    return retlist


# This function is used to 
def get_raw_data_dict(raw_data):

    # initializing the 'misc_meas_information_dict'
    raw_data_dict = {
        "measurement_duration_days" : get_measurement_duration(list_file_data=raw_data, flag_unit='days'),
        "recorded_events" : {
            "total" : len(raw_data),
            "thereof_in_ch0" : len(raw_data[(raw_data['pulse_height_adc'] == 0)]),
            "thereof_in_first_50_adcc" : len(raw_data[(raw_data['pulse_height_adc']<50)]),
            "thereof_in_negative_adcc" : len(raw_data[(raw_data['pulse_height_adc'] < 0)]),
            "thereof_above_max_adcc" : len(raw_data[(raw_data['pulse_height_adc'] > adc_channel_max)]),
        },
        "mca_flags" : {},
    }

    # adding mca flags
    mca_flag_list = []
    for i in range(len(raw_data)):
        if raw_data[i]["flag_mca"] not in mca_flag_list:
            mca_flag_list.append(raw_data[i]["flag_mca"])
    for i in range(len(mca_flag_list)):
        raw_data_dict["mca_flags"].update({mca_flag_list[i] : len(raw_data[(raw_data['flag_mca'] == mca_flag_list[i])])})

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


# This function is used to plot the raw energy spectrum.
def plot_raw_data(
    measurement_data_dict,
    raw_data_ndarray,
    plot_settings_dict = {
        "output_pathstrings" : [],
        "plot_format" : "16_9",
        "x_lim" : [0, n_adc_channels],
        "x_label" : "alpha energy / adc channels",
        "y_label" : "entries per channel"
    }
):

    # generating histogram data from the 
    data_histogram = get_histogram_data_from_timestamp_data(timestamp_data=raw_data_ndarray, histval="pulse_height_adc")

    # figure formatting
    fig, ax1 = plt.subplots(figsize=miscfig.image_format_dict[plot_settings_dict["plot_format"]]["figsize"], dpi=150)
    xlim = plot_settings_dict["x_lim"]
    ylim = [0,1.1*(max(data_histogram["counts"][(data_histogram["bin_centers"]>=xlim[0]) & (data_histogram["bin_centers"]<=xlim[1])] ))]
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    #ax1.set_yscale('log')
    ax1.set_xlabel(plot_settings_dict["x_label"])
    ax1.set_ylabel(plot_settings_dict["y_label"])

    # plotting stepized histogram data
    bin_centers, counts, counts_errors_lower, counts_errors_upper, bin_centers_mod, counts_mod = stepize_histogram_data(
        bincenters = data_histogram["bin_centers"],
        counts = data_histogram["counts"],
        counts_errors_lower = data_histogram["counts_errors_lower"],
        counts_errors_upper = data_histogram["counts_errors_upper"],
        flag_addfirstandlaststep = True)
    plt.plot(
        bin_centers_mod,
        counts_mod,
        linewidth = 0.4,
        color=color_histogram,
        linestyle='-',
        zorder=1,
        label="fdfa")

    # annotations
    #monxeana.annotate_documentation_json(
    #    annotate_ax=ax1,
    #    filestring_documentation_json = "./documentation.json",
    #    text_fontsize = 11,
    #    text_color = "black",
    #    text_x_i = 0.03,
    #    text_y_i = 0.75,
    #    text_parskip = 0.09,
    #    #flag_keys = ["amplifier", "start", "end"],
    #    flag_addduration=True,
    #    flag_print_comment=False,
    #    flag_orientation="left"
    #)
    #plt.legend()

    # saving the output plot
    plt.show()
    for i in range(len(plot_settings_dict["output_pathstrings"])):
        fig.savefig(plot_settings_dict["output_pathstrings"][i])

    return fig, ax1





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
def stepize_histogram_data(bincenters, counts, counts_errors_lower, counts_errors_upper, flag_addfirstandlaststep=True):
    # calculating the binwidth and initializing the lists
    binwidth = bincenters[1]-bincenters[0]
    bincenters_stepized = np.zeros(2*len(bincenters))
    counts_stepized = np.zeros(2*len(counts))
    counts_errors_lower_stepized = np.zeros(2*len(counts))
    counts_errors_upper_stepized = np.zeros(2*len(counts))
    # stepizing the data
    for i in range(len(bincenters)):
        bincenters_stepized[2*i] = bincenters[i] -0.5*binwidth
        bincenters_stepized[(2*i)+1] = bincenters[i] +0.5*binwidth
        counts_stepized[2*i] = counts[i]
        counts_stepized[2*i+1] = counts[i]
        counts_errors_lower_stepized[2*i] = counts_errors_lower[i]
        counts_errors_lower_stepized[2*i+1] = counts_errors_lower[i]
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



# one possible energy-channel relation: linear
def energy_channel_relation_function_linear(x, m, t):
    if hasattr(x, "__len__"):
        return [m*xi +t for xi in x]
    else:
        return m*x +t


# This function is used to calculate the energy-channel relation coefficients for a given peak_data_dict.
def energy_calibration(peak_data_dict, energy_channel_relation_function):

    # extracting the (adcc,energy) data points
    adcc_list = []
    adcc_error_list = []
    e_mev_list = []
    for peaknum in [peaknum for peaknum in [*peak_data_dict] if "a_priori" in [*peak_data_dict[peaknum]]]:
        adcc_list.append(peak_data_dict[peaknum]["fit_data"]["mu"])
        adcc_error_list.append(peak_data_dict[peaknum]["fit_data_errors"]["mu"])
        e_mev_list.append(peak_data_dict[peaknum]["a_priori"]["alpha_energy_kev"]/1000)

    # curve_fit output: 
    p_opt, p_cov = curve_fit(
        f = energy_channel_relation_function,
        xdata = adcc_list,
        ydata = e_mev_list,
        #sigma = fit_data["counts_errors"],
        #absolute_sigma = True,
        method = 'lm' # "lm" cannot handle covariance matrices with deficient rank
    )
    
    # extracting the 'linear_function' coefficients
    p_err = np.sqrt(np.diag(p_cov))

    # returning the determined optimal parameters for the energy-channel relation
    return p_opt, p_err


# This function is used to extract the 'peak_data_dict' from 'raw_data' and 'measurement_data'.
def get_peak_data_dict(
    measurement_data_dict,
    raw_data_ndarray,
    peak_fit_settings_dict = {
        "fit_range" : [8000, 13000],
        "rebin" : 1,
        "p_opt_guess" : ( # n x 5-tuple, whereas n corresponds to the number of peaks visible in the spectrum (fit parameters: mu, sigma, alpha, n, N)
            8600, 25, 1, 1, 100, # po210
            9500, 25, 1, 1, 100, # po218
            12400, 25, 1, 1, 100 # po214
        ),
        "a_priori_knowledge" : {
            "1" : "po218",
            "2" : "po214"
        },
    },
):

    # manual input
    flag_axis = ["adc_channels", "energy_calibrated"][1]
    rebin = peak_fit_settings_dict["rebin"]
    fitrange = peak_fit_settings_dict["fit_range"]
    p_opt_guess = peak_fit_settings_dict["p_opt_guess"]

    # generating rebinned histogram data
    data_histogram_rebinned = get_histogram_data_from_timestamp_data(
        timestamp_data = raw_data_ndarray,
        number_of_bins = int(n_adc_channels/rebin))

    # fitting the peaks
    peak_data_dict = fit_range_mult_crystal_ball(
        n = int(len(p_opt_guess)/5),
        histogram_data = data_histogram_rebinned,
        fit_range = fitrange,
        p0 = p_opt_guess)

    # including the 'peak_fit_settings' into the 'peak_data_dict'
    peak_data_dict.update({"peak_fit_settings" : peak_fit_settings_dict})
    
    # including the 'a_priori_knowledge_dict' into the 'peak_data_dict'
    for i in [*peak_fit_settings_dict["a_priori_knowledge"]]:
        peak_data_dict[i].update({"a_priori" : isotope_dict[peak_fit_settings_dict["a_priori_knowledge"][i]]})

    # peak data calculations (i.e. determine resolution, number of counts, etc...)
    #calc_peak_data(
    #    peak_data_dictionary = peak_data_dict,
    #    timestamp_data_ndarray = data_raw,
    #    n_sigma_left = 10,
    #    n_sigma_right = 4)
    
    return peak_data_dict


# This function is used to plot the peak data.
def plot_peak_data(
    measurement_data_dict,
    raw_data_ndarray,
    plot_settings_dict = {
        "output_pathstrings" : [],
        "plot_format" : "16_9",
        "x_lim_adcc" : [0, n_adc_channels],
        "rebin" : 1,
        "flag_x_axis_units" : ["adc_channels", "mev"][1],
        "energy_channel_relation" : [energy_channel_relation_function_linear][0],
        "flag_errors" : ["poissonian"][0],
        "flag_plot_errors" : [False, True][0],
        "flag_plot_fits" : ["none", "all", "known", "po218+po214"][3],
        "flag_preliminary" : [True, False][0],
    }
):

    # generating rebinned histogram data
    data_histogram_rebinned = get_histogram_data_from_timestamp_data(
        timestamp_data = raw_data_ndarray,
        number_of_bins = int(n_adc_channels/plot_settings_dict["rebin"]),
        flag_errors = plot_settings_dict["flag_errors"])

    # figure formatting
    fig, ax1 = plt.subplots(figsize=miscfig.image_format_dict[plot_settings_dict["plot_format"]]["figsize"], dpi=150)
    y_lim = [0, 1.1*(max(data_histogram_rebinned["counts"]) +max(data_histogram_rebinned["counts_errors_upper"]))]
    y_lim = [0, 1.1*(max(data_histogram_rebinned["counts"]))]
    ax1.set_ylim(y_lim)
    x_lim = plot_settings_dict["x_lim_adcc"]
    x_width = x_lim[1] -x_lim[0]
    y_width = y_lim[1] -y_lim[0]
    #ax1.set_yscale('log')
    binwidth_adcc = float(data_histogram_rebinned['bin_centers'][2]-data_histogram_rebinned['bin_centers'][1])
    if plot_settings_dict["flag_x_axis_units"] == "adc_channels":
        ax1.set_xlabel(r"alpha energy / $\mathrm{adc\,channels}$")
        ax1.set_ylabel(r"entries per " +f"{binwidth_adcc:.1f}" +r" adc channels")
        ax1.set_xlim(x_lim)
    elif plot_settings_dict["flag_x_axis_units"] == "mev":
        p_opt, p_err = energy_calibration(peak_data_dict=measurement_data_dict["peak_data"], energy_channel_relation_function=plot_settings_dict["energy_channel_relation"])
        binwidth_mev = np.sqrt((plot_settings_dict["energy_channel_relation"](1, *p_opt)-plot_settings_dict["energy_channel_relation"](0, *p_opt))**2)
        ax1.set_xlabel(r"alpha energy / $\mathrm{MeV}$")
        ax1.set_ylabel(r"entries per " +f"{binwidth_mev*1000:.2f}" +r" $\mathrm{keV}$")
        ax1.set_xlim([plot_settings_dict["energy_channel_relation"](x_lim[0], *p_opt), plot_settings_dict["energy_channel_relation"](x_lim[1], *p_opt)])

    # plotting the stepized histogram
    bin_centers, counts, counts_errors_lower, counts_errors_upper, bin_centers_mod, counts_mod = stepize_histogram_data(
        bincenters = data_histogram_rebinned["bin_centers"],
        counts = data_histogram_rebinned["counts"],
        counts_errors_lower = data_histogram_rebinned["counts_errors_lower"],
        counts_errors_upper = data_histogram_rebinned["counts_errors_upper"],
        flag_addfirstandlaststep = True)
    plt.plot(
        bin_centers_mod if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else [plot_settings_dict["energy_channel_relation"](xi, *p_opt) for xi in bin_centers_mod],
        counts_mod,
        linewidth=linewidth_histogram_std,
        color=color_histogram,
        linestyle='-',
        zorder=1,
        label="fdfa")

    # plotting the Poissonian errors
    plt.fill_between(
        bin_centers if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else [plot_settings_dict["energy_channel_relation"](xi, *p_opt) for xi in bin_centers],
        counts-counts_errors_lower,
        counts+counts_errors_upper,
        color = color_histogram_error,
        alpha = 1,
        zorder = 0,
        interpolate = True)

    ### plotting the fits
    if plot_settings_dict["flag_plot_fits"] != "none":
        # determining what peak fits to plot
        fit_data_x_vals = np.linspace(start=x_lim[0], stop=x_lim[1], endpoint=True, num=500)
        all_peaks = [*measurement_data_dict["peak_data"]]
        known_peaks = [peaknum for peaknum in all_peaks if "a_priori" in [*measurement_data_dict["peak_data"][peaknum]]]
        if plot_settings_dict["flag_plot_fits"] == "all":
            fitlist = all_peaks
        elif plot_settings_dict["flag_plot_fits"] == "known":
            fitlist = known_peaks
        elif plot_settings_dict["flag_plot_fits"] == "po218+po214":
            fitlist = [peaknum for peaknum in known_peaks if measurement_data_dict["peak_data"][peaknum]["a_priori"]["isotope"] in ["po218", "po214"]]
        # plotting the selected peaks
        for peaknum in fitlist:
            plt.plot(
                fit_data_x_vals if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else [plot_settings_dict["energy_channel_relation"](xi, *p_opt) for xi in fit_data_x_vals],
                [function_crystal_ball_one(adcc, **measurement_data_dict["peak_data"][peaknum]["fit_data"]) for adcc in fit_data_x_vals],
                linewidth = 0.5,
                color = isotope_dict[measurement_data_dict["peak_data"][peaknum]["a_priori"]["isotope"]]["color"] if "a_priori" in [*measurement_data_dict["peak_data"][peaknum]] else color_monxe_cyan,
                linestyle = '-',
                zorder = 1,
                label = "fit") #r"$" +peak_data_dict[key]["isotope_data"]["latex_label"] +r"$")


    ### annotations
    # annotationg the MonXe logo
    #miscfig.image_onto_plot(
    #    filestring = "monxe_logo__transparent_bkg.png",
    #    ax=ax1,
    #    position=(x_lim[0]+0.90*(x_lim[1]-x_lim[0]),y_lim[0]+0.87*(y_lim[1]-y_lim[0])),
    #    pathstring = pathstring_miscellaneous_figures +"monxe_logo/",
    #    zoom=0.02,
    #    zorder=0)
    # peak labels
    #for key in [k for k in list(peak_data_dict.keys()) if "isotope_data" in peak_data_dict[k]]:
    #    plt.text(
    #        x = peak_data_dict[key]["fit_data"]["mu"] -0.02*x_width if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](peak_data_dict[key]["fit_data"]["mu"] -0.02*x_width, *p_opt),
    #        y = 1.02*monxeana.function_crystal_ball_one(x=peak_data_dict[key]["fit_data"]["mu"], **peak_data_dict[key]["fit_data"]),
    #        #transform = ax1.transAxes,
    #        s = r"$" +peak_data_dict[key]["isotope_data"]["latex_label"] +r"$",
    #        color = "black",
    #        fontsize = 11,
    #        verticalalignment = 'center',
    #        horizontalalignment = 'right')
    # extracted counts
    #adc_selection_window_left = 1000
    #adc_selection_window_right = 250
    #for key in [k for k in list(peak_data_dict.keys()) if "isotope_data" in peak_data_dict[k]]:
    #    if peak_data_dict[key]["isotope_data"]["label"] in ["po218", "po214"]:
    #        left_border_adcc = peak_data_dict[key]["fit_data"]["mu"]-adc_selection_window_left
    #        right_border_adcc = peak_data_dict[key]["fit_data"]["mu"]+adc_selection_window_right
    #        ax1.axvspan(
    #            left_border_adcc if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](left_border_adcc, *p_opt),
    #            right_border_adcc if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](right_border_adcc, *p_opt),
    #            alpha = 0.5,
    #            linewidth = 0,
    #            color = monxeana.isotope_dict[peak_data_dict[key]["isotope_data"]["label"]]["color"],
    #            zorder = -50)
    # shading the fit region
    #ax1.axvspan(
    #    fitrange[0] if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](fitrange[0], *p_opt),
    #    fitrange[1] if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](fitrange[1], *p_opt),
    #    alpha = 0.5,
    #    linewidth = 0,
    #    color = 'grey',
    #    zorder = -50)
    #for key in ["3", "4"]:
    #    ax1.axvspan(
    #        peak_data_dict[key]["counts"]["left_border_adc"] if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](, *p_opt),
    #        peak_data_dict[key]["counts"]["right_border_adc"] if plot_settings_dict["flag_x_axis_units"]=="adc_channels" else plot_settings_dict["energy_channel_relation"](, *p_opt),
    #        alpha = 0.5,
    #        linewidth = 0,
    #        color = "purple" if key=="3" else "green",
    #        zorder = -50)
    # measurement comments
    #monxeana.annotate_comments(
    #    comment_ax = ax1,
    #    comment_list = [
    #        r"calibration activity: $(19.6\pm 2.1)\,\mathrm{mBq}$",
    #        r"entries total: " +f"{len(data_raw)}",
    #        r"measurement duration: " +f"{monxeana.get_measurement_duration(list_file_data=data_raw, flag_unit='days'):.3f} days",
    #    ],
    #    comment_textpos = [0.025, 0.9],
    #    comment_textcolor = "black",
    #    comment_linesep = 0.1,
    #    comment_fontsize = 11)
    # annotating 
    #plt.text(
    #    x = peak_data_dict["2"]["fit_data"]["mu"] -0.02*x_width,
    #    y = 1.02*monxeana.function_crystal_ball_one(x=peak_data_dict["2"]["fit_data"]["mu"], **peak_data_dict["2"]["fit_data"]) -0.06*y_width,
    #    #transform = ax1.transAxes,
    #    s = r"$R=" +f"{peak_data_dict['2']['resolution']['resolution_in_percent']:.1f}" +r"\,\%$",
    #    color = "black",
    #    fontsize = 11,
    #    verticalalignment = 'center',
    #    horizontalalignment = 'right')
    # marking as 'preliminary'
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

    # saving the output plot
    plt.show()
    for i in range(len(plot_settings_dict["output_pathstrings"])):
        fig.savefig(plot_settings_dict["output_pathstrings"][i])

    return fig, ax1





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
            verticalalignment = "center",
            transform = comment_ax.transAxes
        )
        ctr_textpos += 1

    return


# This function returns an analytical expression for the activity of Rn222 nuclei (retrieved from the linear Rn222 decay chain model with additional emanation source term).
def rn222(
    t, # time in s
    n222rn0, # number of initial rn222 nuclei at t_i
    r): # 222rn emanation rate in Bq
    
    if t<= 0:
        return 0
    else:
        return isotope_dict["rn222"]["decay_constant"] *((isotope_dict["rn222"]["decay_constant"]*n222rn0 + (-1 + np.exp(isotope_dict["rn222"]["decay_constant"]*t))*r)/(np.exp(isotope_dict["rn222"]["decay_constant"]*t)*isotope_dict["rn222"]["decay_constant"]))


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
        return isotope_dict["bi214"]["decay_constant"] *((isotope_dict["bi214"]["decay_constant"]**4*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*n214bi0 + (-1 + np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)))*isotope_dict["pb214"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*r + isotope_dict["bi214"]["decay_constant"]**2*((-isotope_dict["pb214"]["decay_constant"]**2)*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["rn222"]["decay_constant"]**2*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + isotope_dict["po218"]["decay_constant"]**2*(n214bi0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*(n214pb0 + n218po0)) + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(n214bi0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*(2*n214pb0 + n218po0))) + isotope_dict["po218"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*n214bi0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]*r + (-np.exp(isotope_dict["bi214"]["decay_constant"]*t) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*r) + isotope_dict["pb214"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*isotope_dict["rn222"]["decay_constant"]**2*n218po0 + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**3*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 + (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) - (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**3*r + isotope_dict["po218"]["decay_constant"]**3*(isotope_dict["rn222"]["decay_constant"]*(n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n222rn0 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) + (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r)) + isotope_dict["pb214"]["decay_constant"]**3*(isotope_dict["po218"]["decay_constant"]**2*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0) - isotope_dict["rn222"]["decay_constant"]*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*r) + isotope_dict["po218"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n218po0 + (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n222rn0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r))) + isotope_dict["bi214"]["decay_constant"]*((-isotope_dict["pb214"]["decay_constant"])*isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]**2*(n214bi0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*(n214pb0 + n218po0 + n222rn0)) + (np.exp(isotope_dict["bi214"]["decay_constant"]*t) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*isotope_dict["rn222"]["decay_constant"]**2*r + isotope_dict["pb214"]["decay_constant"]**2*((np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["po218"]["decay_constant"]**2*isotope_dict["rn222"]["decay_constant"]**2*n218po0 - isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**3*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) + (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**3*r + isotope_dict["po218"]["decay_constant"]**3*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) + (-np.exp(isotope_dict["bi214"]["decay_constant"]*t) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r)) + isotope_dict["pb214"]["decay_constant"]**3*(isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**2*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 - (-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) - (np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]**2*r - isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0 + n222rn0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0) + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r))) + isotope_dict["bi214"]["decay_constant"]**3*((-isotope_dict["pb214"]["decay_constant"]**3)*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(n214bi0 + n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0) + isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]*(-isotope_dict["po218"]["decay_constant"] + isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"]*n214bi0 + isotope_dict["rn222"]["decay_constant"]*n214bi0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*r) + isotope_dict["pb214"]["decay_constant"]**2*(isotope_dict["po218"]["decay_constant"]**2*(n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*n214pb0 + (-np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*n218po0) + isotope_dict["po218"]["decay_constant"]*(np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*isotope_dict["rn222"]["decay_constant"]*n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*isotope_dict["rn222"]["decay_constant"]*(n218po0 + n222rn0) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*(isotope_dict["rn222"]["decay_constant"]*n222rn0 - r) + np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t))*r) + isotope_dict["rn222"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*isotope_dict["rn222"]["decay_constant"]*n214pb0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*r)) + isotope_dict["pb214"]["decay_constant"]*(isotope_dict["po218"]["decay_constant"]**3*n214bi0 - isotope_dict["po218"]["decay_constant"]*isotope_dict["rn222"]["decay_constant"]**2*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 + (np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)) - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*(n218po0 + n222rn0)) - isotope_dict["rn222"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*n214bi0 + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t)))*r) + isotope_dict["po218"]["decay_constant"]**2*(isotope_dict["rn222"]["decay_constant"]*((-1 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t)))*n214pb0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*t))*n218po0 - np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t))*n222rn0 + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*t))*(n218po0 + n222rn0)) + (-np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t)) + np.exp(np.float128((isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*t)))*r))))/np.exp(np.float128(isotope_dict["bi214"]["decay_constant"]*t))/(isotope_dict["bi214"]["decay_constant"]*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["pb214"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["po218"]["decay_constant"])*(isotope_dict["bi214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["pb214"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])*(isotope_dict["po218"]["decay_constant"] - isotope_dict["rn222"]["decay_constant"])))


# This function is used to extract the 'peak_data_dict' from 'raw_data' and 'measurement_data'.
def get_activity_data_dict(
    measurement_data_dict,
    raw_data_ndarray,
    decay_model_fit_settings_dict = {
        "fit_range_s" : [8000, 13000], # time interval since DAQ start (in seconds) that is considered for the fit
        "activity_interval_ps" : activity_interval_h* 60 *60 *10**(12), # time interval (in picoseconds) during which the po218 and po214 decays are accumulated
        "flag_calibration" : ["self_absolute_adcc", "self_relative_adcc", "self_relative_sigma"][1],
        "po218_window" : [1000, 250],
        "po214_window" : [1000, 250],
        "flag_model_fit" : ["fit_po218_and_po214_independently", "fit_po218_and_po214_simultaneously"][1],
        "p_opt_bounds" : ([0,0,0,0,0],[+np.inf,10,10,10,+np.inf]), # 
        "p_opt_guess" : [0,0,0,0,0.02], # 
        "flag_errors" : ["poissonian"][0]
    },
):

    # analysis preparation
    timestamp_edges_ps = [0]
    timestamp_ctr = 1
    while timestamp_edges_ps[len(timestamp_edges_ps)-1] +decay_model_fit_settings_dict["activity_interval_ps"] < max(raw_data_ndarray["timestamp_ps"]):
        timestamp_edges_ps.append(decay_model_fit_settings_dict["activity_interval_ps"]*timestamp_ctr)
        timestamp_ctr += 1
    timestamp_centers_ps = [i +0.5*decay_model_fit_settings_dict["activity_interval_ps"] for i in timestamp_edges_ps[:-1]]
    timestamp_centers_seconds = [i/(1000**4) for i in timestamp_centers_ps]

    # extracting the detected counts per time bin
    decays_per_activity_interval_po218 = []
    decays_per_activity_interval_po214 = []
    decays_per_activity_interval_po218_errors_lower = []
    decays_per_activity_interval_po214_errors_lower = []
    decays_per_activity_interval_po218_errors_upper = []
    decays_per_activity_interval_po214_errors_upper = []

    # determining the adcc selection windows for the individual peaks
    if decay_model_fit_settings_dict["flag_calibration"] not in ["self_absolute_adcc", "self_relative_adcc", "self_relative_sigma"]:
        print(f"include calibration by external file here")
    else:
        list_of_peaknums_with_a_priori_entries = [peaknum for peaknum in[*measurement_data_dict["peak_data"]] if "a_priori" in [*measurement_data_dict["peak_data"][peaknum]]]
        po218_peaknum = [a_priori_peaknum for a_priori_peaknum in list_of_peaknums_with_a_priori_entries if measurement_data_dict["peak_data"][a_priori_peaknum]["a_priori"]["isotope"]=="po218"][0]
        po214_peaknum = [a_priori_peaknum for a_priori_peaknum in list_of_peaknums_with_a_priori_entries if measurement_data_dict["peak_data"][a_priori_peaknum]["a_priori"]["isotope"]=="po214"][0]
        if decay_model_fit_settings_dict["flag_calibration"] == "self_absolute_adcc":
            adcc_selection_window_po218_left = decay_model_fit_settings_dict["po218_window"][0]
            adcc_selection_window_po218_right = decay_model_fit_settings_dict["po218_window"][1]
            adcc_selection_window_po214_left = decay_model_fit_settings_dict["po214_window"][0]
            adcc_selection_window_po214_right = decay_model_fit_settings_dict["po214_window"][1]
        elif decay_model_fit_settings_dict["flag_calibration"] == "self_relative_adcc":
            adcc_selection_window_po218_left = measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["mu"] -decay_model_fit_settings_dict["po218_window"][0]
            adcc_selection_window_po218_right = measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["mu"] +decay_model_fit_settings_dict["po218_window"][1]
            adcc_selection_window_po214_left = measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["mu"] -decay_model_fit_settings_dict["po214_window"][0]
            adcc_selection_window_po214_right = measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["mu"] +decay_model_fit_settings_dict["po214_window"][1]
        elif decay_model_fit_settings_dict["flag_calibration"] == "self_relative_sigma":
            adcc_selection_window_po218_left = measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["mu"] -(decay_model_fit_settings_dict["po218_window"][0] *measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["sigma"])
            adcc_selection_window_po218_right = measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["mu"] +(decay_model_fit_settings_dict["po218_window"][1] *measurement_data_dict["peak_data"][po218_peaknum]["fit_data"]["sigma"])
            adcc_selection_window_po214_left = measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["mu"] -(decay_model_fit_settings_dict["po214_window"][0] *measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["sigma"])
            adcc_selection_window_po214_right = measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["mu"] +(decay_model_fit_settings_dict["po214_window"][1] *measurement_data_dict["peak_data"][po214_peaknum]["fit_data"]["sigma"])

    # determining the detected po218 and po214 decays per 'activity_interval_ps'
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
        decays_per_activity_interval_po218_errors_lower.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po218[i], flag_mode=decay_model_fit_settings_dict["flag_errors"])[0])
        decays_per_activity_interval_po218_errors_upper.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po218[i], flag_mode=decay_model_fit_settings_dict["flag_errors"])[1])
        decays_per_activity_interval_po214_errors_lower.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po214[i], flag_mode=decay_model_fit_settings_dict["flag_errors"])[0])
        decays_per_activity_interval_po214_errors_upper.append(calc_poissonian_error(number_of_counts=decays_per_activity_interval_po214[i], flag_mode=decay_model_fit_settings_dict["flag_errors"])[1])

    # model fit
    if decay_model_fit_settings_dict["flag_model_fit"] == "fit_po218_and_po214_independently":
        print(f"This still needs to be implemented")
    elif decay_model_fit_settings_dict["flag_model_fit"] == "fit_po218_and_po214_simultaneously":
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
            sigma = decays_per_activity_interval_po218_errors_upper +decays_per_activity_interval_po214_errors_upper,
            absolute_sigma = True,
            bounds = decay_model_fit_settings_dict["p_opt_bounds"],
            p0 = decay_model_fit_settings_dict["p_opt_guess"])
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

    # returning the 'activity_data_dict'
    activity_data_dict = {
        "decay_model_fit_settings" : decay_model_fit_settings_dict,
        "decay_model_fit_results" : {
            "po218" : {
                "timestamp_centers_seconds" : timestamp_centers_seconds,
                "decays_per_activity_interval" : decays_per_activity_interval_po218,
                "decays_per_activity_interval_errors_lower" : decays_per_activity_interval_po218_errors_lower,
                "decays_per_activity_interval_errors_upper" : decays_per_activity_interval_po218_errors_upper,
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
        "activity_extrapolation" : {}
    }
    return activity_data_dict


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
def plot_activity_data(
    measurement_data_dict,
    raw_data_ndarray,
    plot_settings_dict = {
        "output_pathstrings" : [],
        "plot_format" : "talk",
        "x_axis_units" : ["seconds", "radon_half_lives"][1],
        "x_lim_s" : ["max", [0, 2*24*60*60]][0],
        "flag_preliminary" : [False, True][1],
        "legend_kwargs" : {}
    }
):

    # figure formatting
    fig, ax1 = plt.subplots(figsize=miscfig.image_format_dict[plot_settings_dict["plot_format"]]["figsize"], dpi=150)
    if plot_settings_dict["x_lim_s"] == "max":
        xlim = [0,max(raw_data_ndarray["timestamp_ps"])/(10**12)]
    else:
        xlim = plot_settings_dict["x_lim_s"]
    if plot_settings_dict["x_axis_units"] == "seconds":
        latex_time_unit_string = r"$\mathrm{s}$"
        time_unit_conversion_factor = 1
    elif plot_settings_dict["x_axis_units"] == "radon_half_lives":
        time_unit_conversion_factor = 1/(isotope_dict["rn222"]["half_life_s"])
        latex_time_unit_string = r"$T_{\frac{1}{2},\,^{222}\mathrm{Rn}}$"
    ax1.set_xlabel(r"time since $t_{\mathrm{meas}}^{\mathrm{i}}$ / " +latex_time_unit_string)
    ax1.set_ylabel(r"decays detected within $" +f"{measurement_data_dict['activity_data']['decay_model_fit_settings']['activity_interval_ps']/(10**12*60*60):.1f}" +"\,\mathrm{h}$")
    fit_plot_x_vals_s = np.linspace(start=xlim[0], stop=xlim[1], endpoint=True, num=500)
    xlim = [xlim[0]*time_unit_conversion_factor, xlim[1]*time_unit_conversion_factor]
    ax1.set_xlim(xlim)

    # plotting the activity data
    for isotope in ["po218", "po214"]:
        plt.plot(
            [ts*time_unit_conversion_factor for ts in measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["timestamp_centers_seconds"]],
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["decays_per_activity_interval"],
            linewidth = 1,
            marker = "o",
            markersize = 3.8,
            markerfacecolor = "white",
            markeredgewidth = 0.5,
            markeredgecolor = isotope_dict[isotope]["color"],
            linestyle = "",
            alpha = 1,
            label = isotope_dict[isotope]["latex_label"] +" (data)",
            zorder = 1)
        plt.errorbar(
            marker = "", # plotting just the errorbars
            linestyle = "",
            fmt = '',
            x = [ts*time_unit_conversion_factor for ts in measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["timestamp_centers_seconds"]],
            y = measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["decays_per_activity_interval"],
            yerr = measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["decays_per_activity_interval_errors_lower"],
            xerr = [0.5*measurement_data_dict["activity_data"]["decay_model_fit_settings"]["activity_interval_ps"]*(1/10**12)*time_unit_conversion_factor for ts in measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["timestamp_centers_seconds"]],
            ecolor = isotope_dict[isotope]["color"],
            elinewidth = 0.5,
            capsize = 1.2,
            barsabove = True,
            capthick = 0.5)

    # plotting the fits
    for isotope in ["po218", "po214"]:
        list_of_peaknums_with_a_priori_entries = [peaknum for peaknum in[*measurement_data_dict["peak_data"]] if "a_priori" in [*measurement_data_dict["peak_data"][peaknum]]]
        peaknum = [a_priori_peaknum for a_priori_peaknum in list_of_peaknums_with_a_priori_entries if measurement_data_dict["peak_data"][a_priori_peaknum]["a_priori"]["isotope"]==isotope][0]
        fit_function = integral_function_po218 if isotope == "po218" else integral_function_po214
        x_vals = [ts*time_unit_conversion_factor for ts in fit_plot_x_vals_s]
        y_vals = fit_function(
            fit_plot_x_vals_s,
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["n222rn0"],
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["n218po0"],
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["n214pb0"],
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["n214bi0"],
            measurement_data_dict["activity_data"]["decay_model_fit_results"][isotope]["r"]
            )
        plt.plot(
            x_vals,
            y_vals,
            linewidth = 2,
            linestyle = "-",
            color = isotope_dict[measurement_data_dict["peak_data"][peaknum]["a_priori"]["isotope"]]["color"],
            alpha = 1,
            label = isotope_dict[measurement_data_dict["peak_data"][peaknum]["a_priori"]["isotope"]]["latex_label"] +" (fit)",
            zorder = 30)

    # marking as 'preliminary'
    plt.text(
        x = 0.97,
        y = 0.95,
        transform = ax1.transAxes,
        s = "preliminary",
        color = "red",
        fontsize = 13,
        verticalalignment = 'center',
        horizontalalignment = 'right')

    # saving the output plot
    plt.legend(**plot_settings_dict["legend_kwargs"])
    plt.show()
    for i in range(len(plot_settings_dict["output_pathstrings"])):
        fig.savefig(plot_settings_dict["output_pathstrings"][i])

    # canvas and axes
    return fig, ax1



