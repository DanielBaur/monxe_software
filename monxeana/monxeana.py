
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
relpath_output = "./output/" # this is the folder (within the measurement folder) where the analysis output is stored
relpath_data = "./data/" # this is the folder where CoMPASS stores the measurement data
relpath_data_compass = "./data/DAQ/run/RAW/" # this is the folder where CoMPASS stores the measurement data


# filenames
filename_data_csv = "DataR_CH0@DT5781A_840_run.csv" # timestamp (and eventually waveform) data
filename_data_txt = "CH0@DT5781A_840_EspectrumR_run.txt" # adc spectrum data
filename_histogram_png = "histogram" # histogram plot name


# format
color_uni_blue = '#004A9B'
color_uni_red = '#C1002A'
color_monxe_cyan = "#00E8E8" # the cyan-like color that was used within the MonXe logo
color_histogram = "black"
color_histogram_error = color_monxe_cyan
linewidth_histogram_std = 0.8 # the standard linewidth for a histogram plot


# isotope colors
rn222_color = color_monxe_cyan
po218_color = "#8204d6" # purple
pb214_color = "#ff1100" # red
bi214_color = "#00b822" # green
po214_color = "#ff00e6" # pink
po210_color = "#cc047f" # purple-pink


# miscellaneous
n_adc_channels = 16383 # from channel 0 to channel 16383
adc_channel_min = 0
adc_channel_max = 16383


# isotope data
isotope_dict = {
    "rn222" : {
        "half_life" : 3.8232 *24 *60 *60, # 3.8232 d in seconds
        "decay_constant" : np.log(2)/(3.8232 *24 *60 *60),
        "color" : color_monxe_cyan,
        "initial_number_of_isotopes" : 50,
        "latex_label" : r"$^{222}\,\mathrm{Rn}$",
    },
    "po218" : {
        "half_life" : 3.071 *60, # 3.071 min
        "decay_constant" : np.log(2)/(3.071 *60),
        "color" : po218_color,
        "initial_number_of_isotopes" : 40,
        "latex_label" : r"$^{218}\,\mathrm{Po}$",
    },
    "pb214" : {
        "half_life" : 26.916 *60, # 26.916 min
        "decay_constant" : np.log(2)/(26.916 *60),
        "color" : pb214_color,
        "initial_number_of_isotopes" : 30,
        "latex_label" : r"$^{214}\,\mathrm{Pb}$",
    },
    "bi214" : {
        "half_life" : 19.8 *60, # 19.8 min
        "decay_constant" : np.log(2)/(19.8 *60),
        "color" : bi214_color,
        "initial_number_of_isotopes" : 20,
        "latex_label" : r"$^{214}\,\mathrm{Bi}$",
    },
    "po214" : {
        "half_life" : 162.3 *10**(-6), # 162.3 µs
        "decay_constant" : np.log(2)/(162.3 *10**(-6)),
        "color" : po214_color,
        "initial_number_of_isotopes" : 10,
        "latex_label" : r"$^{214}\,\mathrm{Po}$",
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





#######################################
### Retrieving Data
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


# This function is used to print the
def print_general_measurement_data(
    list_file_data
):

    # 
    print("---------------------------------------")
    print("--- General Measurement Information ---")
    print("---------------------------------------")

    # measurement duration
    t_m_days = get_measurement_duration(list_file_data=list_file_data, flag_unit="days")
    print(f"measurement duration: {t_m_days:.3f} days")

    # recorded events
    print(f"recorded events: {len(list_file_data)} (ch0: {len(list_file_data[(list_file_data['pulse_height_adc'] == 0)])})")

    print("---------------------------------------")
    return





#######################################
### Histogram Stuff
#######################################


# This is the dtype used for histogram data.
histogram_data_dtype = np.dtype([
    ("bin_centers", np.int16), # max adc channel is ~16000, np.int16 ranges from -32768 to 32767
    ("counts", np.uint64), # better safe than sorry
    ("counts_errors", np.uint64) # better safe than sorry
])


# This function is used to calculate the Poissonian error of a number of counts.
def calc_poissonian_error(
    number_of_counts,
    flag_mode = "fabian_symmetrical" # ["sqrt", "fabian", "fabian_symmetrical"]
):
    # for a large number of entries
    if flag_mode == "sqrt":
        if number_of_counts == 0:
            poissonian_error = 1
        else:
            poissonian_error = np.sqrt(number_of_counts)
        return poissonian_error
    # asymmetrical error; use "fabian_symmetrical" for curve_fit
    elif flag_mode in ["fabian", "fabian_symmetrical"]:
        alpha = 0.318
        low, high = (chi2.ppf(alpha/2, 2*number_of_counts) / 2, chi2.ppf(1-alpha/2, 2*number_of_counts + 2) / 2)
        if number_of_counts == 0:
            low = 0.0
        low_interval = number_of_counts - low
        high_interval = high - number_of_counts
        if flag_mode == "fabian":
            return low_interval, high_interval
        elif flag_mode == "fabian_symmetrical":
            return max(low_interval, high_interval)
    # catching exceptions
    else:
        raise Exception("Invalid input: 'flag_mode'.")


# This function is used to convert raw timestamp data into histogram data
def get_histogram_data_from_timestamp_data(
    timestamp_data, # the timestamp data retrieved by 'get_timestamp_data'
    histval = "pulse_height_adc",
    number_of_bins = n_adc_channels # the number of bins, per default every adc channel counts as one bin
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
            calc_poissonian_error(data_histogram_counts[i])
        ))
    histogram_data = np.array(histogram_data_tuplelist, histogram_data_dtype)
    return histogram_data


# This function is used to stepize arbitrary histogram data.
# I.e. it takes two list-like objects representing both the bin centers and also the corresponding counts and calculates two new lists containing both the left and right edges of the bins and two instances of the counts.
def stepize_histogram_data(bincenters, counts, counts_errors="", flag_addfirstandlaststep=True):
    # calculating the binwidth and initializing the lists
    binwidth = bincenters[1]-bincenters[0]
    bincenters_stepized = np.zeros(2*len(bincenters))
    counts_stepized = np.zeros(2*len(counts))
    counts_errors_stepized = np.zeros(2*len(counts))
    # stepizing the data
    for i in range(len(bincenters)):
        bincenters_stepized[2*i] = bincenters[i] -0.5*binwidth
        bincenters_stepized[(2*i)+1] = bincenters[i] +0.5*binwidth
        counts_stepized[2*i] = counts[i]
        counts_stepized[2*i+1] = counts[i]
        if counts_errors != "":
            counts_errors_stepized[2*i] = counts_errors[i]
            counts_errors_stepized[2*i+1] = counts_errors[i]
    # appending a zero to both the beginning and end so the histogram can be plotted even nicer
    bin_centers_stepized_mod = [bincenters_stepized[0]] +list(bincenters_stepized) +[bincenters_stepized[len(bincenters_stepized)-1]]
    counts_stepized_mod = [0] +list(counts_stepized) +[0]

    if flag_addfirstandlaststep==False:
        if counts_errors != "":
            return bincenters_stepized, counts_stepized, counts_errors_stepized
        else:
            return bincenters_stepized, counts_stepized
    else:
        if counts_errors != "":
            return bincenters_stepized, counts_stepized, counts_errors_stepized, bin_centers_stepized_mod, counts_stepized_mod
        else:
            return bincenters_stepized, counts_stepized, bin_centers_stepized_mod, counts_stepized_mod


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
### Fitting
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
        sigma = fit_data["counts_errors"],
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


# This function is used to print the contents of 'peak_parameter_dict'
def print_peak_data_dict_contents(pdd):
    for k in pdd:
        print("\n########################################")
        print(f"peak: {k}")
        print("########################################\n")
        spacingstr = "   "
        for ke in pdd[k]:
            print(ke)
            if type(pdd[k][ke]) == dict:
                for key in pdd[k][ke]:
                    print(f"{spacingstr}{key} : {pdd[k][ke][key]}")
            elif type(pdd[k][ke]) == np.ndarray:
                print(f"{spacingstr}np.ndarray of length {len(pdd[k][ke])}")
            else:
                print(f"{spacingstr}{pdd[k][ke]}")
            print("")
        print("\n")
    return





#######################################
### Analysis
#######################################


# This function is corresponding to the exponential rise of the 218Po and 214Po activities.
def exp_rise(t, a, lambd):
    return a*(1-np.exp(-lambd*t))




# This function is fitted with curve_fit() in order to extract a better measurement of the initial radon activity.
def integral_function_po218(
        t, # time at which the integral of the integrand_function over 'integration_interval_width' is evaluated
        n222rn0,
        n218po0,
        r):

    # case a: array-input (e.g. if function is fitted with curve_fit() )
    if hasattr(t, '__len__'):
        return [integrate.quad(
                func = po218,
                a = ts-0.5*activity_interval_h*60*60,
                b = ts+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, r)
            )[0] for ts in t]

    # case b: scalar-like (e.g. for explicit calculations)
    else:
        return integrate.quad(
                func = po218,
                a = t-0.5*activity_interval_h*60*60,
                b = t+0.5*activity_interval_h*60*60,
                args = (n222rn0, n218po0, r)
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


# This function is used to linearly transform an array.
def linear_transform(input_array_or_val, peak_param_dict):
    
    # calculating the linear function coefficients
    p1 = [peak_param_dict["3"]["fit_data"]["mu"], peak_param_dict["3"]["isotope_data"]["energy_in_MeV"]]
    p2 = [peak_param_dict["4"]["fit_data"]["mu"], peak_param_dict["4"]["isotope_data"]["energy_in_MeV"]]
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    t = p2[1] -m*p2[0]
    
    # case: input is array-like
    if hasattr(input_array_or_val, "__len__"):
        ret_val = input_array_or_val.copy()
        for i in range(len(ret_val)):
            ret_val[i] = ret_val[i]*m +t

    # case: input is scalar-like
    else:
        ret_val = input_array_or_val*m +t

    return ret_val


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




