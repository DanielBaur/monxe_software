
#####################################################
### README
#####################################################


# How to Use:
#     - Implement sensors by adding an entry to the 'sensor_list' defined within 'Definitions: Sensor Initialization'.
#     - Run by executing this file ($ python3 monmonxe.py)
#     - Stop by x.
# 
# Helpful Stuff:
#     - RevPi tutorial by Kunbus:
#         - analog inputs: https://revolution.kunbus.de/tutorials/revpi-aio/analoge-eingaenge-konfigurieren/
#         - piTest: https://revolution.kunbus.de/tutorials/software-2/pitest/
#         - RevPi Forum: https://revolution.kunbus.de/forum/





#####################################################
### Imports
#####################################################


import time
import datetime
import subprocess
import argparse
import numpy as np
import os





#####################################################
### Definitions: General
#####################################################


# time in between sensor readouts (in seconds)
sleeptime = 1


# ip addresses
ip_revpi_c3 = "pi@10.42.0.187"
ip_lptp = "monxe@"


# paths within the readout laptop (lptp)
path_measurement_data = "/home/monxe/Desktop/measurement_data/" # folder within the measurement data is stored (each measurement corresponds to one subfolder)


# paths within the slow control machine (revpi_c3)
path_monmonxe_folder = "/home/pi/monmonxe/" # folder within which the slow control files are stored
path_prozessabbild_binary = "/dev/piControl0" # 'Prozessabbild' binary file within the RevPi root file system
path_sensor_outputs = "/home/pi/monmonxe/sensor_readings/" # folder within which the sensor output files are stored


# led addresses
led_offset = 6





#####################################################
### Definitions: Sensor Initializsation
#####################################################


# This function is used to generate a conversion function that will be used to compute physical quantities derived from the raw sensor readings.
# pressure sensor output --->  absolute pressure in bar.
def ret_func_to_convert_pressure_sensor_current_reading_into_absolute_pressure_in_bar():
    return lambda sensor : (((sensor.raw_reading/1000 -sensor.sensor_output[1])/(sensor.sensor_output[2]-sensor.sensor_output[1]))*(sensor.measured_quantity[2]-sensor.measured_quantity[1]))+sensor.measured_quantity[1]


# This function is used to generate a conversion function that will be used to compute physical quantities derived from the raw sensor readings.
# PT100 reading ---> temperature in Kelvin
def ret_func_to_convert_pt100_reading_into_temperature_in_kelvin():
    return lambda sensor : (sensor.raw_reading/10) +273.15


# This function is used to generate a conversion function that will be used to compute physical quantities derived from the raw sensor readings.
# PT100 reading ---> temperature in degrees Celsius
def ret_func_to_convert_pt100_reading_into_temperature_in_celsius():
    return lambda sensor : sensor.raw_reading/10


# This is the sensor class.
# For every sensor read out one sensor object has to be generated.
class sensor:

    # initializing the sensor with the minimum required data
    def __init__(
            self,
            name,
            address,
            prozessabbild_binary_string,
            offset,
            sensor_output,
            measured_quantity,
            datetimestamp=None,
            raw_reading=None,
            derived_readings={}
        ):
        self.name = name # name of the sensor (e.g. as it is used in the documentation)
        self.address = address # the 'address' is specified within 'Pictory'
        self.prozessabbild_binary = prozessabbild_binary_string # path to the prozessabbild binary
        self.offset = offset # offset of the sensor readout within the 'Prozessabbild' binary file, retrieved via 'piTest': $ piTest -v InputValue_2
        self.sensor_output = sensor_output # syntax: [<min_output_range>, <max_output_range>, <output_unit>, <readout_unit_conversion_(1000*muA=mA)>]
        self.measured_quantity = measured_quantity # syntax: [<lower_bound_of_the_measured_range>, <upper_bound_of_the_measured_range>, <unit_of_the_measured_quantity>]
        self.datetimestamp = datetimestamp # the current datetimestamp will be stored here
        self.raw_reading = raw_reading # the raw reading of the sensor (the value displayed within the 'Prozessabbild') will be stored here
        self.derived_readings = derived_readings # empty dict that is supposed to hold the reading values that are forwarded to .csv files
        return

    # retrieving the current raw sensor reading
    def get_raw_sensor_reading(self):
        with open(self.prozessabbild_binary, "wb+", 0) as f: # opening the 'Prozessabbild' binary file
            f.seek(self.offset) # offsetting the coursor within the 'Prozessabbild'
            self.raw_reading = int.from_bytes(f.read(2), 'little') # generating an integer object from the two bytes retrieved from the offset position
        return

    # updating the derived readings
    def update_derived_readings(self):
        for key in sorted(self.derived_readings):
            self.derived_readings[key][0] = self.derived_readings[key][1](self)
        return

    # generating the header line for the .csv file for the current sensor configuration
    def gen_csv_header_line(self):
        headerstring = "timestamp, raw_reading"
        for key in sorted(self.derived_readings):
            headerstring = headerstring +", " +key
        return headerstring +"\n"

    # generating a containing the current sensor readings
    def gen_sensor_readings_line(self):
        sensor_readings_line = list(self.datetimestamp.split("_"))[1] +"_" +list(self.datetimestamp.split("_"))[2] +", " +str(self.raw_reading)
        for key in sorted(self.derived_readings):
            sensor_readings_line = sensor_readings_line +", " +str(self.derived_readings[key][0])
        return sensor_readings_line +"\n"


# This list contains all the connected sensors.
# Accordingly, if you want to add a sensor, add a corresponding entry.
# Everything else is supposed to work from scratch.
sensor_list = [
    
    sensor(
        name="ps_1",
        address="InputValue_2",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=13,
        sensor_output=["mA", 4, 20],
        measured_quantity=["pressure", 0, 2, "bar"],
        derived_readings={
            "pressure_in_bar" : [None, ret_func_to_convert_pressure_sensor_current_reading_into_absolute_pressure_in_bar()]
        }
    ),

    sensor(
        name="ps_2",
        address="InputValue_3",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=15,
        sensor_output=["mA", 4, 20],
        measured_quantity=["pressure", 0, 2, "bar"],
        derived_readings={
            "pressure_in_bar" : [None, ret_func_to_convert_pressure_sensor_current_reading_into_absolute_pressure_in_bar()]
        }
    ),

    sensor(
        name="ps_3",
        address="InputValue_4",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=17,
        sensor_output=["mA", 4, 20],
        measured_quantity=["pressure", 0, 2, "bar"],
        derived_readings={
            "pressure_in_bar" : [None, ret_func_to_convert_pressure_sensor_current_reading_into_absolute_pressure_in_bar()]
        }
    ),

    sensor(
        name="t_1",
        address="RTDValue_1",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=23,
        sensor_output=["PT100"],
        measured_quantity=["temperature"],
        derived_readings={
            "temperature_in_celsius" : [None, ret_func_to_convert_pt100_reading_into_temperature_in_celsius()],
            "temperature_in_kelvin" : [None, ret_func_to_convert_pt100_reading_into_temperature_in_kelvin()]
        }
    )

]





#####################################################
### Definitions: Helper Functions
#####################################################


# This function is used to generate a datestring (e.g. datestring() ---> "20190714" for 14th of July 2019)
def datestring():
    return str(datetime.datetime.today().year) +str(datetime.datetime.today().month).zfill(2) +str(datetime.datetime.today().day).zfill(2)


# This function is used to generate a timestring (e.g. timestring() ---> "172043_123" for 17h 20min 43sec 123msec)
def timestring(flag_showmilliseconds=True):
    hours = str(datetime.datetime.today().hour).zfill(2)
    minutes = str(datetime.datetime.today().minute).zfill(2)
    seconds = str(datetime.datetime.today().second).zfill(2)
    milliseconds = str(datetime.datetime.today().microsecond).zfill(6)[:-3]
    if flag_showmilliseconds == True:
        return hours +minutes +seconds +"_" +milliseconds
    elif flag_showmilliseconds == False:
        return hours +minutes +seconds
    else:
        raise Exception


# This function is used to write the sensor readints into a .txt. file.
def write_sensor_readings_to_csv_file(input_sensor, path_file=path_sensor_outputs):
    with open(path_file +datestring() +"__" +sensorname +".csv", "a+") as f:
        f.write(timestring() +", " +"".format(str(input_sensor_dict["raw_reading"])) +"\t" +"{}".format(str(input_sensor_dict["converted_reading"])) +"\n")
    return


# This function is used to determine/generate the filestring the sensor readings are appended to.
def get_sensor_csv_file(sensor, savefolderstring=path_sensor_outputs):
    sensor_header_line = sensor.gen_csv_header_line()
    savefolderstring = savefolderstring +sensor.name +"/"
    appendstring = list(sensor.datetimestamp.split("_"))[0] +"_" +sensor.name +".csv"
    ### case 1: no file exists
    if not os.path.isfile(savefolderstring +appendstring):
        with open(savefolderstring +appendstring, "w+") as f:
            f.write(sensor_header_line)
        return savefolderstring +appendstring
    ### case 2:  If it does exist...
    else:
        # get all available files corresponding to the current sensor and  date
        file_list = []
        for filename in os.listdir(savefolderstring):
            if appendstring[:-4] in filename:
                file_list.append(filename)
        # check the available files for their 'sensor_header_lines'
        for filename in sorted(file_list):
            with open(savefolderstring +filename, "r") as f:
                file_header_line = ""
                for line in f:
                    if line[0] == "#":
                        pass
                    else:
                        file_header_line = line
                        print("file_header_line: {}".format(file_header_line))
                        break
                ### case 2: corresponding file exists
                if file_header_line == sensor_header_line:
                    appendstring = filename
                    return savefolderstring +appendstring
                else:
                    pass
        ### case 3: the existing files don't match
        appendstring = sorted(file_list)[len(file_list)-1][:-4] +"_a.csv"
        with open(savefolderstring +appendstring, "w+") as f:
            f.write(sensor_header_line)
        return savefolderstring +appendstring


# This function is used to control the A1 LED of the RevPi.
# 1: green
# 2: red
# 0: off
def set_revpi_led(value, offset=led_offset, prozessabbild_binary_string=path_prozessabbild_binary):
    with open(prozessabbild_binary_string, "wb+", 0) as f: # opening the 'Prozessabbild' binary file
        f.seek(offset) # offsetting the coursor within the 'Prozessabbild'
        f.write(value.to_bytes(1, byteorder='big')) # writing one byte to the 'Prozessabbild' binary file
    return


# This function is used for the sleep led control.
def control_sleep_and_led(input_sleeptime=sleeptime, prozessabbild_binary_string=path_prozessabbild_binary, offset=led_offset):
    time.sleep(sleeptime)
    blinkylist = [
        [1, 0.3],
        [0, 0.1],
        [1, 0.5],
        [0, 0.1]
    ]
    with open(prozessabbild_binary_string, "wb+", 0) as f:
        for i in range(len(blinkylist)):
            set_revpi_led(offset=offset, value=blinkylist[i][0], prozessabbild_binary_string=prozessabbild_binary_string)
            time.sleep(blinkylist[i][1]*input_sleeptime)
    return





#####################################################
### Main: init, display, finish
#####################################################


# This is the main function used to initialize a measurement with the MonXe detector.
# The corresponding folders are created and the monmonxe_main() function is executed on 'revpi_c3' within a detached screen.
def monmonxe_init(
    measurement_data = path_measurement_data,
    slow_control_machine_address = ip_revpi_c3
):
    return


# This is the main function used to display the sensor readings from the currently running slow control session.
def monmonxe_display(
    measurement_data = path_measurement_data,
    slow_control_machine_ip_address = ip_revpi_c3,
    slow_control_machine_monmonxe_folder = path_monmonxe_folder
):
    return


# This is the main function used to finish both the running slow control session and also the current measurement
# by copying the slow control data from the slow control machine into the measurement data folder, syncing the measurement data folders and deleting the slow control data from the sc machine.
def monmonxe_finish(
    measurement_data = path_measurement_data,
    slow_control_machine_ip_address = ip_revpi_c3,
    slow_control_machine_monmonxe_folder = path_monmonxe_folder
):
    return





#####################################################
### Main: Sensor Readout
#####################################################


# This is the main function used to retrieve the readings from the 'Prozessabbild' of the RevPi.
def monmonxe_main(
    input_sensor_list = sensor_list,
    input_path_sensor_outputs = path_sensor_outputs,
    input_sleeptime = sleeptime
):

    ### initializing
    print("\n\n\n#########################################################")
    print("### monmonxe: initializing")
    print("#########################################################\n")
    # generating the directories containing the sensor output (if not already existing)
    for i in range(len(sensor_list)):
        subprocess.call("mkdir ./sensor_readings/" +sensor_list[i].name +"/", shell=True)

    ### main program
    try:
    #if True:
        while True:

            # generating datetime- date- and timestamps valid for all sensor readings for this specific iteration of the while loop
            datetimestamp = datestring() +"_" +timestring()
            datestamp = list(datetimestamp.split("_"))[0]
            timestamp = list(datetimestamp.split("_"))[1] +list(datetimestamp.split("_"))[2]

            print("\n\n#############################################")
            print("datetime: {}\n".format(datetimestamp))

            # looping over all sensors and processing their current readings
            for i in range(len(sensor_list)):
                # storing the current readings within the dictionary
                sensor_list[i].datetimestamp = datetimestamp
                sensor_list[i].get_raw_sensor_reading()
                sensor_list[i].update_derived_readings()
                # saving the current readings to the .csv output file
                appendstring = get_sensor_csv_file(sensor_list[i])
                with open(appendstring, "a+") as f:
                    f.write(sensor_list[i].gen_sensor_readings_line())
                # printing the current readings to the screen
                print("{}".format(sensor_list[i].name))
                print("raw_reading: {:.2f}".format(sensor_list[i].raw_reading))
                for key in sorted(sensor_list[i].derived_readings):
                    print("{}: {:.2f}".format(key, sensor_list[i].derived_readings[key][0]))
                print("")

            print("#############################################")

            # controling sleeping and the LED
            control_sleep_and_led(input_sleeptime=sleeptime, prozessabbild_binary_string=path_prozessabbild_binary, offset=led_offset)

    ### exiting
    # exiting gracefully
    except (KeyboardInterrupt, SystemExit):
        # turning off the control LED
        set_revpi_led(0)
        print("\n\n\n#########################################################")
        print("### monmonxe: finished")
        print("#########################################################\n")
    # catching any other exception
    except:
        # turning off the control LED
        set_revpi_led(0)
        print("\n\n\n#########################################################")
        print("### monmonxe: AN EXCEPTION OCCURED !")
        print("#########################################################\n")





#####################################################
### Executing Main
#####################################################


# loading a list containing 
if __name__=="__main__":

    # processing the input given when this file is executed
    parser = argparse.ArgumentParser(description='Initialize MonXe measurement and slow control.')
    parser.add_argument('-r', '--runmode', dest='runmode', type=str, required=False, default="slow_control")
    runmode = parser.parse_args().runmode

    # case 1: initializing a new measurement
    if runmode in ["i", "in", "init", "initialize", "initialise"]:
        monmonxe_init()

    # case 2: running the slow control (default)
    elif runmode in ["slow_control", "run_slow_control", "sc", "main"]:
        monmonxe_main()

    # case 3: display the current sensor readings
    elif runmode in ["display", "read", "d", "sensor_readings"]:
        monmonxe_display()

    # case 4: finishing the current measurement
    elif runmode in ["f", "finish", "finished", "final", "fin"]:
        monmonxe_finish()

    # case 5: invalid input
    else:
        print("That's falsch!")
        print("It's not working.")
        print("But it should.")
        print("It isn't.")
        print("But it should...")
















