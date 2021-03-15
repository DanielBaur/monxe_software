
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
import sqlite3






#####################################################
### Definitions: General
#####################################################


# time in between sensor readouts (in seconds)
sleeptime = 1


# configuring the readout machine (currently 'lptp')
ip_readout_machine = "monxe@"
path_software = "/home/monxe/Desktop/monxe_software/monmonxe/"
path_measurement_data = "/home/monxe/Desktop/monxe_measurements/" # folder within the measurement data is stored (each measurement corresponds to one subfolder)
path_temp_folder = path_measurement_data +"00000000__temp/"
thisfilename = "monmonxe.py"


# paths within the slow control machine (revpi_c3)
ip_slow_control_machine = "pi@10.42.0.187"
path_monmonxe_folder = "/home/pi/monmonxe/" # folder within which the slow control files are stored
path_prozessabbild_binary = "/dev/piControl0" # 'Prozessabbild' binary file within the RevPi root file system
path_sensor_outputs = "/home/pi/monmonxe/sensor_readings/" # folder within which the sensor output files are stored


# miscellaneous definitions
slow_control_db_file_name = "monmonxe_slow_control_data.db"
slow_control_db_table_name = "slow_control_data"
led_offset = 6
sqlite_db_format = "(datetime, sensorname, reading_raw, reading, reading_error)" # this is the format (i.e. the names of the columns) of the database table





#####################################################
### Definitions: Sensor Initialization
#####################################################


# This function is used to generate a conversion function that will be used to compute physical quantities derived from the raw sensor readings.
# pressure sensor output --->  absolute pressure in bar.
def sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor):
    return lambda sensor : (((sensor.reading_raw*conversion_factor -sensor.sensor_output[2])/(sensor.sensor_output[3]-sensor.sensor_output[2]))*(sensor.measured_quantity[3]-sensor.measured_quantity[2]))+sensor.measured_quantity[2]


# This function is used to generate a function that is calculating the error on the pressure reading of the OMEGA pressure sensors.
# This error is comprised of the accurracy of the pressure sensors themselves (0.05% FS) and the readout of the sensor current (72muA, see RevPi AIO data sheet).
# The total error is calculated as the squared sum of those: s = sqrt(s_aio^2 +s_ps^2).
def sensorfunction__error_for_omega_pressure_sensors():
    s_aio = (20/(20000-4000)) *2 # accurracy of the current measurement is 20muA, sensor output is 4m to 20ma, pressure range is 2bar
    s_ps = 0.05 *0.01 *2 # accurracy of the sensor is 0.05% FS (i.e. 0.05% of the full pressure range)
    s_total = np.sqrt(s_ps**2 +s_aio**2)
    return lambda sensor : s_total


# This function is used to generate a conversion function that will be used to compute physical quantities derived from the raw sensor readings.
# PT100 reading ---> temperature in Kelvin
def sensorfunction__rtd_pt100_reading_to_temperature_in_kelvin():
    return lambda sensor : (sensor.reading_raw/10) +273.15


# This function is used to generate a function that is calculating the error of the temperature reading of PT100.
# This error is comprised of the accurracy of the PT100 thermistors themselves () and the readout of the sensor resistance ().
# The total error is calculated as the squared sum of those: s = sqrt(s_aio^2 +s_pt100^2).
def sensorfunction__error_for_rtd_pt100_reading(rtd_class):
    s_aio = 0.5 # accurracy of the PT100 measurement is 0.5K at 20°C and 1.5K from -30°C to +55°C
    rtd_class_dict = {
        "AA" : [0.1, 0.17],
        "A" : [0.15, 0.2], # accurracy of the sensor is +-(classb[0] +0.01*classb[1]*temperature_reading); https://blog.beamex.com/pt100-temperature-sensor; assuming class B
        "B" : [0.3, 0.5],
        "C" : [0.6, 1.0],
        "1/3DIN" : [0.1, 0.5],
        "1/10DIN" : [0.03, 0.5]
    }
    return lambda sensor : np.sqrt(s_aio**2 +(rtd_class_dict[rtd_class][0] +(0.01 *rtd_class_dict[rtd_class][1] *((sensor.reading_raw/10) -273.15)))**2)


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
            reading_raw =None,
            reading=None,
            reading_error=None
        ):
        self.name = name # name of the sensor (e.g. as it is used in the documentation)
        self.address = address # the 'address' is specified within 'Pictory'
        self.prozessabbild_binary = prozessabbild_binary_string # path to the prozessabbild binary
        self.offset = offset # offset of the sensor readout within the 'Prozessabbild' binary file, retrieved via 'piTest': $ piTest -v InputValue_2
        self.sensor_output = sensor_output # syntax: [<min_output_range>, <max_output_range>, <output_unit>, <readout_unit_conversion_(1000*muA=mA)>]
        self.measured_quantity = measured_quantity # syntax: [<lower_bound_of_the_measured_range>, <upper_bound_of_the_measured_range>, <unit_of_the_measured_quantity>]
        self.datetimestamp = datetimestamp # the current datetimestamp will be stored here
        self.reading_raw = reading_raw # the raw reading of the sensor (the value displayed within the 'Prozessabbild') will be stored here
        self.reading = [None, reading]
        self.reading_error = [None, reading_error]
        return

    # retrieving the current raw sensor reading
    def get_raw_sensor_reading(self):
        with open(self.prozessabbild_binary, "wb+", 0) as f: # opening the 'Prozessabbild' binary file
            f.seek(self.offset) # offsetting the coursor within the 'Prozessabbild'
            self.reading_raw = int.from_bytes(f.read(2), 'little') # generating an integer object from the two bytes retrieved from the offset position
        return

    # updating the derived readings
    def update_derived_readings(self):
        self.reading[0] = self.reading[1](self)
        self.reading_error[0] = self.reading_error[1](self)
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
        name="p_ev",
        address="InputValue_2",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=13,
        sensor_output=["current", "mA", 4, 20],
        measured_quantity=["pressure", "bara", 0, 2],
        reading = sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor=0.001),
        reading_error = sensorfunction__error_for_omega_pressure_sensors()
    ),

    sensor(
        name="m_ml",
        address="InputValue_3",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=15,
        sensor_output=["current", "mA", 4, 20],
        measured_quantity=["pressure", "bara",0, 2],
        reading = sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor=0.001),
        reading_error = sensorfunction__error_for_omega_pressure_sensors()
    ),

    sensor(
        name="p_dv",
        address="InputValue_4",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=17,
        sensor_output=["current", "mA", 4, 20],
        measured_quantity=["pressure", "bara", 0, 2],
        reading = sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor=0.001),
        reading_error = sensorfunction__error_for_omega_pressure_sensors()
    ),

    sensor(
        name="t_1",
        address="RTDValue_1",
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=23,
        sensor_output=["rtd", "pt100"],
        measured_quantity=["temperature", "kelvin"],
        reading = sensorfunction__rtd_pt100_reading_to_temperature_in_kelvin(),
        reading_error = sensorfunction__error_for_rtd_pt100_reading(rtd_class="B")
    ),

    sensor(
        name="p_ct1",
        address="InputValue_2_i03", # at 'revpi_aio_2'
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=102,
        sensor_output=["current", "mA", 4, 20],
        measured_quantity=["pressure", "bara", 0, 15],
        reading = sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor=0.001),
        reading_error = sensorfunction__error_for_omega_pressure_sensors()
    ),

    sensor(
        name="p_ct2",
        address="InputValue_3_i03", # at 'revpi_aio_2'
        prozessabbild_binary_string=path_prozessabbild_binary,
        offset=104,
        sensor_output=["current", "mA", 4, 20],
        measured_quantity=["pressure", "bara", 0, 15],
        reading = sensorfunction__linear_conversion_from_sensor_current_or_voltage_into_physical_reading(conversion_factor=0.001),
        reading_error = sensorfunction__error_for_omega_pressure_sensors()
    ),

]





#####################################################
### Helper Functions: General
#####################################################


# This function is used to generate a datestring (e.g. datestring() ---> "20190714" for 14th of July 2019)
def datestring():
    return str(datetime.datetime.today().year) +str(datetime.datetime.today().month).zfill(2) +str(datetime.datetime.today().day).zfill(2)


# This function is used to generate a timestring (e.g. timestring() ---> "172043_123" for 17h 20min 43sec 123msec)
def timestring(flag_showmilliseconds=True, flag_separate_milliseconds=True):
    hours = str(datetime.datetime.today().hour).zfill(2)
    minutes = str(datetime.datetime.today().minute).zfill(2)
    seconds = str(datetime.datetime.today().second).zfill(2)
    milliseconds = str(datetime.datetime.today().microsecond).zfill(6)[:-3]
    if flag_showmilliseconds == True and flag_separate_milliseconds == True:
        return hours +minutes +seconds +"_" +milliseconds
    elif flag_showmilliseconds == True and flag_separate_milliseconds == False:
        return hours +minutes +seconds +milliseconds
    elif flag_showmilliseconds == False:
        return hours +minutes +seconds
    else:
        raise Exception


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
    #time.sleep(sleeptime)
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
### Helper Functions: Writing Data to .csv
#####################################################


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
                        #print("file_header_line: {}".format(file_header_line))
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





#####################################################
### Helper Functions: Writing Data to .db
#####################################################




# This function is used to add an entry to a SQLite database file.
def add_entry_to_sqlite_database(
        dbconn,
        values,
        tablename=slow_control_db_table_name,
        databaseformat=sqlite_db_format
    ):
    # writing the command to add data to the database into a multiple line string
    qmstring = str(("?",)*len(values))
    print(qmstring)
    sqlstring = "INSERT INTO {}{} VALUES(?, ?, ?, ?, ?)".format(tablename, databaseformat)
    print(sqlstring)
    cur = dbconn.cursor()
    cur.execute(sqlstring, values)
    return





#####################################################
### Main: init, display, finish, update
#####################################################


# This is the main function used to initialize a measurement with the MonXe detector.
# The corresponding folders are created and the monmonxe_main() function is executed on 'revpi_c3' within a detached screen.
def monmonxe_init(
    measurement_data = path_measurement_data,
    slow_control_machine_address = ip_slow_control_machine,
    slow_control_machine_path_to_monmonxe_py = path_monmonxe_folder
):
    ### initializing
    print("\n\n\n#########################################################")
    print("### monmonxe_init: initializing a new measurement")
    print("#########################################################\n\n\n")

    ### generating the folder structure on the readout machine
    print("#############################################")
    print("### monmonxe_init: creating new measurement folder")
    datestamp = datestring()
    print("Today's datestamp is {}.".format(datestamp))
    foldername = input("How would you like to name the new measurement?\n")
    print("#############################################\n")
    folderstring = measurement_data +datestamp +"__" +foldername
    subprocess.call("mkdir {}".format(folderstring), shell=True)
    print("#############################################")
    print("### monmonxe_init: created folder {}:{}".format(slow_control_machine_address, folderstring))
    print("#############################################\n")

    ### executing 'monmonxe.py' via ssh in a detached screen 
    subprocess.call("ssh {} screen -Sdm monmonxe python3 {}monmonxe.py".format(slow_control_machine_address, slow_control_machine_path_to_monmonxe_py), shell=True)
    print("#############################################")
    print("### monmonxe_init: 'monmonxe.py' started on {}".format(slow_control_machine_address))
    print("#############################################\n")

    ### end of program
    print("\n\n#########################################################")
    print("### monmonxe_init: new measurement initialized")
    print("#########################################################\n\n\n")
    return


# This is the main function used to display the sensor readings from the currently running slow control session.
# 20210302: This function is deprecated as we are now using in-line process monitor displays to display each sensor reading.
def monmonxe_display(
    temp_filestring = path_temp_folder,
    slow_control_machine_ip_address = ip_slow_control_machine,
    slow_control_machine_monmonxe_folder = path_monmonxe_folder,
    slow_control_data_filename = slow_control_db_file_name,
    slow_control_data_tablename = slow_control_db_table_name,
    time_sleep = sleeptime
):
    ### initializing
    print("\n\n\n#########################################################")
    print("### monmonxe_display: displaying the current slow control measurement")
    print("#########################################################\n\n\n")

    ### repetedly get the current slow control data and print the latest values
    try:
        while True:
            print("#############################################")
            subprocess.call("scp {}:{}{} {}{}".format(slow_control_machine_ip_address, slow_control_machine_monmonxe_folder, slow_control_data_filename, temp_filestring, slow_control_data_filename), shell=True)
            print("### monmonxe_display: retrieved current slow control data")
            print("#############################################")
            conn = sqlite3.connect(temp_filestring +slow_control_data_filename)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM {} ORDER BY {} DESC LIMIT 6".format(slow_control_data_tablename, "datetime"))
            rows = cursor.fetchall()
            dt = str(rows[0][0])
            print("datetime: {}_{}_{}".format(dt[0:8], dt[8:12], dt[12:14]))
            print("sensor\t\treading_raw\treading\t\treading_error")
            for row in rows:
                print("{}\t\t{}\t\t{:.6f}\t\t{:.6f}".format(row[1], row[2], row[3], row[4]))
            conn.close()
            print("#############################################\n\n")
            time.sleep(time_sleep)

    ### end of program: clearing the temp folder
    # keyboard interrupt
    except (KeyboardInterrupt, SystemExit):
        print("###########################################")
        print("### monmonxe_display: stopped via keyboard interrupt")
        print("#############################################\n")
    # catching any other exceptions
    except:
        # turning off the control LED
        print("#############################################")
        print("### monmonxe_display: AN ERROR OCCURRED")
        print("#############################################\n")
    # clearing the temp folder
    finally:
        subprocess.call("rm -r {}*".format(temp_filestring), shell=True)
        print("#############################################")
        print("### monmonxe_display: cleared {}".format(temp_filestring))
        print("#############################################\n")
        print("\n\n#########################################################")
        print("### monmonxe_display: finished")
        print("#########################################################\n\n\n")
        return


# This is the main function used to finish both the running slow control session and also the current measurement
# by copying the slow control data from the slow control machine into the measurement data folder, syncing the measurement data folders and deleting the slow control data from the sc machine.
def monmonxe_finish(
    readout_machine_path_to_measurement_data = path_measurement_data,
    slow_control_machine_ip_address = ip_slow_control_machine,
    slow_control_machine_monmonxe_folder = path_monmonxe_folder,
    slowcontrol_filename = slow_control_db_file_name
):
    ### initializing
    print("\n\n\n#########################################################")
    print("### monmonxe_finish: retrieving, syncing and deleting slow control data")
    print("#########################################################\n\n\n")

    ### retrieving the pathstring of the measurement folder
    print("#############################################")
    print("### monmonxe_finish: retrieving the current measurement folder")
    print("#############################################")
    # retrieving list of available directories
    #enumerate_measurement_directories = list(enumerate(os.walk(readout_machine_path_to_measurement_data))) # for each directory within 'deck_path' a tuple is returned: (<the_current_path>, <a_list_of_subdirectories_within>, <a_list_of_files_within>)
    #measurement_directories = []
    #for i in range(len(enumerate_measurement_directories)):
    #    measurement_directories.append(list(enumerate_measurement_directories[i][1][0].split("/"))[len(list(enumerate_measurement_directories[i][1][0].split("/")))-1])
    #measurement_directories.pop(0)
    # retrieving a list of available directories 
    folder_list = list(os.listdir(path_measurement_data))
    measurement_directories = [i for i in folder_list if os.path.isdir(path_measurement_data +i)]

    # determining the most probable folder
    measurement_directories_dates = []
    for i in range(len(measurement_directories)):
        measurement_directories_dates.append(int(list(measurement_directories[i].split("__"))[0]))
    default_measurement_folder_number = measurement_directories_dates.index(max(measurement_directories_dates))
    isdefault_list = list(("",)*len(measurement_directories))
    isdefault_list[default_measurement_folder_number] = "<------------------------------------ default measurement folder"
    # asking the user for definitive input
    print("Here's a list of the available measurement directories:")
    for i in range(len(measurement_directories)):
        print("{} | {} {}".format(i, measurement_directories[i], isdefault_list[i]))
    measurement_folder_number = input("In which one would you like to store the slow control data?\n")
    try:
        measurement_folder_number = int(measurement_folder_number)
    except:
        pass
    while not (measurement_folder_number in range(0, len(measurement_directories)) or measurement_folder_number == ""):
        print("Your input was invalid!")
        measurement_folder_number = input("In which one would you like to store the slow control data?\n")
        try:
            measurement_folder_number = int(measurement_folder_number)
        except:
            pass
    # determining the measurement folder
    if measurement_folder_number == "":
        slow_control_data_pathstring = readout_machine_path_to_measurement_data +measurement_directories[default_measurement_folder_number]
    else:
        slow_control_data_pathstring = readout_machine_path_to_measurement_data +measurement_directories[measurement_folder_number]
    print("The slow control data will be stored in '{}'.".format(slow_control_data_pathstring))
    print("#############################################\n")

    ### killing the slow control process on the RevPi
    subprocess.call("ssh -t {} screen -XS monmonxe quit".format(slow_control_machine_ip_address), shell=True)
    #subprocess.call("ssh pi@10.42.0.187 screen -r monmonxe".format(slow_control_machine_ip_address), shell=True) 
    print("#############################################")
    print("### monmonxe_finish: killed slow control data acquisition")
    print("#############################################\n")

    ### retrieving the slow control data .db file
    subprocess.call("scp {}:{}{} {}{}".format(slow_control_machine_ip_address, slow_control_machine_monmonxe_folder, slowcontrol_filename, slow_control_data_pathstring+"/", slowcontrol_filename), shell=True)
    print("#############################################")
    print("### monmonxe_finish: copied slow control data into '{}'".format(slow_control_data_pathstring +"/" +slowcontrol_filename))
    
    ### syncing the measurement data between the readout machine and the desktop pc
    
    ### end of program: clearing the slow control .db file from the slow control machine
    #if False:
    subprocess.call("ssh {} rm {}{}".format(slow_control_machine_ip_address, slow_control_machine_monmonxe_folder, slowcontrol_filename), shell=True)
    print("### monmonxe_finish: cleared {}".format(slow_control_machine_ip_address +":" +slow_control_machine_monmonxe_folder +slowcontrol_filename))
    print("#############################################\n")
    print("\n\n#########################################################")
    print("### monmonxe_display: finished")
    print("#########################################################\n\n\n")
    return


# This function is used to upload the current version of 'monmonxe.py' (this file) onto the slow control machine.
def monmonxe_update(
    slow_control_machine_ip_address=ip_slow_control_machine,
    slow_control_machine_monmonxe_folder=path_monmonxe_folder,
    path_to_file=path_software,
    filename=thisfilename
):
    subprocess.call("scp {} {}".format(path_to_file +filename, slow_control_machine_ip_address +":" +slow_control_machine_monmonxe_folder +filename), shell=True)
    print("\n\n\n#########################################################")
    print("### monmonxe_update: copied 'monmonxe.py' to '{}'".format(slow_control_machine_ip_address +":" +slow_control_machine_monmonxe_folder +"monmonxe.py"))
    print("#########################################################\n\n\n")
    return





#####################################################
### Main: Sensor Readout
#####################################################


# This is the main function used to retrieve the readings from the 'Prozessabbild' of the RevPi.
def monmonxe_main(
    input_sensor_list = sensor_list,
    input_path_sensor_outputs = path_sensor_outputs,
    input_sleeptime = sleeptime,
    mode = ".db",
    databasestring = path_monmonxe_folder +slow_control_db_file_name,
    databasetablename = slow_control_db_table_name,
    databaseformat = sqlite_db_format
):

    ### initializing
    print("\n\n\n#########################################################")
    print("### monmonxe_main: recording slow control data")
    print("#########################################################\n")
    # generating the directories containing the sensor output (if not already existing)
    if mode == ".csv":
        for i in range(len(sensor_list)):
            subprocess.call("mkdir /home/pi/monmonxe/sensor_readings/" +sensor_list[i].name +"/", shell=True)
    elif mode == ".db":
        if not os.path.isfile(databasestring):
            conn = sqlite3.connect(databasestring)
            sql_monxe_table_string = """ CREATE TABLE IF NOT EXISTS {} (
                                        datetime integer NOT NULL,
                                        sensorname text NOT NULL,
                                        reading_raw integer NOT NULL,
                                        reading real NOT NULL,
                                        reading_error real NOT NULL
                                     ); """.format(databasetablename)
            conn.execute(sql_monxe_table_string)
            conn.commit()
            conn.close()
        else:
            print("\n\n#############################################")
            print("### monmonxe_main: There was already an existing database file. Delete it first - and don't forget to sync.")
            print("#############################################")
            raise Exception
    else:
        raise Exception
    ### main program
    try:
    #if True:
        while True:
            # generating datetime- date- and timestamps valid for all sensor readings for this specific iteration of the while loop
            datetimestamp = datestring() +"_" +timestring(flag_separate_milliseconds=True)
            datestamp = list(datetimestamp.split("_"))[0]
            timestamp = list(datetimestamp.split("_"))[1] +list(datetimestamp.split("_"))[2]
            print("\n\n#############################################")
            print("datetime: {}\n".format(datetimestamp))
            datetimestamp = int(datestring() +timestring(flag_separate_milliseconds=False))
            # looping over all sensors and processing their current readings
            for i in range(len(sensor_list)):
                # storing the current readings within the dictionary
                sensor_list[i].datetimestamp = datetimestamp
                sensor_list[i].get_raw_sensor_reading()
                sensor_list[i].update_derived_readings()
                # mode: saving the current readings to the .csv output file
                if mode == ".csv":
                    appendstring = get_sensor_csv_file(sensor_list[i])
                    with open(appendstring, "a+") as f:
                        f.write(sensor_list[i].gen_sensor_readings_line())
                # mode: saving the current readings to the SQLite database
                elif mode == ".db":
                    values = (datetimestamp, sensor_list[i].name, sensor_list[i].reading_raw, sensor_list[i].reading[0], sensor_list[i].reading_error[0])
                    print(values)
                    conn = sqlite3.connect(databasestring)
                    add_entry_to_sqlite_database(dbconn=conn, values=values, tablename=databasetablename, databaseformat=databaseformat)
                    conn.commit()
                    conn.close()
                # mode: no valid mode given
                else:
                    raise Exception
                # printing the current readings to the screen
                print("{}".format(sensor_list[i].name))
                print("reading_raw: {:.2f}".format(sensor_list[i].reading_raw))
                print("reading: {:.2f}".format(sensor_list[i].reading[0]))
                print("reading_error: {:.2f}".format(sensor_list[i].reading_error[0]))
                print("")
            print("#############################################")

            # controling sleeping and the LED
            control_sleep_and_led(input_sleeptime=sleeptime, prozessabbild_binary_string=path_prozessabbild_binary, offset=led_offset)

    ### exiting
    # exiting gracefully
    except (KeyboardInterrupt, SystemExit):
        # turning off the control LED
        conn.commit()
        conn.close()
        set_revpi_led(0)
        print("\n\n\n#########################################################")
        print("### monmonxe_main: finished")
        print("#########################################################\n")
    # catching any other exception
    except:
        # turning off the control LED
        conn.commit()
        conn.close()
        set_revpi_led(0)
        print("\n\n\n#########################################################")
        print("### monmonxe_main: AN EXCEPTION OCCURED !")
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
    elif runmode in ["s", "sc", "slow_control", "run_slow_control", "main"]:
        monmonxe_main()

    # case 3: display the current sensor readings
    elif runmode in ["display", "read", "d", "sensor_readings"]:
        monmonxe_display()

    # case 4: finishing the current measurement
    elif runmode in ["f", "finish", "finished", "final", "fin", "stop", "interrupt"]:
        monmonxe_finish()

    # case 5: update the 
    elif runmode in ["u", "update"]:
        monmonxe_update()

    # case 6: invalid input
    else:
        print("That's falsch!")
        print("It's not working.")
        print("But it should.")
        print("It isn't.")
        print("But it should...")
















