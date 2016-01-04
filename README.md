# GKKR
Codes related to running Gaussian Kernel Ridge Regression
-For most codes the FitImport.oy file is also needed because it imports the data as needed. Most codes also have a MethodSelect function or the place where the GKRR codes is is marked to allow them to be adapted to different learning methods. The main function for the program is always the last method in the file. Throughout all the files: 
Xt = training descriptor values, XT = testing descriptor values, Yt = training target values, YT = testing target values

Codes

AddDescSect (Add Descriptor Section): 
Useful if you want to see how RMS evolves over entire set of Descriptors and learning method is slow.
Adds desciptors to Data set in groups of length (m) and iterates over training-testing sets "r" times. Descriptors are either added in the order they are present in the data set or in the order specified in a secondary file if "F" is set to true.

CVTest (Cross Validation Test): 
Runs the CVTest using GKRR. 

DNTest (Descriptor Needs Test): 
Adds Descriptors one add a time to the learning Method and iterates over "i" training and testing splits to get an average
rms. Descriptors are either added in order they are listed in in the data file, or can be added in order from a file specified by FILENAME that has one number on each line that corresponds to a Descriptor index. This option is available if "F" is set to true. It continues to add Descriptors until it reaches the specified amount "Limit" or all Descriptors have been added.

FitImport: 
Import Method used for all GKRR Files. Asks what file to import and looks in current folder for file that is specified. Should work on any any data file that has purely numeric data and any labels are all in rows and columns before the data.
NOTE: If trying to use this method for a different data set (Not diffusion) comment out/remove the call to getLabels(allData). That is designed specifically for the Diffusion dataset.

FWS (Forward Selection Code):
Performs Forward Selection. Testing all Descriptors individually, adding in the best one to a best descriptor list. Then it tests all remaining Descriptors adding the one that increases the RMS by the most to the list, iteratively until a target number of Descriptors are found or no more descriptors remain untested.

INPlot (Impurity Needs Plotter): 
Plots data from a txt file for the INTest graph. After using INTest-G to get the data, and changing the formatting slightly, this file will quickly plot the data in the standard INTest format

INTest-G (impurity Needs without grpahing): 
Impurity Needs Test without Graphing just printing out results Useful if INtest is long and you want to run it on bardeen. Can then use the INPlot code with slight changes to this program's output to graph the data as if you have run the basic INTest program.

INTest (Impurity Needs Test):
Basic INTest file. 

SimpleGKRR:
A simple GKRR code that takes 3 arguments, Descriptor Data (X) , Target Values (Y), and a number of iterations(I). A good place to start when learning to code GKRR with out all the additional aspects of the more complicated code like CVTest.
