# GKKR
Codes related to running Gaussian Kernel Ridge Regression

Codes

Add Descriptor Section
Useful if you want to see how RMS evolves over entire set of Descriptors, but learning method is slow. Adds desciptors to Data
set in groups of length (m) and iterates over training-testing sets "r" times. Descriptors are either added in the order they
are present in the data set or in the order specified in a secondary file if "F" is set to true.

CVTest
Runs the CVTest using GKRR

Descriptor Needs Test
Adds Descriptors one add a time to the learning Method and iterates over "i" training and testing splits to get an average
rms. Descriptors are either added in order they are listed in in the data file, or can be added in order from a file specified
by FILENAME that has one number on each line that corresponds to a Descriptor index. This option is available if "F" is set to
true. It continues to add Descriptors until it reaches the specified amount "Limit" or all Descriptors have been added.

FitImport
Import Method used for all GKRR Files. Asks what file to import

INPlot
Plots data from a txt file for the INTest graph. After using INTest-G to get the data, and changing the formatting slightly,
this file will quickly plot the data in the standard INTest format

INTest-G
Impurity Needs Test without Graphing just printing out results

INTest
