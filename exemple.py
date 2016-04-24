#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Include the fmax file of package pyfmax using fm namespace
import pyfmax.fmax as fm
#import numpy lib using np namespace
import numpy as np

#Load the file as a 1D array
clustering=np.loadtxt("data/exemple/clustering")
labels_col=[]
#Read the labels of columns (features)
labels=open("data/exemple/label")
for ligne in labels:
	labels_col.append(ligne.strip())
#Load the file as a 2D array
matrix=np.loadtxt("data/exemple/matrix")

#Create a MatrixClustered object using fm namespace which refers to fmax.py in package pyfmax
obj=fm.MatrixClustered(matrix, clustering,labels_col=labels_col)


print "Feature F-Measure for feature 0, cluster 0 :", obj.ff(0, 0)
print "Feature F-Measure for feature 0, cluster 1 :",obj.ff(0, 1)
print "Feature F-Measure for feature 1, cluster 0 :",obj.ff(1, 0)
print "Feature F-Measure for feature 1, cluster 1 :",obj.ff(1, 1)
print "Feature F-Measure for feature 2, cluster 0 :",obj.ff(2,0)
print "Feature F-Measure for feature 2, cluster 1 :",obj.ff(2,1)
print "\n"
print "Mean Feature F-Measure for feature 0 :",obj.ff_mean(0)
print "Mean Feature F-Measure for feature 1 :",obj.ff_mean(1)
print "Mean Feature F-Measure for feature 2 :",obj.ff_mean(2)
print "Mean Feature F-Measure for all features:",obj.ff_mean_all()
print "\n"

print "Contrast for feature 0, cluster 0 :",obj.contrast(0, 0)
print "Contrast for feature 0, cluster 1 :",obj.contrast(0, 1)
print "Contrast for feature 1, cluster 0 :",obj.contrast(1, 0)
print "Contrast for feature 1, cluster 1 :",obj.contrast(1, 1)
print "Contrast for feature 2, cluster 0 :",obj.contrast(2,0)
print "Contrast for feature 2, cluster 1 :",obj.contrast(2,1)
print "\n"

for idx, list_features in enumerate(obj.get_features_selected()):
	print "Feature selected for cluster ", idx
	for f in list_features:
		print obj.get_col_label(f)
		
print "Features selected for the whole data :",obj.get_features_selected_flat()
print "\n"
print "Contrasted matrix after feature selection",obj.contrast_and_select_matrix()
print "\n"
print "PC Value :",obj.get_PC()
print "EC Value :",obj.get_EC()
