#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Include the fmax file of package pyfmax using fm namespace
import pyfmax.fmax as fm
#import numpy lib using np namespace
import numpy as np

#Load the file as a 1D array
clustering=np.loadtxt("../data/exemple_isko/clustering_isko")
labels_col=[]
#Read the labels of columns (features)
labels=open("../data/exemple_isko/label_isko")
for ligne in labels:
	labels_col.append(ligne.strip())
#Load the file as a 2D array
matrix=np.loadtxt("../data/exemple_isko/matrix_isko")

#Create a MatrixClustered object using fm namespace which refers to fmax.py in package pyfmax
obj=fm.MatrixClustered(matrix, clustering,labels_col=labels_col)
print obj

print obj.ff(0, 0)
print obj.ff(0, 1)
print obj.ff(1, 0)
print obj.ff(1, 1)
print obj.ff(2,0)
print obj.ff(2,1)

print obj.ff_mean(0)
print obj.ff_mean(1)
print obj.ff_mean(2)
print obj.ff_mean_all()


print obj.contrast(0, 0)
print obj.contrast(0, 1)
print obj.contrast(1, 0)
print obj.contrast(1, 1)
print obj.contrast(2,0)
print obj.contrast(2,1)


for idx, list_features in enumerate(obj.get_features_selected()):
	print "cluster ", idx
	for f in list_features:
		print obj.get_col_label(f)
		
print obj.get_features_selected_flat()

print obj.contrast_and_select_matrix()
