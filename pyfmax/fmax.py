#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
from math import pow
from PIL.ImageEnhance import Contrast

class MatrixClustered:
	"""
		This class allows to define a matrix which rows (objects) were clustered
		Labels can be used to describe rows (objects) and columns (features)
    	"""

	def __init__(self, matrix, clustering, labels_row=[], labels_col=[]):
		"""
		Matrix and clustering should be at least passed.
		Matrix should be a 2D Array or already a sparse (csr or csc) matrix.
		Clustering should be an array where a value v at index i defines that the object i (row i in matrix) belongs to cluster v
		Labels can be used to describe rows (objects) and columns (features) in the same way as the clustering object
    		"""

		self.matrix_csr = csr_matrix(matrix)

		self.matrix_csc = csc_matrix(matrix)

		self.clustering=clustering
		
		self.clusters=[]
		for idx,elmt in enumerate(self.clustering):
			elmt=int(elmt)
			taille=(len(self.clusters) -1) 
			if elmt >= taille:
				for i in range(elmt - taille):
					self.clusters.append([])
			self.clusters[elmt].append(idx)

		self.labels_row=labels_row

		self.labels_col=labels_col

		self.sum_rows=self.matrix_csr.sum(axis=1)

		self.sum_cols=self.matrix_csc.sum(axis=0)
		
		self.ffmean=np.empty(self.matrix_csr.shape[1])
		self.ffmean.fill(-1.0)
		
		self.features_selected=[]
		

		

	def sum_row(self, i):
		"""
		Get the sum of row i
    		"""
		return self.sum_rows[i]

	def sum_col(self, j):
		"""
		Get the sum of column j
		Used in Feature Precision (Predominance)
    		"""
		return self.sum_cols[:,j]
	
	def sum_col_of_cluster(self, j, k):
		"""
		Get the sum of column j
		Used in Feature Precision (Predominance)
    		"""
		column=self.matrix_csc.getcol(j).toarray();
		som=0
		for idx,elmt in enumerate(column):
			if (self.clustering[idx] == k):
				som+=elmt
		return som	
	
	def sum_cluster(self, i):
		"""
		Get the sum of cluster i
		Used in feature recall
    	"""   	
		cluster=self.clusters[i]
		som=0
		for row in cluster:
			som+=self.sum_row(row)
		return som
	
	def fp(self, j, k):
		"""
		Get the feature precision (or predominance) of feature j in cluster k
    	"""
		numerator=self.sum_col_of_cluster(j, k)
		denominator =self.sum_cluster(k)
		if denominator == 0:
			return 0
		else:
			return numerator / denominator
		
	def fr(self, j, k):
		"""
		Get the feature recall of feature j in cluster k
    	"""
		numerator=self.sum_col_of_cluster(j, k)
		denominator =self.sum_col(j)
		if denominator == 0:
			return 0
		else:
			return numerator / denominator
	
	def ff(self, j, k):
		"""
		Get the feature f measure of feature j in cluster k
    	"""
		fr=self.fr(j,k)
		fp=self.fp(j,k)
		if fr == 0 and fp == 0:
			return 0
		else:
			return (2*fr*fp) /(fr + fp) 
		
	def ff_mean(self, j):
		"""
		Get the mean value of feature f-measure for feature j across all clusters
    	"""
		mean=0
		for k in range(len(self.clusters)):
			mean+=self.ff(j,k)
		self.ffmean[j]=mean / len(self.clusters)
		return self.ffmean[j]
	
	def contrast(self, j,k):
		"""
		Get the contrast of feature j in cluster k
    	"""
		return self.ff(j, k) / self.ff_mean(j)
	
	def ff_mean_all(self):
		"""
		Get the mean value of feature f-measure for all features
    	"""
		mean=0
		for j in range(self.get_cols_number()):
			if (self.ffmean[j] == -1):
				self.ff_mean(j)
			mean+=self.ffmean[j]
		return mean / self.get_cols_number()
			
		
	def get_row_label(self, i):
		"""
		Get the label of row i
    	"""
		if len(self.labels_row) == len(self.clustering):
			return self.labels_row[i]
		else:
			return i+""
		
	def get_col_label(self, j):
		"""
		Get the label of col j
    	"""
		if len(self.labels_col) > 0:
			return self.labels_col[j]
		else:
			return j+""
		
	def get_features_selected(self):
		'''
		Return for each cluster the set of features selected
		'''
		if len(self.features_selected) == 0:
			for k in range(self.get_clusters_number()):
				selected=[]
				for j in range(self.get_cols_number()):
					ff=self.ff(j, k)
					if ff >= self.ff_mean(j) and ff >= self.ff_mean_all():
						selected.append(j)
				self.features_selected.append(selected)
		return self.features_selected
	
	def get_features_selected_flat(self):
		'''
		Return an array of the feactures selected for the whole dataset, namely all the classes
		'''
		fs=self.get_features_selected()
		fs_flat=[item for sublist in fs for item in sublist]
		return set(fs_flat)
	
	def get_rows_number(self):
		"""
		Get the number of rows
    	"""
		return self.matrix_csr.shape[0]
	
	def get_cols_number(self):
		"""
		Get the number of cols
    	"""
		return self.matrix_csr.shape[1]
	
	def get_cluster(self, k):
		return self.clusters[k]
	
	def get_size_cluster(self, k):
		return len(self.get_cluster(k))
	
	def get_clusters_number(self):
		"""
		Get the number of cols
    	"""
		return len(self.clusters)
	
	def get_cluster_of(self, i):
		'''
		Get cluster of object i
		'''
		return int(self.clustering[i])
	
	def contrast_and_select_features(self, vector, k, magnitude=1):
		'''
		Applies contrast and feature selection to a data vector supposed to belong to cluster k
		'''
		fs=self.get_features_selected_flat()
		new_vector=[]
		for j, elmt in enumerate(vector):
			if j in fs:
				new_vector.append(float(pow(self.contrast(j, k), magnitude) * elmt))
		return new_vector
		
	def contrast_and_select_matrix(self, magnitude=1):
		'''
		Applies contrast and feature selection to the current matrix
		'''
		matrix=[]
		for i in range(self.get_rows_number()):
			matrix.append(self.contrast_and_select_features(self.matrix_csr.getrow(i).toarray()[0], self.get_cluster_of(i), magnitude))
		return matrix
	
	def get_macro_PC(self):
		'''
		Return macro Positive Contrast (PC) index which is a clustering quality index using contrast
		which is the mean across all clusters of the positive contrast values divided by the cluster size
		'''
		pc=0.0
		
		for k in range(self.get_clusters_number()):
			set_of_f= self.get_features_selected()[k] #range(self.get_cols_number())
			#print(str(k) + " " + str(set_of_f))
			for j in set_of_f:
				contrast=self.contrast(j, k)
				if contrast > 1:
					pc+= 1.0 / self.get_size_cluster(k) * contrast
		return pc / self.get_clusters_number()

	def get_avg_PC(self):
		'''
		Return Positive Contrast (PCm) index which is a clustering quality index using contrast
		which is the mean across all clusters of the average positive contrast in the cluster
		'''
		pc=0.0
		
		for k in range(self.get_clusters_number()):
			set_of_f= self.get_features_selected()[k] #range(self.get_cols_number())
			#print(str(k) + " " + str(set_of_f))
			for j in set_of_f:
				contrast=self.contrast(j, k)
				if contrast > 1:
					pc+= 1.0 / len(self.get_features_selected()[k]) * contrast
		return pc / self.get_clusters_number()
	
	def get_macro_EC(self):
		'''
		Return macro Extended Contrast (EC) index which is a clustering quality index using contrast
		This quality index gives better results than PC when dealing with high dimensional data
		'''
		ec=0.0
		for k in range(self.get_clusters_number()):
			positive_contrast_k=0.0
			negative_contrast_k=0.0
			nb_pos=0
			nb_neg=0
			set_of_f=self.get_features_selected_flat() #range(self.get_cols_number()):
			for j in set_of_f:
				contrast=self.contrast(j, k)
				if contrast > 1:
					nb_pos+=1
					positive_contrast_k+= float(contrast) / self.get_size_cluster(k) 
				else:
					nb_neg+=1
					negative_contrast_k+= 1.0 / (self.get_size_cluster(k) * contrast)
			positive_contrast_k*=nb_pos
			negative_contrast_k*=nb_neg
			ec+=(positive_contrast_k+negative_contrast_k)/(nb_pos+nb_neg)
		return 1.0/ self.get_clusters_number() * ec		


	def get_avg_EC(self):
		'''
		Return the average Extended Contrast (EC) index which is a clustering quality index using contrast
		This quality index gives better results than PC when dealing with high dimensional data
		'''
		nb_pos=0
		nb_neg=0
		vPsum=0.0
		vNsum=0.0
		for k in range(self.get_clusters_number()):
			positive_contrast_k=0.0
			negative_contrast_k=0.0
			set_of_f=self.get_features_selected_flat() #range(self.get_cols_number()):
			for j in set_of_f:
				contrast=self.contrast(j, k)
				if contrast > 1:
					nb_pos+=1
					positive_contrast_k+= float(contrast) 
				else:
					nb_neg+=1
					negative_contrast_k+= 1.0 / contrast
			positive_contrast_k/=len(self.get_features_selected()[k])
			negative_contrast_k/=len(self.get_features_selected()[k])
			vPsum+=positive_contrast_k
			vNsum+=negative_contrast_k
		vPsum/=self.get_clusters_number()
		vNsum/=self.get_clusters_number()
		if (self.get_clusters_number() <= 5):
			ec=(vPsum*nb_pos+  (1-1.0/(self.get_clusters_number()))*vNsum*nb_neg)/(nb_pos+nb_neg)
		else:
			ec=(vPsum*nb_pos+vNsum*nb_neg)/(nb_pos+nb_neg)
		return ec			

	def __str__(self):
		"""
		toString()
    		"""
		return "Matrix CSR (ordered by rows) :\n" + str(self.matrix_csr)+ "\nMatrix CSC (ordered by columns): \n"+ str(self.matrix_csc) + "\nColumns labels (features) " + str(self.labels_col) + "\nRows labels (objects) " + str(self.labels_row) + "\nClustering :  " + str(self.clustering)+"\nClusters : "+str(self.clusters)
		
