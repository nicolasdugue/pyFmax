#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
from math import pow

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
	
	def contrast_and_select_features_matrix(self, matrix, classes, magnitude=1):
		'''
		Applies contrast and feature selection to a data matrix
		'''
		result=[]
		for idx,vector in enumerate(matrix):
			result.append(self.contrast_and_select_features(vector, int(classes[idx]), magnitude))
		return result
		
	def contrast_and_select_matrix(self, magnitude=1):
		'''
		Applies contrast and feature selection to the current matrix
		'''
		matrix=[]
		for i in range(self.get_rows_number()):
			matrix.append(self.contrast_and_select_features(self.matrix_csr.getrow(i).toarray()[0], self.get_cluster_of(i), magnitude))
		return matrix

	def __str__(self):
		"""
		toString()
    		"""
		return "Matrix CSR (ordered by rows) :\n" + str(self.matrix_csr)+ "\nMatrix CSC (ordered by columns): \n"+ str(self.matrix_csc) + "\nColumns labels (features) " + str(self.labels_col) + "\nRows labels (objects) " + str(self.labels_row) + "\nClustering :  " + str(self.clustering)+"\nClusters : "+str(self.clusters)
	
	
	
		
	
#For Dataset splitting
from sklearn.cross_validation import train_test_split
#For logging
import logging
#For PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

class MetaLearner:
	def __init__(self, X, Y, labels_row=[], labels_col=[], perct_test=0.25, magnitude=1):
		'''
		X the matrix of data, Y the classes
		'''
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=perct_test)
		self.Y_train=self.Y_train.astype(int)
		self.Y_test=self.Y_test.astype(int)
		print self.Y_test
		
		self.Y_error=[]
		self.magnitude=magnitude
		logger = logging.getLogger()
		logger.setLevel(20)
		self.contrasted_bool=False
		logging.info("Dataset split in train and test sets")
		logging.info("Train set : " + str(len(self.X_train)))
		logging.info("Test set : " + str(len(self.X_test)))
		self.matrix=MatrixClustered(self.X_train, self.Y_train, labels_row, labels_col)
		self.matrix_contrasted=self.matrix.contrast_and_select_matrix(self.magnitude)
		self.matrix_contrasted=np.array(self.matrix_contrasted).reshape(-1,len(self.matrix.get_features_selected_flat()))
		
	def get_cluster_numbers(self):
		return self.matrix.get_clusters_number()
	
	def get_original_matrix_size(self):
		return len(self.X_train) + len(self.X_test)
	
	def get_original_train(self):
		'''
		Get the original train data : data are not contrasted
		'''
		return self.X_train
	
	def get_contrasted_train(self):
		'''
		Return the train set contrasted
		'''
		return self.matrix_contrasted
	
	def get_train_classes(self):
		'''
		Return the train classes
		'''
		return self.Y_train
	
	def get_original_matrix(self):
		'''
		Return the original data without any contrast applied, and before split in train and test
		'''
		return np.array(np.append(self.X_train,self.X_test)).reshape(-1,len(self.X_train[0,:]))
	
	def get_contrasted_original_matrix(self):
		'''
		Return the original data without any contrast applied, and before split in train and test
		'''
		return self.matrix.contrast_and_select_features_matrix(self.get_original_matrix(), self.get_classes(), self.magnitude)
	
	def get_classes(self):
		'''
		Get classes for the whole dataset
		'''
		return np.append(self.Y_train, self.Y_test)
	
	def train(self, contrasted_bool, classifier):
		'''
		contrasted_bool allows to use or not feature selection process
		classifier should be a scikit learn classifier
		'''
		self.contrasted_bool=contrasted_bool
		if not contrasted_bool:
			self.classifier=classifier.fit(self.get_original_train(), self.get_train_classes())
		else:
			self.classifier=classifier.fit(self.get_contrasted_train(), self.get_train_classes())
	
	def predict_with_contrasted_class(self):
		'''
		predict knowing class applying contrast : just to test the code
		'''
		self.Y_predicted=[]
		self.Y_error=[]
		for idx,vector in enumerate(self.X_test):
			best=0
			maxi=-1
			vector_contrasted=self.matrix.contrast_and_select_features(vector, self.Y_test[idx], self.magnitude)
			prediction=self.classifier.predict(vector_contrasted)
			print prediction[0], self.Y_test[idx]
			if prediction[0] != self.Y_test[idx]:
				print self.classifier.predict_proba(vector)
				print self.classifier.predict_proba(vector_contrasted)
				self.Y_error.append(idx)
			self.Y_predicted.append(prediction[0])
			#print "\n"
		return(self.Y_predicted == self.Y_test)
	
	def predict(self):
		'''
		predict results using the trained classifier with train function
		'''
		print ("Contrast : "+ str(self.contrasted_bool))
		self.Y_error=[]
		if not self.contrasted_bool:
			self.Y_predicted=self.classifier.predict(self.X_test)
		else:
			
			self.Y_predicted=[]
			for idx,vector in enumerate(self.X_test):
				best=0
				maxi=-1
				for k in range(self.matrix.get_clusters_number()):
					vector_contrasted=self.matrix.contrast_and_select_features(vector, k, self.magnitude)
					prediction=self.classifier.predict_proba(vector_contrasted)
					if prediction[0][k] > maxi:
						maxi=prediction[0][k]
						best=k
					#print prediction, k, maxi, best, self.Y_test[idx]
				if (self.Y_test[idx] != best):
					self.Y_error.append(idx)
				self.Y_predicted.append(best)
				#print "\n"
		return(self.Y_predicted == self.Y_test)
	
	def pca_train(self, contrasted_bool):
		'''
		Allows to run a pca on train data
		'''
		if contrasted_bool:
			X=self.get_contrasted_train()
		else:
			X=self.get_original_train()
		
		# Plot the training points
		fig = plt.figure(2, figsize=(8, 6))
		X_reduced = PCA(n_components=2).fit_transform(X)
		plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.get_train_classes(), cmap= plt.cm.get_cmap('RdYlBu'))
		plt.xlabel("1st eigenvector")
		plt.ylabel("2nd eigenvector")
		
		if (len(self.matrix.get_features_selected_flat())>2):
		
			# To getter a better understanding of interaction of the dimensions
			# plot the first three PCA dimensions
			fig = plt.figure(1, figsize=(8, 6))
			ax = Axes3D(fig, elev=-150, azim=110)
			X_reduced = PCA(n_components=3).fit_transform(X)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=self.get_train_classes(), cmap= plt.cm.get_cmap('RdYlBu'))
			ax.set_title("First three PCA directions")
			ax.set_xlabel("1st eigenvector")
			ax.w_xaxis.set_ticklabels([])
			ax.set_ylabel("2nd eigenvector")
			ax.w_yaxis.set_ticklabels([])
			ax.set_zlabel("3rd eigenvector")
			ax.w_zaxis.set_ticklabels([])
		
		plt.show()

	def pca_dataset(self, contrasted_bool, error_bool=False):
		'''
		Allows to run data on the whole dataset
		If error_bool is True, classification errors will be colored differently
		'''
		if contrasted_bool:
			X=self.get_contrasted_original_matrix()
		else:
			X=self.get_original_matrix()
		classes=self.get_classes()
		if error_bool:
			for idx in self.Y_error:
				classes[idx+len(self.X_train)]=self.matrix.get_clusters_number()
		
		# Plot the training points
		fig = plt.figure(2, figsize=(8, 6))
		X_reduced = PCA(n_components=2).fit_transform(X)
		plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=classes, cmap= plt.cm.get_cmap('RdYlBu'), s=50)
		plt.xlabel("1st eigenvector")
		plt.ylabel("2nd eigenvector")
		
		if (len(self.matrix.get_features_selected_flat())>2):
		
			# To getter a better understanding of interaction of the dimensions
			# plot the first three PCA dimensions
			fig = plt.figure(1, figsize=(8, 6))
			ax = Axes3D(fig, elev=-150, azim=110)
			X_reduced = PCA(n_components=3).fit_transform(X)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=classes, cmap= plt.cm.get_cmap('RdYlBu'), s=50)
			ax.set_title("First three PCA directions")
			ax.set_xlabel("1st eigenvector")
			ax.w_xaxis.set_ticklabels([])
			ax.set_ylabel("2nd eigenvector")
			ax.w_yaxis.set_ticklabels([])
			ax.set_zlabel("3rd eigenvector")
			ax.w_zaxis.set_ticklabels([])
		
		plt.show()
		
	def pca_dataset_errors_contrast(self):
		'''
		Allows to run data on the whole dataset
		If error_bool is True, classification errors will be colored differently
		'''
		X=self.get_original_matrix()
		classes=self.get_classes()
		
		#We add the errors with all the classes to compare the different values contrasted
		for idx in self.Y_error:
			for k in range(self.get_cluster_numbers()):
				if k != classes[idx+len(self.X_train)]:
					classes=np.append(classes, k)
					vector=[]
					vector.append(X[idx+len(self.X_train)])
					X=np.array(np.append(X, vector)).reshape(-1,len(self.X_train[0,:]))
		
		X=self.matrix.contrast_and_select_features_matrix(X, classes, self.magnitude)
		cpt_error=0
		cpt=0
		for idx in self.Y_error:
			for k in range(self.get_cluster_numbers()):
				if k != classes[idx+len(self.X_train)]:
					print self.get_original_matrix_size()+cpt
					classes[self.get_original_matrix_size()+cpt]=self.get_cluster_numbers() +cpt_error
					cpt+=1
			classes[idx+len(self.X_train)]=self.get_cluster_numbers() +cpt_error
			cpt_error+=1
		
		# Plot the training points
		fig = plt.figure(2, figsize=(8, 6))
		X_reduced = PCA(n_components=2).fit_transform(X)
		plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=classes, cmap= plt.cm.get_cmap('RdYlBu'), s=50)
		plt.xlabel("1st eigenvector")
		plt.ylabel("2nd eigenvector")
		
		if (len(self.matrix.get_features_selected_flat())>2):
		
			# To getter a better understanding of interaction of the dimensions
			# plot the first three PCA dimensions
			fig = plt.figure(1, figsize=(8, 6))
			ax = Axes3D(fig, elev=-150, azim=110)
			X_reduced = PCA(n_components=3).fit_transform(X)
			ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=classes, cmap= plt.cm.get_cmap('RdYlBu'),s=50)
			ax.set_title("First three PCA directions")
			ax.set_xlabel("1st eigenvector")
			ax.w_xaxis.set_ticklabels([])
			ax.set_ylabel("2nd eigenvector")
			ax.w_yaxis.set_ticklabels([])
			ax.set_zlabel("3rd eigenvector")
			ax.w_zaxis.set_ticklabels([])
		
		plt.show()
		