import pandas as pd
import numpy as np

class EmbeddingGenerator ():
	def __init__(self, crs_df, course_to_id):
		self.crs_df = crs_df
		self.course_to_id = course_to_id
		self.n_valid = int(2*np.sqrt(len(self.crs_df)))
		self.n_train = len(self.crs_df) - self.n_valid
		self.courses_set = courses_set

	def train_generator():    
	    while True:
	        for (student, term), df in self.crs_df.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
	            self.courses_set = make_set(df)
	            for crs_1 in self.courses_set:
	                for crs_2 in self.courses_set: 
	                    ## we just get the ids and the embedding
	                    ## takes care of the one-hot-i-fying
	                    x = self.course_to_id[crs_1]
	                    y = self.course_to_id[crs_2]
	                    if x!=y:
	                        yield x,y  
	                    else:
	                        continue
	                        
	# def valid_generator():    
	#     while True:
	#         for (student, term), sets in self.crs_df.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
	#             self.courses_set = make_set(df)
	#             for crs_1 in self.courses_set:
	#                 for crs_2 in self.courses_set: 
	#                     x = self.course_to_id[crs_1]
	#                     y = self.course_to_id[crs_2]
	#                     if x!=y:
	#                         yield x,y  
	#                     else:
	#                         continue