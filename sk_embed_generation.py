import pandas as pd
import numpy as np

class EmbeddingGenerator ():
    def __init__(self, crs_df, course_to_id):
        self.crs_df = crs_df
        self.course_to_id = course_to_id
        self.unique_students = unique_students
        self.courses_set = courses_set

        np.random.seed(9)
        np.random.shuffle(self.unique_students)
        self.n_train = int(0.1*len(self.unique_students))
        self.n_valid = int(0.02*len(self.unique_students))
        self.train_students = self.unique_students[:self.n_train]
        self.valid_students = self.unique_students[self.n_train : self.n_train + self.n_valid]
        self.crs_df_train = self.crs_df[self.crs_df['PRSN_UNIV_ID'].isin(self.train_students)]
        self.crs_df_valid = self.crs_df[self.crs_df['PRSN_UNIV_ID'].isin(self.valid_students)]
		

    def train_generator(self): 
        negative_courses = self.crs_df['agg_id'].map(self.course_to_id)
        n_neg = len(negative_courses)
        while True:
            for (student, term), df in self.crs_df_train.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
                self.courses_set = make_set(df)
                if len(self.courses_set) > 1:
                    for crs_1 in self.courses_set:
                        contexts = []
                        courses_x = []
                        matches = []
                        for crs_2 in self.courses_set: 
                            x = crs_1
                            y = crs_2
                            if x!=y:
                                context = list(negative_courses.iloc[np.random.choice(n_neg,4)]) + [y]
                                course = 5*[x]
                                match = [0,0,0,0,1]
                                contexts.append(np.array(context).reshape(5,1))
                                courses_x.append(np.array(course).reshape(5,1))
                                matches.append(np.array(match).reshape(5,1))
                        contexts = np.concatenate(contexts, axis=0)
                        courses_x = np.concatenate(courses_x, axis=0)
                        matches = np.concatenate(matches, axis=0)
                        yield [contexts, courses_x], matches
                    
    def valid_generator(self): 
        negative_courses = self.crs_df['agg_id'].map(self.course_to_id)
        n_neg = len(negative_courses)
        while True:
            for (student, term), df in self.crs_df_valid.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
                self.courses_set = make_set(df)
                if len(self.courses_set) > 1:
                    for crs_1 in self.courses_set:
                        contexts = []
                        courses_x = []
                        matches = []
                        for crs_2 in self.courses_set: 
                            x = crs_1
                            y = crs_2
                            if x!=y:
                                context = list(negative_courses.iloc[np.random.choice(n_neg,4)]) + [y]
                                course = 5*[x]
                                match = [0,0,0,0,1]
                                contexts.append(np.array(context).reshape(5,1))
                                courses_x.append(np.array(course).reshape(5,1))
                                matches.append(np.array(match).reshape(5,1))
                        contexts = np.concatenate(contexts, axis=0)
                        courses_x = np.concatenate(courses_x, axis=0)
                        matches = np.concatenate(matches, axis=0)
                        yield [contexts, courses_x], matches
                        