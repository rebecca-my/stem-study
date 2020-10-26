import pandas as pd
import numpy as np

class EmbeddingGenerator ():
    def __init__(self, folder_path):
        self.folder_path = folder_path

        self._load_courses()
        self.embedding_id = list(self.crs_df['agg_id'].unique())
        self.course_to_id = dict([(name, i) for i, name in enumerate(self.embedding_id)])
        self.unique_students = list(self.crs_df['PRSN_UNIV_ID'].unique())

        np.random.seed(9)
        np.random.shuffle(self.unique_students)
        self.n_train = int(0.1*len(self.unique_students))
        self.n_valid = int(0.02*len(self.unique_students))
        self.train_students = self.unique_students[:self.n_train]
        self.valid_students = self.unique_students[self.n_train : self.n_train + self.n_valid]
        self.crs_df_train = self.crs_df[self.crs_df['PRSN_UNIV_ID'].isin(self.train_students)]
        self.crs_df_valid = self.crs_df[self.crs_df['PRSN_UNIV_ID'].isin(self.valid_students)]

    def make_set(self, df):
        return set(df['agg_id'].map(self.course_to_id))

    def _load_courses(self):
        courses = pd.read_csv(self.folder_path, encoding='latin')
        mask_type = courses['CRS_TYPE']=='ENRL'
        discarded_grades = ['ZZ']
        mask_grade = ~courses['CRS_OFCL_GRD_CD'].isin(discarded_grades)                          
        mask = mask_grade&mask_type
        crs_embed = courses[mask]
        
        self.crs_df = pd.DataFrame(crs_embed)
        self.crs_df['agg_id'] = self.crs_df['CRS_ID'].astype(str)


        subj_by_id = self.crs_df['agg_id'].value_counts()
        big_subj_by_id = subj_by_id[subj_by_id>10].index
        self.crs_df['agg_id'] = np.where(self.crs_df['agg_id'].isin(big_subj_by_id), self.crs_df['agg_id'],'not_in_final_results')

        orphan_classes = self.crs_df[self.crs_df['agg_id']=='not_in_final_results']
        crs_dept_cd = orphan_classes['CRS_SUBJ_DEPT_CD'].value_counts()
        big_dept = crs_dept_cd[crs_dept_cd>10].index
        self.crs_df['agg_id'] = np.where(self.crs_df['agg_id'] != 'not_in_final_results', self.crs_df['agg_id'],
                                                np.where(self.crs_df['CRS_SUBJ_DEPT_CD'].isin(big_dept),
                                                self.crs_df['CRS_SUBJ_DEPT_CD'], 'other_dept'))

    def course_to_label(self, course):
        return self.crs_df[self.crs_df['agg_id']==course].iloc[0]['CRS_NM']


    def train_generator(self): 
        negative_courses = self.crs_df['agg_id'].map(self.course_to_id)
        n_neg = len(negative_courses)
        while True:
            for (student, term), df in self.crs_df_train.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
                courses_set = self.make_set(df)
                if len(courses_set) > 1:
                    for crs_1 in courses_set:
                        contexts = []
                        courses_x = []
                        matches = []
                        for crs_2 in courses_set: 
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
                courses_set = self.make_set(df)
                if len(courses_set) > 1:
                    for crs_1 in courses_set:
                        contexts = []
                        courses_x = []
                        matches = []
                        for crs_2 in courses_set: 
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
