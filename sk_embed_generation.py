import pandas as pd
import numpy as np

class EmbeddingGenerator ():
    def __init__(self, crs_folder_path, student_attr_folder_path):
        self.crs_folder_path = crs_folder_path
        # load in student attribute info. save to self.student_attributes
        self.student_attr_folder_path = student_attr_folder_path

        self._load_student_attributes()
        self._load_courses()
        self.embedding_id = list(self.crs_df['agg_id'].unique())
        self.course_to_id = dict([(name, i) for i, name in enumerate(self.embedding_id)])
        self.unique_students = list(self.crs_df['PRSN_UNIV_ID'].unique())
        self.name_to_course = dict(self.crs_df[['CRS_NM','agg_id']].values)
        self.embedding = None

        # self.major_id = list(student_attr_df['ENTRY_MAJOR_DESC'].unique())
        # self.major_to_id = dict([(major, i) for i, major in enumerate(self.major_id)])
        # self.intended_school_id = list(student_attr_df['ENTRY_INTENDED_SCHOOL'].unique())
        # self.indended_school_to_id = dict([(pre_major, i) for i, pre_major in enumerate(self.intended_school_id)])


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

    def _load_student_attributes(self):
        student_attributes = pd.read_excel(self.student_attr_folder_path)
        self.student_attr_df = pd.DataFrame(student_attributes)


        self.ethnicity_id = list(self.student_attr_df['ETHNICITY'].unique())
        self.ethnicity_to_id = dict([(ethnicity, i) for i, ethnicity in enumerate(self.ethnicity_id)])
        self.urm_flag_id = list(self.student_attr_df['URM_FLAG'].unique())
        self.urm_flag_to_id = dict([(urm, i) for i, urm in enumerate(self.urm_flag_id)])
        self.gender_id = list(self.student_attr_df['GENDER'].unique())
        self.gender_to_id = dict([(gender, i) for i, gender in enumerate(self.gender_id)])
        self.pell_eligibility_id = list(self.student_attr_df['PELL_ELIGIBILITY'].unique())
        self.pell_status_to_id = dict([(status, i) for i, status in enumerate(self.pell_eligibility_id)])

        self.student_attr_df['ethnicity_id'] = self.student_attr_df['ETHNICITY'].map(self.ethnicity_to_id)
        self.student_attr_df['urm_flag_id'] = self.student_attr_df['URM_FLAG'].map(self.urm_flag_to_id)
        self.student_attr_df['gender_id'] = self.student_attr_df['GENDER'].map(self.gender_to_id)
        self.student_attr_df['pell_eligibility_id'] = self.student_attr_df['PELL_ELIGIBILITY'].map(self.pell_status_to_id)

    def _load_courses(self):
        courses = pd.read_csv(self.crs_folder_path, encoding='latin', low_memory=False)
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

    def set_embedding(self, embedding):
    	self.embedding = embedding


    def demographic_valid_generator(self):
        while True:
            for (student, term), df in self.crs_df_valid.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
                courses_set = self.make_set(df)
                if len(courses_set) > 1:
                    # calculate the vector describing courses
                    x = self.embedding[list(courses_set)] / len(courses_set)

                    # gather demographic info
                    ethnicity = self.student_attr_df[self.student_attr_df['PRSN_UNIV_ID'==student]]['ethnicity_id'] ## and grab row from student attributes that matches row for student var
                    urm_flag = self.student_attr_df[self.student_attr_df['PRSN_UNIV_ID'==student]]['urm_flag_id']
                    gender = self.student_attr_df[self.student_attr_df['PRSN_UNIV_ID'==student]]['gender_id']
                    pell_status = self.student_attr_df[self.student_attr_df['PRSN_UNIV_ID'==student]]['pell_eligibility_id']

                    y = np.array([ethnicity, urm_flag, gender, pell_status])
                    # yield data
                    yield x,y





