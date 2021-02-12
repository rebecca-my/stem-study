import pandas as pd
import numpy as np

class EmbeddingGenerator ():
    def __init__(self, crs_folder_path, student_attr_folder_path, student_retention_folder_path, stem_majors_folder_path, student_raw_major_folder_path):
        self.crs_folder_path = crs_folder_path
        self.student_attr_folder_path = student_attr_folder_path
        self.student_retention_folder_path = student_retention_folder_path
        self.stem_majors_folder_path = stem_majors_folder_path
        self.student_raw_major_folder_path = student_raw_major_folder_path

        self._load_stem_majors()
        self._load_raw_majors()
        self._load_student_retention()
        self._load_student_attributes()
        self._create_attribute_retention_major_table()
        self._load_courses()
        self.embedding_id = list(self.crs_df['agg_id'].unique())
        self.course_to_id = dict([(name, i) for i, name in enumerate(self.embedding_id)])
        
        self.crs_df['embedding_index'] = self.crs_df['agg_id'].map(self.course_to_id)
        self.unique_students = list(self.crs_df['PRSN_UNIV_ID'].unique())
        self.name_to_course = dict(self.crs_df[['CRS_NM','agg_id']].values)
        self.crs_id_to_course_name = dict(self.crs_df[['agg_id', 'CRS_NM']].values)
        self.embedding = None

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

    def _load_stem_majors(self):
        stem_majors = pd.read_csv(self.stem_majors_folder_path)
        self.stem_major_df = pd.DataFrame(stem_majors)


    def _load_student_retention(self):
        student_retention = pd.read_csv(self.student_retention_folder_path)
        self.stu_retention_df = pd.DataFrame(student_retention)

        temp_stu_retention_major = pd.merge(self.stu_raw_major_df, self.stu_retention_df.reset_index(), how = 'left',
                                                left_on = 'PRSN_UNIV_ID', right_on = 'PRSN_UNIV_ID')

        def retention_helper(df):
            last_major = df[['BA_DEGREE_RECEIVED','MAJOR_1_DESCRIPTION','MAJOR_2_DESCRIPTION']].ffill().iloc[-1]
            first_major = df[['MAJOR_1_DESCRIPTION','MAJOR_2_DESCRIPTION']].bfill().iloc[0].rename({'MAJOR_1_DESCRIPTION' : 'INIT_MAJOR1',
                                                                                                    'MAJOR_2_DESCRIPTION' : 'INIT_MAJOR2'})
                                                                                                                        
            return pd.concat([first_major, last_major])
       
        self.stu_retention_major_df = temp_stu_retention_major.groupby('PRSN_UNIV_ID').apply(retention_helper)

        self.stu_retention_major_df['STEM_end'] = np.logical_or(self.stu_retention_major_df['MAJOR_1_DESCRIPTION'].isin(self.stem_major_df['STEM MAJORS']),
                                                                self.stu_retention_major_df['MAJOR_2_DESCRIPTION'].isin(self.stem_major_df['STEM MAJORS']))

    def _load_raw_majors(self):
        raw_major = pd.read_csv(self.student_raw_major_folder_path)
        self.stu_raw_major_df = pd.DataFrame(raw_major)

    def _create_attribute_retention_major_table(self):
        self.stu_attr_retention_major_df = pd.merge(self.student_attr_df, self.stu_retention_major_df.reset_index(), how='left',
                                                    left_on='PRSN_UNIV_ID', right_on='PRSN_UNIV_ID')

        i = self.stu_attr_retention_major_df['ENTRY_MAJOR_DESC'] == self.stu_attr_retention_major_df['MAJOR_1_DESCRIPTION']
        j = self.stu_attr_retention_major_df['ENTRY_MAJOR_DESC'] == self.stu_attr_retention_major_df['MAJOR_2_DESCRIPTION']
        #k = self.stu_attr_retention_major_df['ENTRY_MAJOR_DESC'] == self.stu_attr_retention_major_df['MAJOR_3_DESCRIPTION']
        ## are we using this 'MAINTAINED_MAJOR' in a useful capacity? ans: Not at the moment.
        self.stu_attr_retention_major_df['MAINTAINED_MAJOR'] = np.logical_or(i, j)

    def _load_student_attributes(self):
        student_attributes_raw = pd.read_excel(self.student_attr_folder_path, engine='openpyxl')
        discarded_cohort_terms = [4148, 4152, 4158, 4162, 4168, 4172, 4178, 4182, 4188, 4192, 4198, 4202]
        mask_cohort_term_code = ~student_attr_raw['COHORT_TERM_CD'].isin(discarded_cohort_terms)
        student_attributes = student_attr_raw[mask_cohort_term_code]

        self.student_attr_df = pd.DataFrame(student_attributes)
        self.student_attr_df['STEM_start'] = self.student_attr_df['ENTRY_MAJOR_DESC'].isin(self.stem_major_df['STEM MAJORS'])

        self.urm_flag_id = list(self.student_attr_df['URM_FLAG'].unique())
        self.urm_flag_to_id = dict([(urm, i) for i, urm in enumerate(self.urm_flag_id)])
        self.gender_id = list(self.student_attr_df['GENDER'].unique())
        self.gender_to_id = dict([(gender, i) for i, gender in enumerate(self.gender_id)])
        self.pell_eligibility_id = list(self.student_attr_df['PELL_ELIGIBILITY'].unique())
        self.pell_status_to_id = dict([(status, i) for i, status in enumerate(self.pell_eligibility_id)])
       # self.student_attr_df['ethnicity_id'] = self.student_attr_df['ETHNICITY'].map(self.ethnicity_to_id)
        self.student_attr_df['urm_flag_id'] = self.student_attr_df['URM_FLAG'].map(self.urm_flag_to_id)
        self.student_attr_df['gender_id'] = self.student_attr_df['GENDER'].map(self.gender_to_id)
        self.student_attr_df['pell_eligibility_id'] = self.student_attr_df['PELL_ELIGIBILITY'].map(self.pell_status_to_id)

    def _load_courses(self):
        courses_raw = pd.read_csv(self.crs_folder_path, encoding='latin', low_memory=False)
        academic_terms = [4068., 4078., 4108., 4112., 4115., 4118., 4128., 4138., 4142., 4098., 4102., 4072., 4082., 
            4088., 4092., 4095., 4122., 4132., 4068., 4182., 4188., 4202., 4198., 4105., 4125., 4135., 4158., 4185., 
            4172., 4192., 4148., 4152., 4175., 4155., 4162., 4165., 4195., 3962., 3842., 3848., 4178., 3808., 4145., 
            4075., 4015., 4025., 4085., 4060., 4080., 4070., 4050., 4090., 
            4120., 4208., 4140., 4150., 4205., 4110., 4130., 4160., 4170.]
        mask_term = courses_raw['ACAD_TERM_CD'].isin(academic_terms)
        mask_type = courses_raw['CRS_TYPE']=='ENRL'
        discarded_grades = ['ZZ']
        mask_grade = ~courses_raw['CRS_OFCL_GRD_CD'].isin(discarded_grades)                          
        mask = mask_grade&mask_type&mask_term
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


    def get_demographic(self, dataset):
        if dataset == 'train':
            data = self.crs_df_train
        elif dataset == 'valid':
            data = self.crs_df_valid
        else:
            raise ValueError('Must be from valid or train data.')

        x_results = []
        y_results = []
        for (student, term), df in data.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
            courses_set = df['embedding_index']
            if len(courses_set) > 1:
                # calculate the vector describing courses
                x = self.embedding[courses_set].mean(axis=0)
                # gather demographic info
                row = self.student_attr_df[self.student_attr_df['PRSN_UNIV_ID']==student]
                #y = row[['ethnicity_id', 'urm_flag_id', 'gender_id', 'pell_eligibility_id']].values[0]
                y = row[['urm_flag_id', 'gender_id', 'pell_eligibility_id']].values[0]
                #y = row[['gender_id', 'pell_eligibility_id']].values[0]
                x_results.append(x)
                y_results.append(y)
        return np.vstack(x_results), np.vstack(y_results)
   
    def socio_demo_generator_complex(self, dataset, major_filter=None):
        if dataset == 'train':
            data = self.crs_df_train
        elif dataset == 'valid':
            data = self.crs_df_valid
        else:
            raise ValueError('Must be from valid or train data.')
        for (student, term), df in data.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
            df = df[~df['CRS_OFCL_GRD_NBR'].isna()]
            courses_set = df['embedding_index'] 
            if len(courses_set) > 1:        
                row = self.stu_attr_retention_major_df[self.stu_attr_retention_major_df['PRSN_UNIV_ID']==student]        
                #ethnicity_onehot = np.eye(len(self.ethnicity_id))[row['ethnicity_id']][0]
                urm_onehot = np.eye(len(self.urm_flag_id))[row['urm_flag_id']][0]
                gender_onehot = np.eye(len(self.gender_id))[row['gender_id']][0]
                pell_eligibility = row['pell_eligibility_id']               
                #x = np.concatenate([ethnicity_onehot, urm_onehot, gender_onehot, pell_eligibility])   
                x = np.concatenate([urm_onehot, gender_onehot, pell_eligibility])     
                #x = np.concatenate([gender_onehot, pell_eligibility])    
                student_status = row[['BA_DEGREE_RECEIVED', 'STEM_start', 'STEM_end']].values[0]
                if student_status[1] == 0:
                    continue
                if student_status[0] == 0:
                    y = 0 ## case 0 is degree non-completer
                else:
                    if student_status[2] == 0:
                        y = 1 ## case 1 STEM non-completer
                    else:
                        y = 2 ## case 2 STEM completer
                yield x, y

    def stem_generator_simple(self, dataset, major_filter=None):
        if dataset == 'train':
            data = self.crs_df_train
        elif dataset == 'valid':
            data = self.crs_df_valid
        else:
            raise ValueError('Must be from valid or train data.')

        for (student, term), df in data.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
            courses_set = df['embedding_index']
            if len(courses_set) > 1:
                row = self.stu_attr_retention_major_df[self.stu_attr_retention_major_df['PRSN_UNIV_ID']==student]
                student_status = row[['BA_DEGREE_RECEIVED', 'STEM_start', 'STEM_end']].values[0]
                if student_status[1] == 0:
                    continue ## we continue only if student is STEM_start
                if student_status[0] == 0:
                    y = 0 ## case 0 is dropout
                else:
                    if student_status[2] == 0:
                        y = 1 ## case 1 transitioned out of STEM
                    else:
                        y = 2 ## completed STEM
                x = self.embedding[courses_set].mean(axis=0)
                yield x, y

    def stem_generator_complex(self, dataset, major_filter=None):
        if dataset == 'train':
            data = self.crs_df_train
        elif dataset == 'valid':
            data = self.crs_df_valid
        else:
            raise ValueError('Must be from valid or train data.')
            
        for (student, term), df in data.groupby(['PRSN_UNIV_ID','ACAD_TERM_CD']):
            df = df[~df['CRS_OFCL_GRD_NBR'].isna()]
            df = df[df['ACAD_UNT_TKN_NBR'] > 0]
            courses_set = df['embedding_index']
            if len(courses_set) > 1:
                row = self.stu_attr_retention_major_df[self.stu_attr_retention_major_df['PRSN_UNIV_ID']==student] 
                major = row['INIT_MAJOR1'].values[0]
                if  major_filter is not None and major_filter != major:
                    continue
                df['grade_points'] = df['CRS_OFCL_GRD_NBR']*df['ACAD_UNT_TKN_NBR']

                ave_grade = df['grade_points'].sum()/df['ACAD_UNT_TKN_NBR'].sum()
                var_grade = ((df['ACAD_UNT_TKN_NBR']/df['ACAD_UNT_TKN_NBR'].sum()) * 
                             np.square(df['CRS_OFCL_GRD_NBR']-ave_grade)).sum()
                credit_hrs = df['ACAD_UNT_TKN_NBR'].sum()
                grade_scalars = np.array([ave_grade, var_grade, credit_hrs]).astype(float)
                
                df_low = df[df['CRS_OFCL_GRD_NBR'] <= ave_grade + 0.001]
                df_high = df[df['CRS_OFCL_GRD_NBR'] >= ave_grade - 0.001]
                if len(df_high) == 0:
                    display(df)
                    print(df['CRS_OFCL_GRD_NBR'])
                    print(grade_scalars)
                assert len(df_high) > 0
                assert len(df_low) > 0
                x_low = self.embedding[df_low['embedding_index']].mean(axis=0)
                x_high = self.embedding[df_high['embedding_index']].mean(axis=0)
                       
                #ethnicity_onehot = np.eye(len(self.ethnicity_id))[row['ethnicity_id']][0]
                urm_onehot = np.eye(len(self.urm_flag_id))[row['urm_flag_id']][0]
                gender_onehot = np.eye(len(self.gender_id))[row['gender_id']][0]
                pell_eligibility = row['pell_eligibility_id']

                x = np.concatenate([x_low, x_high, grade_scalars, urm_onehot, gender_onehot, pell_eligibility])
                #x = np.concatenate([x_low, x_high, grade_scalars, gender_onehot, pell_eligibility])
                student_status = row[['BA_DEGREE_RECEIVED', 'STEM_start', 'STEM_end']].values[0]
                if student_status[1] == 0:
                    continue
                if student_status[0] == 0:
                    y = 0 ## case 0 is degree non-completer
                else:
                    if student_status[2] == 0:
                        y = 1 ## case 1 STEM non-completer
                    else:
                        y = 2 ## case 2 STEM completer
                    
                yield x, y
                
                if urm_onehot[0] == 0 and dataset == 'train': ## oversampling non-whites
                    yield x, y