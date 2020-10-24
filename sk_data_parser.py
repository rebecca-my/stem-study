import pandas as pd 
import numpy as np
import os

#course_fname = '/Users/rsciagli/documents/Fall2020/BAR/STU_CRS_TBL_full.csv'

class DataParser ():
    def __init__(self, folder_path):
    #def __init__(self, course_fname):
        self.folder_path = folder_path
        self._load_courses()

    def _load_courses(self):
        courses = pd.read_csv(self.folder_path, encoding='latin')
        #courses = pd.read_csv(course_fname, encoding='latin')
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

        return self.crs_df[['agg_id','PRSN_UNIV_ID','ACAD_TERM_CD']]

    df = _load_courses(self.folder_path)
    #df = _load_courses(course_fname)
    self.crs_df = pd.DataFrame(df)

    self.embedding_id = list(self.crs_df['agg_id'].unique())
    self.course_to_id = dict([(name, i) for i, name in enumerate(self.embedding_id)])

    agg_course_to_dept = dict(zip(self.crs_df['agg_id'], self.crs_df['CRS_SUBJ_DEPT_CD']))
    departments = [agg_course_to_dept[identifier] for identifier in self.embedding_id]

    def make_set(df):
        return set(df['agg_id'].map(self.course_to_id))

