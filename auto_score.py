# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:58:44 2018

@author: leiwen
"""


import time
start = time.time()
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import FeatureUnion
import xgboost as xgb
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

test = pd.read_csv("sample_test.csv",index_col="id",#low_memory=False,
                    parse_dates=["project_submitted_datetime"])#.sample(1000,random_state=23)

agg_rc = pickle.load(  open( "agg_rc.p", "rb" ) )

test = pd.merge(test,agg_rc, left_index=True, right_index=True, how= "left")

test['text'] = test.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4'])]), axis=1)


text_cols = ["text","project_resource_summary", "project_title", "description"]

# Sentiment Build
print("Hand Made Text Features..")
SIA = SentimentIntensityAnalyzer()
for cols in text_cols:
    test[cols] = test[cols].astype(str) # Make Sure data is treated as string
    test[cols] = test[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    test[cols+'_num_chars'] = test[cols].apply(len) # Count number of Characters
    test[cols+'_num_words'] = test[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    test[cols+'_num_unique_words'] = test[cols].apply(lambda comment: len(set(w for w in comment.split())))
    # Count Unique Words
    test[cols+'_words_vs_unique'] = test[cols+'_num_unique_words'] / test[cols+'_num_words']*100
    # Unique words to Word count Ratio
    if cols == "text":
        test[cols+"_vader_Compound"]= test[cols].apply(lambda x:SIA.polarity_scores(x)['compound'])
        test[cols+'_vader_Negative']= test[cols].apply(lambda x:SIA.polarity_scores(x)['neg'])
        test[cols+'_vader_Positive']= test[cols].apply(lambda x:SIA.polarity_scores(x)['pos'])
        # Test a Stemmer..
    print("{} Done".format(cols))




test["Date_Cutoff"] = None
test.loc[test["project_submitted_datetime"] > "05/16/2016","Date_Cutoff"] = "After"
test.loc[test["project_submitted_datetime"] <= "05/16/2016","Date_Cutoff"] = "Before"

# Time Variables
test["Year"] = test["project_submitted_datetime"].dt.year
test["Date of Year"] = test['project_submitted_datetime'].dt.dayofyear # Day of Year
test["Weekday"] = test['project_submitted_datetime'].dt.weekday
test["Weekd of Year"] = test['project_submitted_datetime'].dt.week
test["Day of Month"] = test['project_submitted_datetime'].dt.day
test["Quarter"] = test['project_submitted_datetime'].dt.quarter


# Split the strings at the comma, and treat them as dummies
test = pd.merge(test, test["project_subject_categories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")
test = pd.merge(test, test["project_subject_subcategories"].str.get_dummies(sep=', '),
              left_index=True, right_index=True, how="left")

teachr_multi_subs = pickle.load( open( "teachr_multi_subs.p", "rb" ) )

# Teacher ID
#teachr_multi_subs = df['teacher_id'].value_counts().reset_index()
test["multi_apps"]= test['teacher_id'].isin(teachr_multi_subs.loc[teachr_multi_subs["teacher_id"]>1,'index'].tolist())
# Percentages
print("Teacher App Distribution:\nTwo Apps: {}%\nOne App: {}%\n".format(*test["multi_apps"].value_counts(normalize=True)*100))


# Teacher Gender
test["Gender"] = None
test.loc[test['teacher_prefix'] == "Mr.","Gender"] = "Male"
test.loc[test['teacher_prefix'] == "Teacher","Gender"] = "Not Specified"
test.loc[(test['teacher_prefix'] == "Mrs.")|(test['teacher_prefix'] == "Ms."),"Gender"] = "Female"

print("Gender Distribution:\nFemale: {}%\nMale: {}%\nNot Specified: {}%".format(*test["Gender"].value_counts(normalize=True)*100))

dumyvars= ["Gender",'school_state','project_grade_category']
timevars = ['Weekday','Weekd of Year','Day of Month','Year','Date of Year',"Quarter"]
encode = ['multi_apps', "Date_Cutoff", 'teacher_prefix',"teacher_id"]
# Decided to go with only encoding, since most of the gradient boosting trees can handle categorical
categorical_features = dumyvars + timevars + encode

test  = pickle.load( open( "sub_df.p", "rb" ) )[:test.shape[0]]

# Text
text_cols = ["project_resource_summary", "project_title","description","text"]

test.drop(['project_subject_categories',"project_subject_subcategories","project_submitted_datetime",
         "project_essay_1","project_essay_2","project_essay_3","project_essay_4"
        ],axis=1,inplace=True)
normalize = ["teacher_number_of_previously_posted_projects","quantity","price"]

# Lets look at these variables!
print("\nDtypes of DF features:\n",test.dtypes.value_counts())
print("\nDF Shape: {} Rows, {} Columns".format(*test.shape))

tfidf_para = {
    "sublinear_tf":True,
    "strip_accents":'unicode',
    "stop_words":"english",
    "analyzer":'word',
    "token_pattern":r'\w{1,}',
    "dtype":np.float32,
    "norm":'l2',
    "min_df":5,
    "max_df":.9,
    "smooth_idf":False
}


def get_col(col_name):
    return lambda x: x[col_name]

test["project_title_count"] = test["project_title"].copy()
textcols = ["text","project_resource_summary","project_title", "project_title_count","description"]


vectorizer = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            **tfidf_para,
            preprocessor=get_col('text'))),
        ('project_resource_summary',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2000,
            preprocessor=get_col('project_resource_summary'))),
        ('project_title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=1500,
            preprocessor=get_col('project_title'))),
        ('project_title_count',CountVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            preprocessor=get_col('project_title_count'))),
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            max_features=2400,
            preprocessor=get_col('description'))),
#         ('Non_text',DictVectorizer())
    ])


ready_df = pickle.load( open( "ready_df.p", "rb" ) )

start_vect=time.time()
ready_df = vectorizer.fit_transform(test.to_dict('records'))
tfvocab = pickle.load( open( "tfvocab.p", "rb" ) )
#tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

test.drop(textcols,axis=1, inplace=True)

# Get categorical var position for CatBoost
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(test,categorical_features)

ready_df = pickle.load( open( "ready_df.p", "rb" ) )

testing = hstack([csr_matrix(test.values),ready_df[182080:182080+test.shape[0]]])

d_test = xgb.DMatrix(testing,feature_names=tfvocab)

model = pickle.load( open( "model.p", "rb" ) )

xgb_pred = model.predict(d_test)

xgb_pred = model.predict(d_test)

xgb_sub = pd.DataFrame(xgb_pred,columns=["project_is_approved"])
xgb_sub.to_csv("xgb_sub.csv",index=True)

print(xgb_sub.head())

