import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Read in the ADMISSIONS and NOTEEVENTS datasets
df_adm = pd.read_csv('ADMISSIONS.csv')
df_notes = pd.read_csv('NOTEEVENTS.csv')

# Convert date columns to datetime format
df_adm['ADMITTIME'] = pd.to_datetime(df_adm['ADMITTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm['DISCHTIME'] = pd.to_datetime(df_adm['DISCHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm['DEATHTIME'] = pd.to_datetime(df_adm['DEATHTIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Sort and reset the admissions dataset by subject ID and admission time
df_adm = df_adm.sort_values(['SUBJECT_ID', 'ADMITTIME'])
df_adm = df_adm.reset_index(drop=True)

# Add columns for the next admission time and type for each subject
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID')['ADMISSION_TYPE'].shift(-1)

# Set the next admission time and type to NaN for elective admissions
rows = df_adm['NEXT_ADMISSION_TYPE'] == 'ELECTIVE'
df_adm.loc[rows, 'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows, 'NEXT_ADMISSION_TYPE'] = np.NaN

# Sort the admissions dataset by subject ID and admission time again
df_adm = df_adm.sort_values(['SUBJECT_ID', 'ADMITTIME'])

# Fill in missing values for next admission time and type using backward fill
df_adm[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']].fillna(method='bfill')

# Calculate the number of days until the next admission
df_adm['DAYS_NEXT_ADMIT'] = (df_adm['NEXT_ADMITTIME'] - df_adm['DISCHTIME']).dt.total_seconds() / (24 * 60 * 60)

# Filter the discharge summary notes to only include those in the DISCHARGE SUMMARY category
df_notes_dis_sum = df_notes.loc[df_notes['CATEGORY'] == 'Discharge summary']

# Select the last discharge summary note for each admission
df_notes_dis_sum_last = (df_notes_dis_sum.groupby(['SUBJECT_ID', 'HADM_ID']).nth(-1)).reset_index()

# Check for duplicate discharge summaries per admission
assert df_notes_dis_sum_last.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'

# Merge the admissions and discharge summary note datasets
df_adm_notes = pd.merge(df_adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME', 'AD
