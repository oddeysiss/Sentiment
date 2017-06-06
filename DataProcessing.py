# This script cleans the initial ISEAR dataset and saves the modified version as 'Processed_ISEAR.csv'

import pandas as pd
import numpy as np
import os.path
import re
from textblob import TextBlob

# Set up path to original data file.
basepath = os.path.dirname('__file__')
datapath = os.path.abspath(os.path.join(basepath, 'Datasets', 
	'ISEAR_0', 'isear_databank', 'ISEAR_emotion.xlsx'))

# Read in data
df = pd.read_excel(datapath)

# Only want fields ID, Field1, and SIT
df = df[['ID', 'Field1', 'SIT']]
# Rename for clarity
df.columns = ['ID', 'Emotion', 'Story']

# Use regex to clean accented a's paired with \n char and brackets from story text.
def clean_stories(line):
	pattern1 = r'.\n'
	pattern2 = r'^\[ '
	pattern3 = r']$'

	s = re.sub(pattern1, '', line)
	s = re.sub(pattern2, '', s)
	return re.sub(pattern3, '', s)

df['Story'] = df['Story'].apply(clean_stories)

# Convert non-answers to np.nan and drop nan & problem users (who gave no responses).
na_answers = '''Do not know.
No response.
NO RESPONSE.
Doesn't apply.
Nothing.
None.
Not applicable.
Normally I do not feel disgusted.
Can not think of anything just now.
Can not think of any situation.
Can not remember.
Can't think of anything.
I do not remember when I last felf ashamed. I do not usually feel ashamed of what I do.
Not applicable to myself.
Can't remember any episode of disgust.
Blank.
Not included on questionnaire.
Cannot recall the emotion with any force.
Haven't been frightened for ages.
Haven't felt shame for ages.
Cannot remember such a situation.
DO NOT REMEMBER.
No description.
I have not felt this emotion.
Never felt the emotion.
I have not felt this emotion in my life.
NO RESPONSE
I have never felt this emotion.
I have felt shame but am unable to remember any particular incident.
I have not felt this emotion yet.
Never experienced.
Never
There are many instances which are all equally irratating.
Honestly, I have never felt disgust at any situation in my life.
Sorry, I was never ashamed of anything in my life.
I can positively say that I have never done anything that made me feel guilty.
I am quite shameless, not applicable.
Do not remember any incident
Same as in anger.
see answer for "shame".
The same as in shame.
The same as in anger.
The same event described under "shame".
As in sadness (A), relating to this slaughter of fur seals.
The same as in guilt.
The same as in SHAME.'''
na_answers = na_answers.split('\n')

problem_users = [261035, 261058]

def remove_na_answers(line):
	if line in na_answers:
		return np.nan
	return line

def remove_problem_users(row):
	if row['ID'] in problem_users:
		row['Story'] = np.nan
		return row
	return row

df['Story'] = df['Story'].apply(remove_na_answers)
df = df.apply(remove_problem_users, axis=1)

# Drop na from the data
df.dropna(inplace=True)

# Correct spelling errors in the data
df['Story'] = df['Story'].apply(lambda txt: ''.join(TextBlob(txt).correct()))

# Remove the ID column from the dataset - it isn't useful from here on.
df = df[['Emotion', 'Story']]

# Save the dataframe as a new csv file.
df = df.to_csv('Processed_ISEAR.csv')