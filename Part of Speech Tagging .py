#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Import all the libraries
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter


# In[30]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[50]:


data = pd.read_excel('New  Assignment 1 Dataset.xlsx')



# In[51]:


data


# In[52]:


# Function to clean text from symbols

def clean_text(text):
    # Define a dictionary of abbreviations and their expanded forms
    abbreviations = {
        "re": "are",
        # Add more abbreviations and their expansions as needed
    }
    
    # Create a regex pattern that matches the abbreviations as separate words
    pattern = r'\b(?:{})\b'.format('|'.join(re.escape(key) for key in abbreviations.keys()))
    
    # Replace the abbreviations with their expanded forms
    cleaned_text = re.sub(pattern, lambda match: abbreviations.get(match.group(0)), text)
    
    # Remove symbols except for underscore and whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Remove single characters
    cleaned_text = re.sub(r'\b\w{1}\b', '', cleaned_text)
    
    return cleaned_text


# In[53]:


data["Clean_Subject"]= data["Subject"].apply(clean_text)
data["Clean_Content"]= data["Content"].apply(clean_text)
data


# In[54]:


# Function to perform part of speech tagging on a single passage
def pos_tagging(passages):
    pos_tags_list = []
    for passage in passages:
        tokens = word_tokenize(passage)
        pos_tags = nltk.pos_tag(tokens)
        pos_tags_list.append(pos_tags)
    return pos_tags_list


# In[55]:


# Function to find the most common words for a specific POS tag
def most_common_words_for_pos(pos_tags_dataset, pos):
    words = [word.lower() for passage in pos_tags_dataset for word, tag in passage if tag.startswith(pos)]
    word_freq = Counter(words)
    return word_freq.most_common(10)


# In[58]:


# Perform part of speech tagging on the passages and add as a new column 'POS_tags'
data['POS_tags_sub'] = pos_tagging(data['Clean_Subject'])
data['POS_tags_cont'] = pos_tagging(data['Clean_Content'])
data


# In[59]:


most_common_words_for_pos(data['POS_tags_sub'],"NN")


# In[64]:


common_sub_nouns = most_common_words_for_pos(data['POS_tags_sub'], 'NN')
print("10 most common nouns in the subject of the email:")
print(common_sub_nouns)


# In[65]:


common_sub_adj = most_common_words_for_pos(data['POS_tags_sub'], 'JJ')
print("10 most common adjective in the subject of the email:")
print(common_sub_adj)


# In[66]:


common_sub_verbs = most_common_words_for_pos(data['POS_tags_sub'], 'VB')
print("10 most common verb in the subject of the email:")
print(common_sub_verbs)


# In[85]:


common_cont_nouns = most_common_words_for_pos(data['POS_tags_cont'], 'NN')
print("10 most common noun in the content of the email:")
print(common_cont_nouns)


# In[69]:


common_cont_adj = most_common_words_for_pos(data['POS_tags_cont'], 'JJ')
print("10 most common verb in the content of the email:")
print(common_cont_adj)


# In[70]:


common_cont_verb = most_common_words_for_pos(data['POS_tags_cont'], 'VB')
print("10 most common verb in the content of the email:")
print(common_cont_verb)


# In[71]:


print("\nDataFrame with POS tags:")
print(data)


# In[75]:


def collect_pos_tags(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    noun_list = [word for word, pos in pos_tags if pos.startswith('NN')]
    adj_list = [word for word, pos in pos_tags if pos.startswith('JJ')]
    verb_list = [word for word, pos in pos_tags if pos.startswith('VB')]
    return {
        'nouns': noun_list,
        'adjectives': adj_list,
        'verbs': verb_list
    }


# In[76]:


data['pos_tags_sub'] = data['Clean_Subject'].apply(collect_pos_tags)


# In[80]:


noun_adj_verb_sub = pd.DataFrame(data['pos_tags_sub'].tolist())


# In[81]:


noun_adj_verb_sub


# In[82]:


data['pos_tags_cont'] = data['Clean_Content'].apply(collect_pos_tags)
noun_adj_verb_cont = pd.DataFrame(data['pos_tags_cont'].tolist()) = pd.DataFrame(data['pos_tags_cont'].tolist())


# In[84]:


noun_adj_verb_cont


# In[ ]:




