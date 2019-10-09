# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('dataset.csv', index_col=0)
brut = pd.read_excel('CommercialsAndPrizes.xlsx', index_col=0)

brut.info()




to_drop = ['id','campaignFriendlyName', 'campaignTitle', 'cmp_campaignFriendlyName', 'cmp_campaignTitle', \
                'title', 'submission_id', 'isGrandPrix', 'category', \
                'analysis_url', 'error', 'track_href', 'campaignDescription', 'outcome', 'clientBriefOrObjective',\
                'execution', 'implementation','annotations','lines', 'isTitaniumAndIntegratedSection',\
                'cmp_campaignId', 'count_of_archiveEntryUrls','campaignId', 'submissionId', 'isCampaignAward',\
                'friendlyName','mainMedia_mediaType','uri','type','count_of_downloadMedia', 'count_of_otherMedia',\
                'contentModeration_bannedWordsCount', 'contentModeration_bannedWordsRatio', 'contentModeration_isAdult', \
                'contentModeration_isSuspectedAsAdult', 'hasVideo',]
                
df = df.drop(to_drop,axis=1)


fill_0 = ['Bronze', 'Silver', 'Shortlist', 'Gold','Grand Prix']
for x in fill_0 :
    df['{}'.format(x)].fillna(0,inplace=True)

df = df.dropna(subset=['award_score'])

df = df.dropna()
df_audio = df[(df.spotify_status == 'Found')]
df_audio = df_audio.drop('spotify_status',axis=1)

df_audio_ok = df_audio[df_audio.award_score != 0] 
df_audio_ok['Loser'] = 0
df_audio_nok = df_audio[df_audio.award_score == 0] 
df_audio_nok['Loser'] = 1
df_audio_nok = df_audio_nok.sample(n=1000,random_state=42)
df_audio = pd.concat([df_audio_ok, df_audio_nok], ignore_index=True)
corr = df_audio.corr()


def get_high_cor_colls(df, threshold) :
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop


corr_threshold = 0.80
to_drop2 = get_high_cor_colls(df_audio, corr_threshold)
df_audio2 = df_audio.drop(to_drop2, axis=1)




high_var_cols = ['duration_ms','audioEffect_HandClaps_duration', 'audioEffect_Silence_duration',\
                 'durationInSeconds' , 'n_faces']

df_audio2[high_var_cols] = df_audio2[high_var_cols].apply(lambda row: np.log(row + 1))


stopwords = ['your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', \
"she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', \
'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', \
'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',\
'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',\
'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',\
'while', 'of', 'at', 'one', 'about', 'ad', 'all', 'by', 'can','day','de', 'every',\
'first', 'for','from','get', 'how','in', 'into', 'my', 'no', 'not', 'on','out','over',\
'so','than','to','up','us','we','will','with','you', 'like','more','our']


tfidf_vec = TfidfVectorizer(max_features = 1000)
ocr_tfidf = tfidf_vec.fit_transform(df_audio2.ocr)
ocr_tfidf_array = ocr_tfidf.toarray()
ocr_tfidf_df = pd.DataFrame(ocr_tfidf_array)
ocr_tfidf_features = tfidf_vec.get_feature_names()
ocr_tfidf_df.columns = ocr_tfidf_features


for word in stopwords :
    for col in ocr_tfidf_features :
        if word == col :
            ocr_tfidf_df= ocr_tfidf_df.drop('{}'.format(word), axis=1)
            
ocr_tfidf_df.drop([col for col, val in ocr_tfidf_df.sum().iteritems() if val < 25], axis=1, inplace=True)             


tfidf_vec = TfidfVectorizer(max_features = 1000)
all_text_tfidf = tfidf_vec.fit_transform(df_audio2.all_text)
all_text_tfidf_array = all_text_tfidf.toarray()
all_text_tfidf_df = pd.DataFrame(all_text_tfidf_array)
all_text_tfidf_features = tfidf_vec.get_feature_names()
all_text_tfidf_df.columns = all_text_tfidf_features

for word in stopwords :
    for col in all_text_tfidf_features :
        if word == col :
            all_text_tfidf_df = all_text_tfidf_df.drop('{}'.format(word), axis=1)

all_text_tfidf_df.drop([col for col, val in all_text_tfidf_df.sum().iteritems() if val < 50], axis=1, inplace=True) 

overlap = ['new','people','world']
all_text_tfidf_df['at_new']= all_text_tfidf_df['new']
all_text_tfidf_df['at_people']= all_text_tfidf_df['people']
all_text_tfidf_df['at_world']= all_text_tfidf_df['world']
all_text_tfidf_df = all_text_tfidf_df.drop(overlap, axis=1)

df_audio2.reset_index(drop=True, inplace=True)



df_text = df_audio2.join(ocr_tfidf_df)
df_text = df_text.join(all_text_tfidf_df)

df_text = df_text.drop('ocr',axis=1)
df_text = df_text.drop('all_text',axis=1)





to_dummies = ['agencies', 'categoryCode', 'client', 'festivalYear', 'mediaCode', 'mediaDescription', 'section']

for x in to_dummies:
    new_subset = pd.get_dummies(df_text['{}'.format(x)], prefix=x, drop_first=True)
    new_subset = new_subset.drop(new_subset.sum().idxmax(),axis=1)
    df_text = df_text.join(new_subset).drop(x,axis=1)

df_text.drop([col for col, val in df_text.sum().iteritems() if val < 50], axis=1, inplace=True) 



# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20,16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
#
#corr2 = df_text.corr()
#mask2 = np.zeros_like(corr2, dtype=np.bool)
#mask2[np.triu_indices_from(mask2)] = True
#
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
to_drop3 = get_high_cor_colls(df_text, corr_threshold)
df_final = df_text.drop(to_drop3, axis=1)   

#df_final['Gold'] = df_audio2.Gold 
df_final['Grand Prix'] = df_audio2['Grand Prix']



print(df_final.info())
df_final.to_csv('data_clean.csv')