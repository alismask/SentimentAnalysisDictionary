# source ../../ys/gitee/myenv_3_10/bin/activate
import pandas as pd
import zhconv
import os
import numpy as np

# 情感词汇本体
def get_emotional_vocabulary_ontology():
    emotional_vocabulary_ontology = pd.read_excel('情感词汇本体/情感词汇本体.xlsx')
    emotional_vocabulary_ontology.head()
    emotional_vocabulary_ontology.columns
    rename_cols = {'词语': 'word', '极性': 'sentiment'}
    emotional_vocabulary_ontology = emotional_vocabulary_ontology.rename(
                                                columns=rename_cols)
    emotional_vocabulary_ontology = emotional_vocabulary_ontology[rename_cols.values()]
    emotional_vocabulary_ontology.loc[
        emotional_vocabulary_ontology['sentiment'].isin([2,3,7]), 'sentiment'] = -1
    emotional_vocabulary_ontology['sentiment'].value_counts()
    # 繁体转简体
    emotional_vocabulary_ontology['word'] = emotional_vocabulary_ontology[
                        'word'].apply(lambda x: zhconv.convert(str(x), 'zh-cn'))
    # 删除缺失值
    emotional_vocabulary_ontology[emotional_vocabulary_ontology['word'].isna()]
    emotional_vocabulary_ontology['source'] = '情感词汇本体'
    return emotional_vocabulary_ontology


# 台湾大学NTUSD简体中文情感词典
def get_ntusd_sentiment():
    ntusd_sentiment = pd.DataFrame()
    for sentiment, sentiment_value in zip(['positive', 'negative'], [1,-1]):
        a = pd.read_csv('台湾大学NTUSD简体中文情感词典/ntusd-{}.txt'.format(sentiment),
                        header=None)
        a.columns = ['word']
        a = a.dropna(how='any')
        a['sentiment'] = sentiment_value
        ntusd_sentiment = pd.concat([ntusd_sentiment, a])
    # 繁体转简体
    ntusd_sentiment['word'] = ntusd_sentiment[
                        'word'].apply(lambda x: zhconv.convert(str(x), 'zh-cn'))
    ntusd_sentiment['source'] = 'NTUSD'
    ntusd_sentiment['sentiment'].value_counts()
    return ntusd_sentiment


# HowNet
def get_hownet_sentiment():
    file_path = '知网Hownet情感词典'
    files = ['负面评价词语（中文）.txt', '负面情感词语（中文）.txt',
            '正面评价词语（中文）.txt', '正面情感词语（中文）.txt']
    hownet_sentiment = pd.DataFrame()
    for file in files:
        a = pd.read_csv(os.path.join(file_path, file), sep='\t')
        a.columns = ['word', 'other']
        if '正面' in file:
            a['sentiment'] = 1
        if '负面' in file:
            a['sentiment'] = -1
        a = a[['word', 'sentiment']]
        hownet_sentiment = pd.concat([hownet_sentiment, a])
    # 繁体转简体
    hownet_sentiment['word'] = hownet_sentiment[
                        'word'].apply(lambda x: zhconv.convert(str(x), 'zh-cn'))
    hownet_sentiment['source'] = 'HowNet'
    hownet_sentiment['sentiment'].value_counts()
    return hownet_sentiment

# 合并多行数据
def join_values_func(df):
    merge_list = df.values
    merge_list = [i for i in merge_list if str(i) != 'nan']
    a = ', '.join(merge_list)
    # print('a: ', a)
    return a

def update_sentiment(x):
    if x >= 1:
        return 1
    if x <= -1:
        return -1
    return x

emotional_vocabulary_ontology = get_emotional_vocabulary_ontology()
ntusd_sentiment = get_ntusd_sentiment()
hownet_sentiment = get_hownet_sentiment()
emotional_vocabulary_ontology.columns
ntusd_sentiment.columns
hownet_sentiment.columns

all_sentiment = pd.concat([emotional_vocabulary_ontology, ntusd_sentiment, hownet_sentiment])
all_sentiment['source'].value_counts()
all_sentiment.to_excel('./all_sentiment.xlsx', index=False)

print(len(all_sentiment))
bb = all_sentiment[['word', 'sentiment']].drop_duplicates()
print(len(bb))
kk = bb['word'].value_counts().reset_index()
kk[kk['word']>1]
# 对情感倾向进一步处理，选用投票法
all_sentiment = all_sentiment.groupby(['word']).agg(
                    sentiment=pd.NamedAgg(column='sentiment', aggfunc=np.sum),
                    source=pd.NamedAgg(column='source', aggfunc=join_values_func),)
all_sentiment = all_sentiment.reset_index()
all_sentiment['sentiment'] = all_sentiment['sentiment'].apply(
                        lambda x: update_sentiment(x))
all_sentiment[all_sentiment['word'].isna()]
all_sentiment.to_excel('./all_sentiment.xlsx', index=False)
all_sentiment['sentiment'].value_counts()
