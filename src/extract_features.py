import os
import random
import json
import gc
import pickle
import gensim
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def get_agg_features(dfs, group_key, stat_value, agg, log):
    # 判定特殊情况
    if type(group_key) == str:
        group_key = [group_key]
    if agg != 'size':
        data = log[group_key + [stat_value]]
    else:
        data = log[group_key]
    f_name = '_'.join(group_key) + "_" + stat_value + "_" + agg
    # 聚合操作
    if agg == "size":
        tmp = pd.DataFrame(data.groupby(group_key).size()).reset_index()
    elif agg == "count":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].count()).reset_index()
    elif agg == "mean":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].mean()).reset_index()
    elif agg == "unique":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].nunique()).reset_index()
    elif agg == "max":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].max()).reset_index()
    elif agg == "min":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].min()).reset_index()
    elif agg == "sum":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].sum()).reset_index()
    elif agg == "std":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].std()).reset_index()
    elif agg == "median":
        tmp = pd.DataFrame(data.groupby(group_key)[stat_value].median()).reset_index()
    else:
        raise "agg error"

    # 赋值聚合特征
    for df in dfs:
        try:
            del df[f_name]
        except:
            pass
        # 列名加一下当前的group key
        tmp.columns = group_key + [f_name]
        df[f_name] = df.merge(tmp, on=group_key, how='left')[f_name]
    del tmp
    del data
    gc.collect()
    return [f_name]

def sequence_text(dfs, user_feat="user_id", item_feat="ad_id", log:pd.DataFrame=None):
    seq_feature_name = 'sequence_text_' + user_feat + '_' + item_feat
    print(seq_feature_name)
    # 遍历log，获得用户的点击序列
    user_item_dict, items = {}, []
    for (user, item) in log[[user_feat, item_feat]].values:
        #user = item[0] # 比如为user_id
        #value1 = item[1] # 该用户所点击的 ad_id
        try:
            user_item_dict[user].append(str(item))
        except:
            user_item_dict[user] = [str(item)]
    # 将用户所点击的所有序列join起来,形成序列特征
    for user in user_item_dict:
        items.append([user, ' '.join(user_item_dict[user])])

    # 赋值序列特征
    temp = pd.DataFrame(items)
    temp.columns = [user_feat, seq_feature_name] # user_id, seq_text_userid_adid
    temp = temp.drop_duplicates(subset=user_feat) # 以user维度去重

    for df in dfs:
        try:
            del df[seq_feature_name] # 删除原始dataframe中的序列特征
        except:
            pass
        temp.columns = [user_feat] + [seq_feature_name]
        # 以用户为维度进行join, 并生成新的序列特征
        df[seq_feature_name] = df.merge(temp, on=user_feat, how='left')[seq_feature_name]

    gc.collect()
    del temp
    del items
    del user_item_dict
    return [seq_feature_name]

def kfold_stat(train_df, test_df, log_data, pivot):
    # 先对log做kflod统计，统计每条记录中pivot特征的性别年龄分布
    kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
    log = log_data[kfold_features + ['user_id', pivot, 'fold']]
    tmps = []
    for fold in range(6):
        tmp = pd.DataFrame(
            log[(log['fold'] != fold) & (log['fold'] != 5)].groupby(pivot)[kfold_features].mean()).reset_index()
        tmp.columns = [pivot] + kfold_features
        tmp['fold'] = fold
        tmps.append(tmp)

    tmp = pd.concat(tmps, axis=0).reset_index()
    tmp = log[['user_id', pivot, 'fold']].merge(tmp, on=[pivot, 'fold'], how='left')
    del log
    del tmps
    gc.collect()
    # 获得用户点击的所有记录的平均性别年龄分布
    tmp_mean = pd.DataFrame(tmp.groupby('user_id')[kfold_features].mean()).reset_index()
    tmp_mean.columns = ['user_id'] + [f + '_' + pivot + '_mean' for f in kfold_features]
    for df in [train_df, test_df]:
        temp = df.merge(tmp_mean, on='user_id', how='left')
        temp = temp.fillna(-1)
        for f1 in [f + '_' + pivot + '_mean' for f in kfold_features]:
            df[f1] = temp[f1]
        del temp
        gc.collect()
    del tmp
    del tmp_mean
    gc.collect()

def kfold_sequence(train_df, test_df, log_data, pivot):
    # 先对log做kflod统计，统计每条记录中pivot特征的性别年龄分布
    kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
    log = log_data[kfold_features + [pivot, 'fold', 'user_id']]
    tmps = []
    for fold in range(6):
        tmp = pd.DataFrame(
            log[(log['fold'] != fold) & (log['fold'] != 5)].groupby(pivot)[kfold_features].mean()).reset_index()
        tmp.columns = [pivot] + kfold_features
        tmp['fold'] = fold
        tmps.append(tmp)
    tmp = pd.concat(tmps, axis=0).reset_index()
    tmp = log[[pivot, 'fold', 'user_id']].merge(tmp, on=[pivot, 'fold'], how='left')
    tmp = tmp.fillna(-1)
    tmp[pivot + '_fold'] = tmp[pivot] * 10 + tmp['fold']
    del log
    del tmps
    gc.collect()
    # 获得用户点击记录的年龄性别分布序列
    tmp[pivot + '_fold'] = tmp[pivot + '_fold'].astype(int)
    kfold_sequence_features = sequence_text([train_df, test_df], 'user_id', pivot + '_fold', tmp)
    tmp = tmp.drop_duplicates([pivot + '_fold']).reset_index(drop=True)
    # 对每条记录年龄性别分布进行标准化
    kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
    ss = StandardScaler()
    ss.fit(tmp[kfold_features])
    tmp[kfold_features] = ss.transform(tmp[kfold_features])
    for f in kfold_features:
        tmp[f] = tmp[f].apply(lambda x: round(x, 4))
        # 将每条记录年龄性别分布转成w2v形式的文件
    with open('data/sequence_text_user_id_' + pivot + '_fold' + ".{}d".format(12), 'w') as f:
        f.write(str(len(tmp)) + ' ' + '12' + '\n')
        for item in tmp[[pivot + '_fold'] + kfold_features].values:
            f.write(' '.join([str(int(item[0]))] + [str(x) for x in item[1:]]) + '\n')
    tmp = gensim.models.KeyedVectors.load_word2vec_format(
        'data/sequence_text_user_id_' + pivot + '_fold' + ".{}d".format(12), binary=False)
    pickle.dump(tmp, open('data/sequence_text_user_id_' + pivot + '_fold' + ".{}d".format(12), 'wb'))
    del tmp
    gc.collect()
    return kfold_sequence_features


if __name__ == "__main__":
    # 读取数据
    click_log = pd.read_pickle('data/click.pkl')
    train_df = pd.read_pickle('data/train_user.pkl')
    test_df = pd.read_pickle('data/test_user.pkl')
    print(click_log.shape, train_df.shape, test_df.shape)
    ################################################################################
    # 获取聚合特征
    print("Extracting aggregate feature...")
    agg_features = []
    agg_features += get_agg_features([train_df, test_df], 'user_id', '', 'size', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'ad_id', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'creative_id', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'advertiser_id', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'industry', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'product_id', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'time', 'unique', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'sum', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'mean', click_log)
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'std', click_log)
    train_df[agg_features] = train_df[agg_features].fillna(-1)
    test_df[agg_features] = test_df[agg_features].fillna(-1)
    print("Extracting aggregate feature done!")
    print("List aggregate feature names:")
    print(agg_features)
    ################################################################################
    # 获取序列特征，用户点击的id序列
    print("Extracting sequence feature...")
    text_features = []
    text_features += sequence_text([train_df, test_df], 'user_id', 'ad_id', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'creative_id', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'advertiser_id', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'product_id', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'industry', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'product_category', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'time', click_log)
    text_features += sequence_text([train_df, test_df], 'user_id', 'click_times', click_log)
    print("Extracting sequence feature done!")
    print("List sequence feature names:")
    print(text_features)

    ################################################################################
    # 获取K折统计特征，求出用户点击的所有记录的年龄性别平均分布
    # 赋值index,训练集为0-4，测试集为5
    print("Extracting Kflod feature...")
    log = click_log.drop_duplicates(['user_id', 'creative_id'])\
        .reset_index(drop=True)
    del click_log
    gc.collect()

    log['cont'] = 1
    train_df['fold'] = train_df.index % 5
    test_df['fold'] = 5
    df = train_df.append(test_df)[['user_id', 'fold']].reset_index(drop=True)
    log = log.merge(df, on='user_id', how='left')
    del df
    gc.collect()

    # 获取用户点击某特征的年龄性别平均分布
    for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        print("Kfold", pivot)
        kfold_stat(train_df, test_df, log, pivot)

    del log
    gc.collect()
    print("Extracting Kflod feature done!")
    ################################################################################
    # 获取K折序列特征,求出用户点击的每一条记录的年龄性别分布
    # 赋值index,训练集为0-4，测试集为5
    print("Extracting Kflod sequence feature...")
    click_log = pd.read_pickle('data/click.pkl')
    log = click_log.reset_index(drop=True)
    del click_log
    gc.collect()
    log['cont'] = 1
    train_df['fold'] = train_df.index % 5
    train_df['fold'] = train_df['fold'].astype(int)
    test_df['fold'] = 5
    df = train_df.append(test_df)[['user_id', 'fold']].reset_index(drop=True)
    log = log.merge(df, on='user_id', how='left')
    # 获取用户点击某特征的年龄性别分布序列
    kfold_sequence_features = []
    for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        print("Kfold sequence", pivot)
        kfold_sequence_features += kfold_sequence(train_df, test_df, log, pivot)
    del log
    gc.collect()
    print("Extracting Kfold sequence feature done!")
    print("List Kfold sequence feature names:")
    print(kfold_sequence_features)
    ################################################################################
    print("Extract features done! saving data...")
    train_df.to_pickle('data/train_user.pkl')
    test_df.to_pickle('data/test_user.pkl')
