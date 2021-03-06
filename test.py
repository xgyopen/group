import pandas as pd
import numpy as np
PATH='data/'

def cal_mean():
    from sklearn.cross_validation import KFold
    a=pd.read_csv('data/labels.csv')
    # a[list_classes]=np.exp(np.log(a[list_classes]) -0.5)
    kf = KFold(a.shape[0], n_folds=6, shuffle=False)
    for train_index,valid_index in kf:
        tr_set = a.iloc[train_index]
        valid_set = a.iloc[valid_index]
        print(tr_set.describe())
        print(valid_set.describe())

def get_corr():
    usecol = [
        'toxicity_score_level',
        'quoting_attack_level',
        'recipient_attack_level',
        'third_party_attack_level',
        'other_attack_level',
        'toxicity_level',
        'attack_level',
    ]
    dataset = pd.read_csv(PATH+'clean_train.csv',usecols=usecol)
    print(dataset.corr())



def bagging():
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # PATH = 'results/'
    file_weight = {
        # 'GRU1.csv.gz':6,
        # 'GRU2.csv.gz':6,
        # 'GRU3.csv.gz':6,
        # 'frGRU1.csv.gz':4,
        # 'gloveGRU1.csv.gz':4,
        # 'kernel.csv.gz':26,
        # 'kernel2.csv.gz':10,
        # 'focalloss19.csv.gz':10,
        'base1.gz':1,
        'base2.gz':1,
    }
    output = pd.read_csv('data/sample_submission.csv')
    output[list_classes] = 0
    norm = 0
    for file,weight in file_weight.items():
        result = pd.read_csv(file)
        output[list_classes] += weight*result[list_classes]
        norm +=weight


    output[list_classes] /= norm

    output.to_csv('output.csv.gz',index=False,compression='gzip')


cal_mean()
# post_deal()
# bagging()
# get_corr()