import numpy as np
from collections import Counter

###################################
# loss functions
# loss_function_1: classification, entropy
def Entropy(y):
    label_dic = Counter(y) # {label: counts}
    n = len(y)
    entropy = sum(-x/n*np.log2(x/n) for x in label_dic.values())
    return entropy

# loss_fucntion_2: classification, gini index
def Gini(y):
    label_dic = Counter(y) # {label: counts}
    n = len(y)
    gini = 1 - sum(np.square(x/n) for x in label_dic.values())
    return gini

# loss_function_3: regression, variance(l2_norm estimated by the mean value)

# loss_function_4: regression, variance by median
def Var_median(y):
    median = np.median(y)
    loss = np.mean((y - median)**2)
    return loss

###################################
# value estimator
# Classification Esitimator: most common vote
def most_common_vote(y):
    label_dict = Counter(y)
    most_common_label = label_dict.most_common(1)[0][0]
    return most_common_label

# Regression Estimator: mean or median

###################################
# Regression Tree




# Classification Tree
