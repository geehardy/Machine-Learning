import numpy as np
import pandas as pd
from math import log2
from collections import Counter

# ===================== 1. 构建数据集 =====================
data = [
    ["青年", "否", "否", "一般", "否"],
    ["青年", "否", "否", "好", "否"],
    ["青年", "是", "否", "非常好", "是"],
    ["青年", "是", "是", "一般", "是"],
    ["青年", "否", "否", "一般", "否"],
    ["中年", "否", "否", "一般", "否"],
    ["中年", "否", "否", "好", "否"],
    ["中年", "是", "是", "非常好", "是"],
    ["中年", "否", "是", "非常好", "是"],
    ["老年", "否", "否", "一般", "否"],
    ["老年", "否", "否", "好", "否"],
    ["老年", "是", "是", "非常好", "是"],
    ["老年", "否", "是", "非常好", "是"],
    ["老年", "否", "是", "一般", "否"],
]

df = pd.DataFrame(data, columns=["年龄", "有工作", "有自己的房子", "信贷情况", "类别"])
X = df.drop("类别", axis=1)
y = df["类别"]

# ===================== 2. ID3 决策树 =====================
def entropy(y):
    cnt = Counter(y)
    ent = 0.0
    total = len(y)
    for val in cnt.values():
        p = val / total
        ent -= p * log2(p)
    return ent

def info_gain(X, y, feat_idx):
    base_ent = entropy(y)
    values = X.iloc[:, feat_idx].unique()
    new_ent = 0.0
    for v in values:
        sub_y = y[X.iloc[:, feat_idx] == v]
        new_ent += len(sub_y) / len(y) * entropy(sub_y)
    return base_ent - new_ent

def id3(X, y, features):
    if len(set(y)) == 1:
        return y.iloc[0]
    if len(features) == 0:
        return Counter(y).most_common(1)[0][0]

    best_gain = -1
    best_feat = None
    best_idx = -1
    for i, feat in enumerate(features):
        gain = info_gain(X, y, i)
        if gain > best_gain:
            best_gain = gain
            best_feat = feat
            best_idx = i

    tree = {best_feat: {}}
    values = X.iloc[:, best_idx].unique()
    new_feats = features[:best_idx] + features[best_idx+1:]

    for v in values:
        sub_X = X[X.iloc[:, best_idx] == v]
        sub_y = y[X.iloc[:, best_idx] == v]
        tree[best_feat][v] = id3(sub_X, sub_y, new_feats)
    return tree

# ===================== 3. CART 分类树（Gini） =====================
def gini(y):
    cnt = Counter(y)
    g = 1.0
    total = len(y)
    for v in cnt.values():
        g -= (v/total)**2
    return g

def cart_best_feat(X, y):
    best_gini = float('inf')
    best_feat = None
    best_idx = -1

    for i in range(X.shape[1]):
        values = X.iloc[:, i].unique()
        for v in values:
            ly = y[X.iloc[:, i] == v]
            ry = y[X.iloc[:, i] != v]
            g = (len(ly)/len(y))*gini(ly) + (len(ry)/len(y))*gini(ry)
            if g < best_gini:
                best_gini = g
                best_feat = X.columns[i]
                best_idx = i
    return best_feat, best_idx

def cart(X, y):
    if len(set(y)) == 1:
        return y.iloc[0]
    if X.shape[1] == 0:
        return Counter(y).most_common(1)[0][0]

    feat, idx = cart_best_feat(X, y)
    tree = {feat: {}}
    values = X.iloc[:, idx].unique()

    for v in values:
        subX = X[X.iloc[:, idx] == v]
        suby = y[X.iloc[:, idx] == v]
        tree[feat][v] = cart(subX.drop(feat, axis=1), suby)
    return tree

# ===================== 4. 生成并打印树 =====================
print("="*50)
print("【ID3 决策树】")
id3_tree = id3(X, y, list(X.columns))
print(id3_tree)

print("\n" + "="*50)
print("【CART 决策树】")
cart_tree = cart(X, y)
print(cart_tree)