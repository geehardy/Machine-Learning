import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据初始化
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9.5])
dataset = np.column_stack((x, y))

def calc_mse(y_vals):   # 计算均方误差,y_vals=[a,b,...]
    if len(y_vals) == 0:
        return 0
    mean_y = np.mean(y_vals)
    return np.mean((y_vals - mean_y) ** 2)

def find_best_split(dataset):   # 寻找最优切分点,dataset=[[x1,y1],[x2,y2],...]
    min_mse = float('inf')  # 初始化最小MSE为正无穷大
    best_split = None   # 初始化最优切分点为None
    x_vals = dataset[:, 0]
    split_candidates = (x_vals[:-1] + x_vals[1:]) / 2  # 1.5, 2.5, ... 9.5
    y_vals = dataset[:, 1]
    for s in split_candidates:
        # 划分数据集:D1(x<=s), D2(x>s)
        D1 = dataset[dataset[:, 0] <= s]
        D2 = dataset[dataset[:, 0] > s]
        
        if len(D1) == 0 or len(D2) == 0:
            continue
        mse1 = calc_mse(D1[:, 1])
        mse2 = calc_mse(D2[:, 1])
        weighted_mse = (len(D1)/len(dataset))*mse1 + (len(D2)/len(dataset))*mse2
        
        # 更新最优切分点
        if weighted_mse < min_mse:
            min_mse = weighted_mse
            best_split = s
    
    return best_split, min_mse

def build_cart_reg_tree(dataset, depth, max_depth):
    y_vals = dataset[:, 1]
    if depth >= max_depth:
        mean_y = np.mean(y_vals)
        return {'value': mean_y, 'samples': len(dataset)}
    if len(dataset) <= 1:
        return {'value': np.mean(y_vals), 'samples': len(dataset)}
    
    # 寻找最优切分点
    best_split, _ = find_best_split(dataset)
    D1 = dataset[dataset[:, 0] <= best_split]
    D2 = dataset[dataset[:, 0] > best_split]
    # 递归构建子树
    tree = {
        'split': best_split,          
        'depth': depth,               
        'left': build_cart_reg_tree(D1, depth+1, max_depth),  
        'right': build_cart_reg_tree(D2, depth+1, max_depth) 
    }
    return tree

def predict_reg_tree(x_val, tree):
    if 'value' in tree:  # 叶节点
        return tree['value']
    # 非叶节点,递归查找
    if x_val <= tree['split']:
        return predict_reg_tree(x_val, tree['left'])
    else:
        return predict_reg_tree(x_val, tree['right'])

# 构建回归树
reg_tree = build_cart_reg_tree(dataset, depth=0, max_depth=3)
print("=== CART Regression Tree (Depth=3) Structure ===")
import json
print(json.dumps(reg_tree, indent=2, ensure_ascii=False))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Original Samples', s=80)

# 绘制预测曲线
x_pred = np.linspace(0.5, 10.5, 500)
y_pred = [predict_reg_tree(xi, reg_tree) for xi in x_pred]
plt.plot(x_pred, y_pred, color='blue', linewidth=3, label='Tree Fit (4 层深度)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('CART 回归树(3层深度)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()