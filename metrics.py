import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def draw_plot(layers, scores, labels, path):
    x_values = layers

    # 创建图形
    plt.figure(figsize=(10, 6))

    for i in range(len(scores)):
        score = scores[i]
        label = labels[i]
        plt.plot(x_values, score, label=label, marker='.')
            
    # 设置图形标题和坐标轴标签
    plt.xlabel('Layers')
    plt.ylabel('Scores')
    # 设置横轴范围
    plt.xlim(0, 40)
    # 添加图例
    plt.legend()
    plt.savefig(path)
    plt.close()

def draw_acc(layers, scores, label, path):
    x_values = layers
    plt.plot(x_values, scores, label=label, marker='.')
    plt.xlabel('Layers')
    plt.ylabel('Acc')
    # 设置横轴范围
    plt.xlim(0, 40)
    # 添加图例
    plt.legend()
    plt.savefig(path)
    plt.close()

def draw_heat(layers, index, scores, path):
    sns.set()
    # if type == 'std':
    #     vmin = 0
    #     vmax = 10
    #     center = 5
    # elif type == 'mean':
    #     vmin = -10
    #     vmax = 10
    #     center = 0
    ax=sns.heatmap(scores, cmap="RdBu_r", center=0, xticklabels=layers, yticklabels=index)
    # ax=sns.heatmap(scores, cmap="RdBu_r", xticklabels=layers, yticklabels=index)
    plt.xticks(size = 4)
    plt.savefig(path)
    plt.close()
  

def draw_line_plot(x_range, results, labels, path):
    layers = []
    scores = []
    tags = []
    for i in range(len(results)):
        scores += results[i]
        layers += x_range * len(labels)
        for i in range(len(labels)): 
            tags += [labels[i]] * len(x_range)

    data_plot = pd.DataFrame({"layers":layers, "scores":scores, "tags":tags})
    sns.lineplot(x = "layers", y = "scores", hue='tags', data=data_plot)
    plt.savefig(path)
    plt.close()
# # def extrat_gsm8k(text):
# #     text = text.s
# #     der data process-for-prediction(self, text):text a text.split(' nin')leltext = text,split()::-1]flag m False
# # ret  ..for i in range( len(text)):s = text[i]for i in range(len[s)):if sil.isdigit():flag = True
# # ret as
# # break
# # if flag:break
# # ret1 for i in range(lenret)):1f ret[11.1sdigit():retl +m ret(i]return retl  
# results = [0.19, 0.18, 0.16, 0.13, 0.14, 0.21, 0.18, 0.2, 0.2]

def draw_attr_heat(scores, x_tokens, y_tokens, path):
    
    ax=sns.heatmap(scores, cmap="RdBu_r", center=0, xticklabels=x_tokens, yticklabels=y_tokens)
    # ax=sns.heatmap(scores, cmap="RdBu_r", xticklabels=layers, yticklabels=index)
    plt.xticks(size = 2)
    plt.yticks(size = 2)
    plt.savefig(path)
    plt.close()