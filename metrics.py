import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def draw_plot(layers, scores, labels, path):
    x_values = layers

    # 创建图形
    plt.figure(figsize=(10, 6))

    for i in range(len(scores)):
        score = scores[i]
        label = labels[i]
        plt.plot(x_values, score, label=label, marker='o')
            
    # 设置图形标题和坐标轴标签
    plt.xlabel('Layers')
    plt.ylabel('Scores')
    # 设置横轴范围
    plt.xlim(0, 40)
    # 添加图例
    plt.legend()
    plt.savefig(path)
    

def draw_acc(layers, scores, label, path):
    x_values = layers
    plt.plot(x_values, scores, label=label, marker='o')
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
    ax=sns.heatmap(scores, vmin=-10,vmax=20, cmap="RdBu_r", center=0, xticklabels=layers, yticklabels=index)
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