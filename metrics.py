import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import rcParams

rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


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

def draw_heat(index, scores, path,  exp=None, vmax=None):
    sns.set()
    if exp == 'attn':
        ax=sns.heatmap(scores, cmap="RdBu_r", center=0, vmin=0, vmax=vmax, yticklabels=index)
    elif exp == 'mlp':
        ax=sns.heatmap(scores, cmap="BrBG", center=0, vmin=0,  vmax=vmax, yticklabels=index)
    else:
        ax=sns.heatmap(scores, cmap="RdBu_r", center=0)

    ticks = [0, 4, 9, 14, 19, 24, 29, 34, 39]
    tick_labels = [i+1 for i in ticks]
    ticks = [i+0.5 for i in ticks]
    if exp:
    # ax=sns.heatmap(scores, cmap="RdBu_r", xticklabels=layers, yticklabels=index)
        # plt.ylabel('ADE', fontdict={'family' : 'Times New Roman', 'size':22})
        plt.xlabel('Layers', fontdict={'family' : 'Times New Roman', 'size':22})
        plt.yticks(fontproperties = 'Times New Roman', fontsize=20)
        plt.xticks(ticks=ticks, labels=tick_labels, fontproperties = 'Times New Roman', fontsize=20)
        plt.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.15)
    else:
        plt.ylabel('Layers', fontdict={'family' : 'Times New Roman', 'size':22})
        plt.xlabel('Heads', fontdict={'family' : 'Times New Roman', 'size':22})
        plt.yticks(ticks=ticks, labels=tick_labels, fontproperties = 'Times New Roman', fontsize=20)
        plt.xticks(ticks=ticks, labels=tick_labels, fontproperties = 'Times New Roman', fontsize=20)
        plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]
    plt.savefig(path)
    plt.close()
  

def draw_line_plot(x_range, results, labels, path, y_label='Scores'):
    layers = []
    scores = []
    tags = []
    for i in range(len(results)):
        scores += list(results[i])
        layers += x_range * len(labels)
        for i in range(len(labels)): 
            tags += [labels[i]] * len(x_range)
    data_plot = pd.DataFrame({"layers":layers, "scores":scores, "tags":tags})
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles=handles[1:], labels=labels[1:])
    ax = sns.lineplot(x = "layers", y = "scores", hue='tags', data=data_plot)
    # plt.axhline(0, linestyle='--')
    plt.gca().legend().set_title('')
    plt.ylabel(ylabel=y_label, fontdict={'family' : 'Times New Roman', 'size':22})
    plt.xlabel('Layers', fontdict={'family' : 'Times New Roman', 'size':22})
    plt.yticks(fontproperties = 'Times New Roman', fontsize=20)
    plt.xticks(fontproperties = 'Times New Roman', fontsize=20)
    plt.legend(prop={'family' : 'Times New Roman', 'size':22})
    plt.rcParams.update({'legend.fontsize':22})
    if y_label == 'Attr Div':
        plt.axhline(0, linestyle='--', color='k')
        plt.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.15)
    else:
        plt.subplots_adjust(left=0.16, right=0.99, top=0.99, bottom=0.15)
    plt.savefig(path)
    plt.close()

def draw_attr_bar(layers, scores, path):
    score_up = np.where(scores > 0, scores, 0)
    score_down = np.where(scores < 0, scores, 0)

    plt.ylabel('Score', fontdict={'family' : 'Times New Roman'})
    plt.xlabel('Layers', fontdict={'family' : 'Times New Roman'})
    plt.yticks(fontproperties = 'Times New Roman')
    plt.xticks(fontproperties = 'Times New Roman')
        
    plt.xticks([1, 10, 20, 30, 40])
    plt.bar(layers, score_up, width=0.5, color='#EC7063')
    plt.bar(layers, score_down, width=0.5, color='#3498DB')

    
    plt.savefig(path)
    
