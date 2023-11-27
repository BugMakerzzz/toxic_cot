import matplotlib.pyplot as plt
import numpy as np

def draw(layers, scores, labels, path):
    x_values = layers

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    for i in range(len(scores)):
        score = scores[i]
        label = labels[i]
        plt.plot(x_values, score, label=label, marker='o')

    # 设置图形标题和坐标轴标签
    plt.xlabel('Layers')
    plt.ylabel('Scores')

    # 设置横轴范围
    plt.xlim(1, 40)

    # 添加图例
    plt.legend()


    plt.savefig(path)