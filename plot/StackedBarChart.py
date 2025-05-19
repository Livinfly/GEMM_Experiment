import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# X轴的类别 (例如图片中的 Sequence Length)
categories = [64, 256, 1024, 4096]

# 每个类别中，各个堆叠部分的数据
# 键是图例名称，值是对应每个类别的高度列表
segment_data = {
    'gemm': np.array([x / 7 for x in [0.913635, 1.623, 5.071, 20.469]]),    # “linear”部分的高度
    'bias': np.array([x / 7 for x in [0.063871, 0.089342, 0.236154, 0.794534]]),   # “attention”部分的高度
    'activation': np.array([x / 6 for x in [0.031264, 0.044447, 0.156505, 0.641872]])       # “others”部分的高度
}
# 堆叠的顺序 (从下往上)
segment_names = ['gemm', 'bias', 'activation'] 

# 为每个部分定义颜色和填充图案 (hatch) 以模仿图片样式
# 您可以根据需要选择更精确的颜色代码
colors = ['#66c2a5', '#a0a0a0', '#fc8d62']  # 青绿色, 灰色, 橙红色
hatches = ['//', None, '\\'] #  'linear'用斜线, 'attention'无填充, 'others'用反斜线

# 2. 开始绘图
fig, ax = plt.subplots(figsize=(8, 8)) # 创建图形和坐标轴，可以调整大小

x_positions = np.arange(len(categories))  # X轴每个柱子的位置 (0, 1, 2...)
bar_width = 0.6  # 柱子的宽度

# 初始化一个数组，用于追踪当前堆叠部分的底部起始高度
bottom_values = np.zeros(len(categories))

# 遍历每个数据部分并绘制
for i, segment_name in enumerate(segment_names):
    heights = segment_data[segment_name] # 当前部分在每个类别上的高度
    
    ax.bar(
        x_positions,          # X轴位置
        heights,              # 当前部分的高度
        bar_width,            # 柱子宽度
        label=segment_name,   # 图例标签
        bottom=bottom_values, # 柱子的起始Y坐标 (关键！)
        color=colors[i],      # 柱子颜色
        edgecolor='black',    # 柱子边缘颜色，使其更清晰
        hatch=hatches[i]      # 柱子的填充图案
    )
    
    # 更新 bottom_values，为下一个部分的堆叠做准备
    bottom_values += heights

# 3. 设置图表的美化属性
ax.set_xlabel("Batch Size", fontsize=14)
ax.set_ylabel("Time (ms)", fontsize=14)
ax.set_title("MLP (Custom Optimized GEMM)", fontsize=16)

# 设置X轴刻度和标签 (模仿图片中的旋转)
ax.set_xticks(x_positions)
ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)

# 设置Y轴范围和网格线 (模仿图片)
# ax.set_ylim(0, 1)
ax.tick_params(axis='y', labelsize=12) # Y轴刻度字体大小
ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Y轴的水平网格线

# 添加图例
ax.legend(title="Component", fontsize=10, title_fontsize=12, loc='upper left')

# 调整布局，防止标签重叠
plt.tight_layout()

# 显示或保存图像
plt.savefig("./result/stacked_bar_chart_mlp.png")
# plt.show()
