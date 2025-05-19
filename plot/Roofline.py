import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(peak_gflops, peak_memory_bandwidth_gb_s,
                  kernel_data_points, # 列表，每个元素是 (name, ai, perf_gflops)
                  title="Roofline Plot", 
                  save_path=None,
                  ai_plot_range=(0.01, 1000), # X轴计算强度的绘图范围
                  perf_plot_min_gflops=0.1):  # Y轴性能的最小绘图值
    """
    绘制 Roofline 图。

    Args:
        peak_gflops (float): 硬件的峰值计算性能 (GFLOPS/s)。
        peak_memory_bandwidth_gb_s (float): 硬件的峰值内存带宽 (GB/s)。
        kernel_data_points (list): 一个列表，每个元素是一个元组 
                                   (kernel_name: str, arithmetic_intensity: float, performance_gflops: float)。
        title (str): 图表标题。
        save_path (str, optional): 保存图像的路径。如果为None，则显示图像。
        ai_plot_range (tuple): X轴（计算强度）的显示范围 (min_ai, max_ai)。
        perf_plot_min_gflops (float): Y轴（性能）的最小显示值。
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 1. 绘制屋顶线 ---
    # 计算强度轴，用于绘制屋顶线 (在对数空间生成点)
    ai_axis = np.logspace(np.log10(ai_plot_range[0]), np.log10(ai_plot_range[1]), 100)

    # 内存带宽屋顶: Performance = Bandwidth * AI
    # 单位: GFLOPS/s = (GB/s) * (FLOPs/Byte)
    # 注意: 1 GB/s * 1 FLOP/Byte = 1 GFLOP/s (因为 Giga = 10^9)
    memory_bound_performance = peak_memory_bandwidth_gb_s * ai_axis

    # 计算性能屋顶 (水平线)
    compute_bound_performance = np.full_like(ai_axis, peak_gflops)

    # 实际的性能上限是这两者的较小值
    effective_roof = np.minimum(compute_bound_performance, memory_bound_performance)
    
    # 绘制计算性能屋顶线
    ax.plot(ai_axis, compute_bound_performance, color='red', linestyle='-', linewidth=2, 
            label=f'Peak Compute ({peak_gflops:.1f} GFLOPS/s)')

    # 绘制内存带宽屋顶线
    ax.plot(ai_axis, memory_bound_performance, color='blue', linestyle='-', linewidth=2, 
            label=f'Peak Memory BW ({peak_memory_bandwidth_gb_s:.1f} GB/s)')
            
    # (可选) 填充屋顶下方的区域或用更粗的线描绘有效屋顶
    ax.plot(ai_axis, effective_roof, color='black', linestyle='-', linewidth=3, label='Effective Roofline')


    # --- 2. 绘制核函数性能点 ---
    if kernel_data_points:
        kernel_names = [k[0] for k in kernel_data_points]
        kernel_ais = np.array([k[1] for k in kernel_data_points])
        kernel_perfs = np.array([k[2] for k in kernel_data_points])
        
        ax.scatter(kernel_ais, kernel_perfs, c='green', s=80, marker='o', label='Measured Kernels', zorder=3, alpha=0.7, edgecolors='black')
        for i, name in enumerate(kernel_names):
            ax.annotate(name, (kernel_ais[i], kernel_perfs[i]), 
                        textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)
    else:
        ax.text(0.5, 0.5, "No kernel data provided", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")


    # --- 3. 设置图表属性 ---
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS/s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围，确保所有重要部分可见
    ax.set_xlim(ai_plot_range[0], ai_plot_range[1])
    # Y轴上限应该是峰值计算性能再高一点，下限根据实际数据点或设定的最小值
    min_y_val = perf_plot_min_gflops
    if kernel_data_points:
        valid_perfs = [p for p in kernel_perfs if p > 0] # 过滤掉0或负值，以防对数坐标报错
        if valid_perfs:
            min_y_val = max(perf_plot_min_gflops, np.min(valid_perfs) / 2) # 比最小性能点低一点

    ax.set_ylim(bottom=min_y_val, top=peak_gflops * 2)


    ax.grid(True, which="both", ls="--", alpha=0.5) # "both" 表示主次刻度都画网格
    ax.legend(loc='lower right')

    plt.tight_layout()

    # plt.show()
    plt.savefig(save_path)
    print(f"Roofline plot saved to {save_path}")

# --- 示例用法 ---
if __name__ == '__main__':
    # 假设的硬件参数 (请替换为您实际的硬件参数)
    # 例如一块GPU：峰值FP32计算 15 TFLOPS/s, 内存带宽 750 GB/s
    my_peak_gflops = 5.027 * 1000  # 15 TFLOPS/s = 15000 GFLOPS/s
    my_peak_bw_gbs = 336    # 750 GB/s

    # 假设您通过分析得到了以下核函数的数据点：
    # (名称, 计算强度 FLOPs/Byte, 实测性能 GFLOPS/s)
    my_kernels_data = [
        ("Size 64", 7.76, 80313725490.20 * 1e-9),   # AI=0.5, Perf=250 GFLOPS (0.5 * 750 = 375, 受限于带宽)
        ("Size 256", 33.02, 1224971962616.82 * 1e-9),  # AI=20, Perf=12000 GFLOPS (20 * 750 = 15000, 接近脊点或计算上限)
        ("Size 1024", 37.38, 2023850658946.29 * 1e-9),# AI=50, Perf=14000 GFLOPS (受限于计算)
        ("Size 4096", 25.41, 2105835486662.53 * 1e-9)    # AI=10, Perf=3000 GFLOPS (10 * 750 = 7500, 远低于屋顶，有优化空间)
    ]

    # 如果没有实际数据点，可以先绘制一个空的屋顶图
    # my_kernels_data = [] 

    plot_roofline(
        my_peak_gflops, 
        my_peak_bw_gbs, 
        my_kernels_data,
        title="Roofline Plot (Custom Optimized GEMM, FP32)",
        save_path="./result/roofline.png",
        ai_plot_range=(0.1, 100), # 根据您的数据调整AI绘图范围
        perf_plot_min_gflops=0   # 根据您的数据调整性能绘图范围下限
    )