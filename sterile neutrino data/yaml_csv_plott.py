import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

def load_hep_matrix(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 1. 提取矩阵数值 (Dependent Variables)
        values = [item['value'] for item in data['dependent_variables'][0]['values']]
        num_values = len(values)
        
        # 2. 尝试从 Independent Variables 获取维度
        rows, cols = None, None
        if 'independent_variables' in data:
            # 统计独立变量的 bin 数量
            dims = []
            for var in data['independent_variables']:
                dims.append(len(var['values']))
            
            if len(dims) == 2:
                # 典型的 2D 矩阵
                rows, cols = dims[0], dims[1]
                # 校验：总数是否匹配
                if rows * cols != num_values:
                    rows, cols = None, None

        # 3. 如果无法从元数据获取，则尝试自动推断
        if rows is None or cols is None:
            side = int(math.sqrt(num_values))
            if side * side == num_values:
                rows, cols = side, side
                print(f"[{file_path}] 推断为方阵: {rows}x{cols}")
            else:
                rows, cols = 1, num_values
                print(f"[{file_path}] 无法推断维度，识别为 1x{num_values} 向量")

        matrix = np.array(values).reshape(rows, cols)
        return matrix, (rows, cols)
    
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return None, (0, 0)

def save_and_plot(file_list):
    all_data = {}
    
    # 创建 2x2 绘图画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, f_path in enumerate(file_list):
        matrix, shape = load_hep_matrix(f_path)
        if matrix is None: continue
        
        all_data[f_path] = matrix
        
        # 处理 0 值或负数（对数坐标不允许非正数）
        plot_data = np.where(matrix <= 0, 1e-10, matrix)
        
        # origin='lower' 让 (0,0) 位于左下角，y轴向上，x轴向右
        # norm=LogNorm(vmin=1e-1, vmax=1e3) 限制对数范围在 10^-1 到 10^3 之间
        im = axes[i].imshow(
            plot_data, 
            norm=LogNorm(vmin=1e-1, vmax=1e3), 
            cmap='viridis', 
            aspect='auto',
            origin='lower'
        )
        
        fig.colorbar(im, ax=axes[i], label='Log Scale Value (10^-1 to 10^3)')
        axes[i].set_title(f"{f_path}\nShape: {shape[0]}x{shape[1]} (Origin: Lower-Left)")
        axes[i].set_xlabel("X-axis")
        axes[i].set_ylabel("Y-axis")

    # 保存为带源文件表头的统一 CSV 文件
    with open('output_matrices.csv', 'w') as f:
        for name, mat in all_data.items():
            f.write(f"--- Source: {name} ---\n")
            pd.DataFrame(mat).to_csv(f, index=False, header=False)
            f.write("\n")

    # 在终端以矩阵形式打印全部四个矩阵
    print("\n" + "="*20 + " 矩阵打印输出 " + "="*20)
    for name, mat in all_data.items():
        print(f"\n[源文件]: {name}")
        print(f"[维度]: {mat.shape[0]}x{mat.shape[1]}")
        # 使用 numpy 默认输出，保留矩阵排版
        print(mat)
    print("\n" + "="*54)

    plt.tight_layout()
    plt.show()

# 使用示例（如有4个文件，请取消下面两行的注释运行）：
# yaml_files = ['file1.yaml', 'file2.yaml', 'file3.yaml', 'file4.yaml']
# save_and_plot(yaml_files)

    yaml_files = ['HEPData-ins1953539-v3-nu_eCC_FC_Energy_Resolution.yaml', 
                  'HEPData-ins1953539-v3-nu_eCC_PC_Energy_Resolution.yaml', 
                  'HEPData-ins1953539-v3-nu_muCC_FC_Energy_Resolution.yaml', 
                  'HEPData-ins1953539-v3-nu_muCC_PC_Energy_Resolution.yaml']
    
    save_and_plot(yaml_files)
