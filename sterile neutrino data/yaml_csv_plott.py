import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

# 设置 NumPy 打印选项，threshold=np.inf 表示不省略任何数据，完整输出
np.set_printoptions(threshold=np.inf, linewidth=1000)

def load_yaml_matrix(file_path):
    """
    读取并解析 HEPData 格式的 YAML 文件中的矩阵数据，
    并将其转化为按列归一化的能量分辨率矩阵（Smearing Matrix）。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 1. 提取 dependent_variables 中的数值
        values = [item['value'] for item in data['dependent_variables'][0]['values']]
        num_values = len(values)
        
        rows, cols = None, None
        
        # 2. 尝试从 independent_variables 中获取矩阵的具体行列数
        if 'independent_variables' in data:
            dims = []
            for var in data['independent_variables']:
                if 'values' in var:
                    dims.append(len(var['values']))
            
            if len(dims) == 2:
                rows, cols = dims[0], dims[1]
                # 校验提取的行列数相乘是否等于总数据量
                if rows * cols != num_values:
                    rows, cols = None, None

        # 3. 如果无法自动推断，则假定其为方阵（Square Matrix）
        if rows is None or cols is None:
            side = int(math.sqrt(num_values))
            if side * side == num_values:
                rows, cols = side, side
                print(f"[{os.path.basename(file_path)}] 自动推断为方阵: {rows}x{cols}")
            else:
                rows, cols = 1, num_values
                print(f"[{os.path.basename(file_path)}] 无法识别行列，已按 1x{num_values} 向量载入")

        # 将一维数据重塑为 2D 迁移矩阵 (Migration Matrix)
        migration_matrix = np.array(values).reshape(rows, cols)
        
        # ===== 1. 轴向修正：原 X/Y 轴读取相反，对此矩阵进行转置处理 (T) =====
        # 转置后，行对应 E_rec (重建能量)，列对应 E_true (真实能量)
        migration_matrix = migration_matrix.T
        
        # ===== 2. 按列归一化：将迁移矩阵转化为能量分辨率矩阵 (Smearing Matrix) =====
        # 沿纵向 axis=0 对每一列 (E_true) 的所有行 (E_rec) 进行求和
        col_sums = np.sum(migration_matrix, axis=0)
        
        # 物理防错：如果某一列全为0（表示该真实能量下没有重建事件），分母设为1.0以防除零错误
        col_sums[col_sums == 0] = 1.0
        
        # 广播相除，使每一列的元素和归一化为 1 (或 0)
        smearing_matrix = migration_matrix / col_sums
        
        return smearing_matrix, smearing_matrix.shape
    
    except Exception as e:
        print(f"解析文件 {os.path.basename(file_path)} 失败: {e}")
        return None, (0, 0)

def matrix_to_python_syntax(matrix):
    """
    将 NumPy 二维数组转换为符合 Python 语法格式的嵌套列表字符串。
    数字之间由逗号分隔，行与行之间也由逗号分隔并换行。
    """
    # 转换为 Python 原生嵌套列表
    nested_list = matrix.tolist()
    
    lines = []
    for row in nested_list:
        # row 是一个一维列表，转换为字符串后默认用逗号分隔元素，如 [0.0, 1.1, 2.2]
        lines.append("    " + str(row))
    
    # 用 ",\n" 连接每一行，并在外层包裹方括号
    return "[\n" + ",\n".join(lines) + "\n]"

def save_and_plot():
    # =============== 自动获取绝对路径 ===============
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前工作目录被自动设为: {current_dir}\n")

    # 定义最新指定的四个文件名
    yaml_filenames = [
        'HEPData-ins1953539-v3-nu_ecc_FC_Energy_Resolution.yaml',
        'HEPData-ins1953539-v3-nu_ecc_PC_Energy_Resolution.yaml',
        'HEPData-ins1953539-v3-nu_muCC_FC_Energy_Resolution.yaml',
        'HEPData-ins1953539-v3-nu_muCC_PC_Energy_Resolution.yaml'
    ]
    
    # 完整绝对路径列表
    yaml_files = [os.path.join(current_dir, fname) for fname in yaml_filenames]

    all_smearing_matrices = {}
    
    # 创建 2x2 的热力图画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, file_path in enumerate(yaml_files):
        fname_only = os.path.basename(file_path)
        print(f"正在读取并进行归一化转化: {fname_only}...")
        
        # 读取得到的 matrix 已经是经过转置和列归一化处理的标准的 Smearing Matrix
        matrix, shape = load_yaml_matrix(file_path)
        if matrix is None:
            continue
        
        all_smearing_matrices[fname_only] = matrix
        
        # ===== 热力图范围选择机制 =====
        flat_matrix = matrix.flatten()
        
        # 1. 自适应上限 (vmax_adaptive): 最大的 10 个值中的最小值
        if len(flat_matrix) >= 10:
            # 降序排列并提取第 10 个（索引 9）
            vmax_adaptive = np.sort(flat_matrix)[::-1][9]
        else:
            vmax_adaptive = np.max(flat_matrix) if len(flat_matrix) > 0 else 1.0
            
        # 2. 自适应下限 (vmin_adaptive): 最小的 10 个大于0的非零值中的最大值
        positive_elements = flat_matrix[flat_matrix > 0]
        if len(positive_elements) >= 10:
            # 升序排列并提取第 10 个（索引 9）
            vmin_adaptive = np.sort(positive_elements)[9]
        else:
            vmin_adaptive = np.min(positive_elements) if len(positive_elements) > 0 else 1e-4
            
        # 安全守卫：防止异常极值导致下限 >= 上限，使得 LogNorm 报错
        if vmin_adaptive >= vmax_adaptive:
            vmin_adaptive = vmax_adaptive / 10.0 if vmax_adaptive > 0 else 1e-4
            if vmin_adaptive <= 0:
                vmin_adaptive = 1e-4
                vmax_adaptive = 1.0
                
        # 将小于或等于 0 的非正数数值替换为自适应下限 vmin_adaptive，保证对数映射无损渲染
        plot_data = np.where(matrix <= 0, vmin_adaptive, matrix)
        
        # 裁剪图像数据以更好地配合自适应的色彩映射范围
        plot_data = np.clip(plot_data, vmin_adaptive, vmax_adaptive)
        
        # origin='lower' : 让 (0,0) 在左下角，y轴向上，x轴向右
        im = axes[i].imshow(
            plot_data, 
            norm=LogNorm(vmin=vmin_adaptive, vmax=vmax_adaptive), 
            cmap='viridis', 
            aspect='auto',
            origin='lower'
        )
        
        # 为每个子图添加单独的色卡，并标明当前的区间范围
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label(f'Probability Log Scale ({vmin_adaptive:.1e} to {vmax_adaptive:.1e})', rotation=270, labelpad=15)
        
        axes[i].set_title(f"{fname_only}\nSmearing Matrix (Col-Normalized)\nShape: {shape[0]}x{shape[1]} (Origin: Lower-Left)", fontsize=9)
        axes[i].set_xlabel("True Energy Bin $E_{true}$ (Columns)")
        axes[i].set_ylabel("Reconstructed Energy Bin $E_{rec}$ (Rows)")

    # =============== 保存到统一的 CSV 文件 ===============
    csv_output_path = os.path.join(current_dir, 'output_matrices.csv')
    with open(csv_output_path, 'w', encoding='utf-8') as f:
        for name, mat in all_smearing_matrices.items():
            f.write(f"--- Smearing Matrix: {name} ---\n")
            pd.DataFrame(mat).to_csv(f, index=False, header=False)
            f.write("\n")
    print(f"\n[成功] 归一化后的分辨率矩阵已写入 CSV 目标文件: {csv_output_path}")

    # =============== 保存符合 Python 语法的完整矩阵到 TXT 文件 ===============
    txt_output_path = os.path.join(current_dir, 'formatted_matrices.txt')
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("# " + "=" * 25 + " 符合 Python 数组/列表语法的能量分辨率矩阵（Smearing Matrix）无省略输出 " + "=" * 25 + "\n")
        f.write("# 这些矩阵已经过转置（行=Erec, 列=Etrue）和按列归一化处理（每列之和为1或0）\n\n")
        for name, mat in all_smearing_matrices.items():
            f.write(f"\n# [源文件名]: {name}\n")
            f.write(f"# [矩阵维度]: {mat.shape[0]}x{mat.shape[1]} (行: 重建能量Bins, 列: 真实能量Bins)\n")
            f.write(f"# 直接复制下方变量即可在 Python 中定义该二维数组：\n")
            
            # 清理文件名中的非法变量字符以生成干净的 Python 变量名
            var_name = "smearing_matrix_" + name.replace('-', '_').replace('.', '_')
            f.write(f"{var_name} = ")
            
            # 写入符合 Python 语法的嵌套列表字符串
            py_matrix_str = matrix_to_python_syntax(mat)
            f.write(py_matrix_str)
            f.write("\n\n" + "#" * 80 + "\n")
            
    print(f"[成功] 符合 Python 语法且无断行省略的 Smearing Matrix 已保存至 TXT 文件: {txt_output_path}")

    # =============== 在终端打印提示 ===============
    print("\n" + "=" * 25 + " 打印矩阵转换概览 " + "=" * 25)
    for name, mat in all_smearing_matrices.items():
        print(f"\n[文件名]: {name}")
        # 抽取第一列的非零元素之和进行简单科学性验证
        col_sum_check = np.sum(mat, axis=0)
        valid_cols = col_sum_check[col_sum_check > 1e-5]
        avg_sum = np.mean(valid_cols) if len(valid_cols) > 0 else 0.0
        print(f" -> 状态: 成功转化为能量分辨率矩阵 (维度: {mat.shape[0]}x{mat.shape[1]})")
        print(f" -> 验证: 有效物理列的归一化均值和为: {avg_sum:.4f} (应极度接近 1.0)")
    print("\n" + "=" * 68)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    save_and_plot()