import yaml
import pandas as pd

# 1. 加载 YAML 数据
# 假设你的文件名是 data.yaml
yaml_file = "HEPData-ins1953539-v3-nu_eCC_PC_Efficiency.yaml"

with open(yaml_file, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# 2. 提取自变量 (Independent Variables) - 通常是能量区间
# 取第一个自变量
indep_var = data['independent_variables'][0]
indep_name = indep_var['header']['name']
indep_unit = indep_var['header'].get('units', '')

x_low = [v['low'] for v in indep_var['values']]
x_high = [v['high'] for v in indep_var['values']]

# 3. 提取因变量 (Dependent Variables) - 通常是效率和误差
dep_var = data['dependent_variables'][0]
dep_name = dep_var['header']['name']
dep_unit = dep_var['header'].get('units', '')

y_values = [v['value'] for v in dep_var['values']]

# 提取误差 (Error)，这里假设 label 是 'stat'
# 这里的逻辑是查找每个 value 下 errors 列表里 label 为 stat 的 symerror
y_errors = []
for v in dep_var['values']:
    stat_err = next((err['symerror'] for err in v['errors'] if err.get('label') == 'stat'), 0)
    y_errors.append(stat_err)

# 4. 构建 DataFrame
df = pd.DataFrame({
    f"{indep_name}_low ({indep_unit})": x_low,
    f"{indep_name}_high ({indep_unit})": x_high,
    f"{dep_name} ({dep_unit})": y_values,
    "stat_error": y_errors
})

# 5. 保存为 CSV
output_file = ("microboone_nu_eCC_PC_Efficiency.csv")
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"转换成功！文件已保存为: {output_file}")
print(df.head())