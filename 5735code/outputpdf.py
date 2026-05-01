import os
from PIL import Image

# 1. 获取当前脚本所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 设置输入输出
image_folder = current_dir
output_pdf = os.path.join(current_dir, "MU5735FDR_latest_time.pdf")

# 3. 获取所有 PNG 文件并排序
files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith('.png')]

image_list = []

for file in files:
    # 拼合完整路径
    img_path = os.path.join(image_folder, file)
    img = Image.open(img_path)
    
    # 关键点：PNG 通常是 RGBA 模式，PDF 必须转换为 RGB
    # 否则在保存时会报错
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    image_list.append(img)

# 4. 执行合并保存
if image_list:
    # 将第一张图作为入口，后续图片通过 append_images 传入
    image_list[0].save(
        output_pdf, 
        save_all=True, 
        append_images=image_list[1:]
    )
    print(f"成功！使用 Pillow 合并了 {len(image_list)} 张图片。")
    print(f"文件保存至: {output_pdf}")
else:
    print("错误：未发现 PNG 图片文件。")