# -*- coding: utf-8 -*-
import os

def fix_vector_file(input_path, output_path):
    """修复向量文件，确保词汇表大小与向量数量匹配"""
    if not os.path.exists(input_path):
        print(f"输入文件 {input_path} 不存在")
        return
    
    lines = []
    with open(input_path, 'rb') as f:
        for line in f:
            try:
                parts = line.strip().split()
                if len(parts) > 1:  # 确保至少有一个词和一个向量值
                    word = parts[0]
                    vector_values = [float(x) for x in parts[1:]]
                    if len(vector_values) == 50:  # 检查是否为50维向量
                        lines.append(line)
            except:
                print(f"跳过无效行: {line}")
                continue
    
    # 写入修复后的文件
    with open(output_path, 'wb') as f:
        f.writelines(lines)
    
    print(f"修复完成: {len(lines)} 个有效行写入 {output_path}")

if __name__ == "__main__":
    input_file = "assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec"
    output_file = "assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec.fixed"
    
    # 创建目录如果不存在
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    
    fix_vector_file(input_file, output_file)
    print("向量文件修复完成")
