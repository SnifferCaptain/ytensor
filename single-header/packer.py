import os
import re

def process_file(file_path, processed_files=None, indent_prefix=""):
    """
    递归处理文件，展开双引号include
    """
    if processed_files is None:
        processed_files = set()
    
    # 避免循环包含
    abs_path = os.path.abspath(file_path)
    if abs_path in processed_files:
        return []
    processed_files.add(abs_path)
    
    if not os.path.exists(file_path):
        return [f"{indent_prefix}// File not found: {file_path}"]
    
    result = []
    pragma_once_added = False
    
    # 获取当前文件所在的目录，用于解析相对路径
    current_file_dir = os.path.dirname(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # 处理空行 - 去掉空格和tab
        if line.strip() == "":
            result.append("")
            continue
            
        # 检查是否是双引号include
        match = re.match(r'^(\s*)#include\s+"([^"]+)"', line)
        if match:
            line_indent = match.group(1)
            include_file = match.group(2)
            
            # 计算include文件的绝对路径 - 相对于当前文件所在目录
            include_path = os.path.join(current_file_dir, include_file)
            include_path = os.path.normpath(include_path)  # 标准化路径
            
            # 递归处理include文件
            new_indent = indent_prefix + line_indent
            included_content = process_file(include_path, processed_files, new_indent)
            result.extend(included_content)
            continue
        
        # 处理#pragma once - 只保留第一个
        if line.strip() == "#pragma once":
            if not pragma_once_added:
                result.append(indent_prefix + line.rstrip())
                pragma_once_added = True
            continue
        
        # 其他行直接添加，加上缩进前缀
        result.append(indent_prefix + line.rstrip())
    
    return result

def main():
    # 获取当前脚本所在目录的上级目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # ytensor.hpp文件路径
    ytensor_file = os.path.join(parent_dir, "ytensor.hpp")
    
    if not os.path.exists(ytensor_file):
        print(f"Error: ytensor.hpp not found at {ytensor_file}")
        return
    
    # 处理文件
    result_lines = process_file(ytensor_file)

    # 输出结果到当前目录的ytensor_single.hpp
    output_file = os.path.join(current_dir, "ytensor_single.hpp")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in result_lines:
            f.write(line + '\n')
    
    print(f"Successfully packed ytensor.hpp to {output_file}")

if __name__ == "__main__":
    main()