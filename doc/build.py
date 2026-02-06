import os
import json

# 配置
TEMPLATE_FILE = 'template.html'
OUTPUT_FILE = 'index.html'

# 排除列表
EXCLUDED_DIRS = {'__pycache__', '.git', '.vscode', '.idea', 'dist', 'build', 'libs', 'node_modules'}
EXCLUDED_FILES = {'build.py', 'template.html', 'index.html', 'README.md', 'LICENSE'}

def build_directory_tree(base_path, relative_path_prefix, docs_payload):
    tree = []
    try:
        # 排序：保证文件夹和文件顺序稳定
        items = sorted(os.listdir(base_path))
    except OSError:
        return []

    for item in items:
        full_path = os.path.join(base_path, item)
        
        # 过滤
        if item in EXCLUDED_DIRS or item in EXCLUDED_FILES or item.startswith('.'):
            continue
            
        if os.path.isdir(full_path):
            # 递归处理文件夹
            children = build_directory_tree(full_path, f"{relative_path_prefix}/{item}", docs_payload)
            if children:
                tree.append({
                    "type": "folder",
                    "name": item.replace('_', ' ').replace('-', ' ').title(),
                    "children": children
                })
        elif item.endswith('.md') or item.endswith('.mdx'):
            # 处理文件
            web_path = f"{relative_path_prefix}/{item}".strip('/')
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    docs_payload["docs"][web_path] = f.read()
            except Exception as e:
                print(f"Skipping {full_path}: {e}")
                continue
                
            tree.append({
                "type": "file",
                "name": os.path.splitext(item)[0].replace('_', ' ').replace('-', ' ').title(),
                "path": web_path
            })
            
    # 排序：文件夹在前，文件在后
    tree.sort(key=lambda x: (x['type'] == 'file', x['name']))
    return tree

def compile_docs():
    print(f"🚀 开始编译 (纯本地模式)...")
    
    payload = {
        "docs": {},
        "menu": {}
    }
    
    current_dir = os.getcwd()
    
    # 扫描根目录下的语言文件夹
    lang_dirs = []
    for d in os.listdir(current_dir):
        if os.path.isdir(d) and d not in EXCLUDED_DIRS and not d.startswith('.'):
            lang_dirs.append(d)
    
    lang_dirs.sort()
    print(f"🌍 检测到的语言目录: {lang_dirs}")

    for lang in lang_dirs:
        # 单独处理根 README
        readme_path = os.path.join(lang, 'README.mdx')
        if not os.path.exists(readme_path):
            readme_path = os.path.join(lang, 'README.md')
            
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                payload["docs"][f"{lang}/README.mdx"] = f.read()
        
        # 构建树
        payload["menu"][lang] = build_directory_tree(lang, lang, payload)

    # 读取模板
    if not os.path.exists(TEMPLATE_FILE):
        print(f"❌ 错误: 找不到 {TEMPLATE_FILE}")
        return

    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # 注入数据
    json_data = json.dumps(payload, ensure_ascii=False)
    output_content = template_content.replace('/* __INJECT_DATA_HERE__ */ null', json_data)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_content)
        
    print(f"✅ 编译成功: {OUTPUT_FILE}")

if __name__ == '__main__':
    compile_docs()