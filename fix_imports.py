import os

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the bad absolute imports
    new_content = content.replace("from ", "from ")
    new_content = new_content.replace("import ", "import ")
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

for root, dirs, files in os.walk("d:/Meta/cloud_sre_v2"):
    if ".git" in root or "__pycache__" in root or ".venv" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            replace_in_file(os.path.join(root, file))

print("Done fixing imports.")
