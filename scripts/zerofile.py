import os

base_dir = os.path.expanduser("~/atmos/OutputStorage")

empty_files = []

for root, dirs, files in os.walk(base_dir):
    for name in files:
        full_path = os.path.join(root, name)
        try:
            if os.path.getsize(full_path) == 0:
                empty_files.append(full_path)
        except OSError as e:
            print(f"Error accessing {full_path}: {e}")

if empty_files:
    print("Empty (0-byte) files found:\n")
    for f in empty_files:
        print(f)
else:
    print("No empty files found.")
