# createJson.py
import os
import json

data_dir = 'train'
output_json_path = 'class_name_map.json'

class_map = {}
for idx, folder in enumerate(os.listdir(data_dir)):
    class_map[folder] = idx

with open(output_json_path, 'w') as f:
    json.dump(class_map, f, indent=4)

print(f"Class mapping saved to {output_json_path}")
