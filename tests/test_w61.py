import os, sys
import json
import gzip

# raw_file = 'data/vlnverse/raw_data/vlnverse/final_splits_with_distance/coarse/val_seen/val_seen.json.gz'
raw_file = 'data/vlnverse/raw_data/val_seen/val_seen.json.gz'
target_dir = 'data/vlnverse/raw_data/vlnverse/mixed_splits/test_w61'

os.makedirs(target_dir, exist_ok=True)

with gzip.open(raw_file, 'rt', encoding='utf-8') as f:
    data = json.load(f)

new_data = {'episodes': []}
for episode in data['episodes']:
    if episode['scene_id'] == 'vlnverse/kujiale_0003':
        new_data['episodes'].append(episode)

with gzip.open(os.path.join(target_dir, 'test_w61.json.gz'), 'wt', encoding='utf-8') as f:
    json.dump(new_data, f)