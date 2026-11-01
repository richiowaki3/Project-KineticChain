import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base_source = r'D:\sound ideas wav\SoundIdeas_Series6000_Sampler_Slices'

with open(os.path.join(base_source, 'curated_golden_onomatopoeia_labels.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

skipped = [v for v in data.values() if v.get('status') == 'skipped']
print(f"Skipped in JSON: {len(skipped)}")

missing_skipped = 0
for s in skipped:
    cat = s['category']
    fname = s['file']
    fpath = os.path.join(base_source, cat, fname)
    if not os.path.exists(fpath):
        missing_skipped += 1
print(f"Missing skipped files: {missing_skipped}")

# Check _rejected files across all categories
total_rejected = 0
missing_rejected = 0
for cat in os.listdir(base_source):
    cat_dir = os.path.join(base_source, cat)
    if not os.path.isdir(cat_dir): continue
    rej_dir = os.path.join(cat_dir, '_rejected')
    if os.path.exists(rej_dir):
        rej_files = [f for f in os.listdir(rej_dir) if f.endswith('.wav')]
        total_rejected += len(rej_files)

print(f"Total _rejected files across all 11 categories: {total_rejected}")
print(f"Total trash/rejected candidate: {len(skipped)} (skipped) + {total_rejected} (rejected) = {len(skipped) + total_rejected}")
