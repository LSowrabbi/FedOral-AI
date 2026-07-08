import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    '''MD5 hash of file contents — identical files = identical hash'''
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# All image directories to check
dirs_to_check = {
    'ORCA': [
        'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset',
        'data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new',
    ],
    'ashenafifasilkebede': [
        'data/raw/kaggle-oral-ashen/train',
        'data/raw/kaggle-oral-ashen/val',
        'data/raw/kaggle-oral-ashen/test',
    ],
    'viditgandhi': [
        'data/raw/vidit-oral/oral_cancer/train',
        'data/raw/vidit-oral/oral_cancer/val',
        'data/raw/vidit-oral/oral_cancer/test',
    ],
}

print('Scanning all images for duplicates...')

# Build hash → (dataset, filepath) mapping
hash_to_files = defaultdict(list)
total = 0

for dataset_name, folders in dirs_to_check.items():
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.bmp')):
                    fpath = os.path.join(root, fname)
                    h = get_file_hash(fpath)
                    hash_to_files[h].append((dataset_name, fpath))
                    total += 1

print(f'Total images scanned: {total}')
print()

# Find duplicates
duplicates = {h: files for h, files in hash_to_files.items()
              if len(files) > 1}

if not duplicates:
    print('No duplicates found — all datasets are unique!')
else:
    print(f'Found {len(duplicates)} duplicate image groups:')
    cross_dataset = 0
    same_dataset  = 0
    for h, files in list(duplicates.items())[:20]:
        datasets = [f[0] for f in files]
        if len(set(datasets)) > 1:
            cross_dataset += 1
            print(f'  CROSS-DATASET duplicate:')
            for ds, fp in files:
                print(f'    [{ds}] {os.path.basename(fp)}')
        else:
            same_dataset += 1

    print()
    print(f'Cross-dataset duplicates: {cross_dataset}')
    print(f'Same-dataset duplicates:  {same_dataset}')
    print()
    if cross_dataset > 0:
        print('Cross-dataset duplicates will inflate your results!')
        print('These images appear in multiple datasets.')
        print('Remove duplicates before retraining.')
    else:
        print('No cross-dataset duplicates — datasets are independent!')
