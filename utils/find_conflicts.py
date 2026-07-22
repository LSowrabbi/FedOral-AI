import os
import hashlib
from collections import defaultdict


def get_hash(fp):
    with open(fp, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def find_conflicting_hashes():
    """Returns set of hashes that have label conflicts."""
    dirs_to_check = {
        'ORCA_v1': ['data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset'],
        'ORCA_v2': ['data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new'],
        'ashenafifasilkebede': [
            'data/raw/kaggle-oral-ashen/train',
            'data/raw/kaggle-oral-ashen/val',
            'data/raw/kaggle-oral-ashen/test',
        ],
    }

    hash_to_labels = defaultdict(set)
    hash_to_files = defaultdict(list)

    for dataset_name, folders in dirs_to_check.items():
        for folder in folders:
            for cls in ['CANCER', 'NON CANCER', 'OSCC', 'Normal']:
                class_dir = os.path.join(folder, cls)
                if not os.path.exists(class_dir):
                    continue
                label = 'CANCER' if cls in ['CANCER', 'OSCC'] else 'NON CANCER'
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fpath = os.path.join(class_dir, fname)
                        h = get_hash(fpath)
                        hash_to_labels[h].add(label)
                        hash_to_files[h].append(fpath)

    conflicting_hashes = {h for h, labels in hash_to_labels.items()
                          if len(labels) > 1}

    print(f"Found {len(conflicting_hashes)} conflicting hash groups")
    print(f"Total images to exclude: "
          f"{sum(len(hash_to_files[h]) for h in conflicting_hashes)}")

    return conflicting_hashes


if __name__ == '__main__':
    conflicts = find_conflicting_hashes()
    # Save to file for use in deduplication step
    with open('data/raw/conflicting_hashes.txt', 'w') as f:
        for h in conflicts:
            f.write(h + '\n')
    print(f"\nSaved {len(conflicts)} conflicting hashes to "
          f"data/raw/conflicting_hashes.txt")