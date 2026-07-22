import os
import shutil
import hashlib


def get_hash(fp):
    with open(fp, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def deduplicate_ashen():
    source_folders = [
        'data/raw/kaggle-oral-ashen/train',
        'data/raw/kaggle-oral-ashen/val',
        'data/raw/kaggle-oral-ashen/test',
    ]
    output_dir = 'data/raw/ashen-deduplicated'

    # Load conflicting hashes to exclude entirely
    conflicting_hashes = set()
    conflict_file = 'data/raw/conflicting_hashes.txt'
    if os.path.exists(conflict_file):
        with open(conflict_file) as f:
            conflicting_hashes = set(line.strip() for line in f)
        print(f"Loaded {len(conflicting_hashes)} conflicting hashes "
              f"to exclude")

    seen_hashes = set()
    copied = 0
    skipped_duplicate = 0
    excluded_conflicts = 0

    for source_folder in source_folders:
        split_name = os.path.basename(source_folder)  # train/val/test
        for cls in ['OSCC', 'Normal']:
            src_folder = os.path.join(source_folder, cls)
            if not os.path.exists(src_folder):
                continue
            dst_folder = os.path.join(output_dir, cls)
            os.makedirs(dst_folder, exist_ok=True)

            for fname in os.listdir(src_folder):
                if not fname.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                src_path = os.path.join(src_folder, fname)
                h = get_hash(src_path)

                if h in conflicting_hashes:
                    excluded_conflicts += 1
                    continue

                if h in seen_hashes:
                    skipped_duplicate += 1
                    continue

                seen_hashes.add(h)
                dst_fname = f"{split_name}_{fname}"
                dst_path = os.path.join(dst_folder, dst_fname)
                shutil.copy2(src_path, dst_path)
                copied += 1

    print(f"Deduplication complete")
    print(f"Copied (unique):          {copied}")
    print(f"Skipped (duplicate):      {skipped_duplicate}")
    print(f"Excluded (label conflict): {excluded_conflicts}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    deduplicate_ashen()