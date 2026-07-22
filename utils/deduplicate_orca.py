import os
import shutil
import hashlib


def get_hash(fp):
    with open(fp, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def deduplicate_orca():
    d1 = 'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset'
    d2 = ('data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/'
          'OC Dataset kaggle new')
    output_dir = 'data/raw/orca-deduplicated'

    # Load conflicting hashes to exclude entirely
    conflicting_hashes = set()
    conflict_file = 'data/raw/conflicting_hashes.txt'
    if os.path.exists(conflict_file):
        with open(conflict_file) as f:
            conflicting_hashes = set(line.strip() for line in f)
        print(f"Loaded {len(conflicting_hashes)} conflicting hashes "
              f"to exclude")

    seen_hashes = set()
    excluded_conflicts = 0
    copied = 0
    skipped = 0

    # d1 processed first — its images are kept preferentially
    # when a duplicate exists in both d1 and d2
    for source_dir, source_name in [(d1, 'v1'), (d2, 'v2')]:
        for cls in ['CANCER', 'NON CANCER']:
            src_folder = os.path.join(source_dir, cls)
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
                    skipped += 1
                    continue

                seen_hashes.add(h)
                # Prefix filename with source to avoid name collisions
                dst_fname = f"{source_name}_{fname}"
                dst_path = os.path.join(dst_folder, dst_fname)
                shutil.copy2(src_path, dst_path)
                copied += 1

    print(f"Deduplication complete")
    print(f"Copied (unique):        {copied}")
    print(f"Skipped (duplicate):    {skipped}")
    print(f"Excluded (label conflict): {excluded_conflicts}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    deduplicate_orca()