import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    """MD5 hash of file contents — identical files = identical hash"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def check_duplicates():
    # Image directories grouped by true independent source
    dirs_to_check = {
        'ORCA_v1': [
            'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset',
        ],
        'ORCA_v2': [
            'data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/'
            'OC Dataset kaggle new',
        ],
        'ashenafifasilkebede': [
            'data/raw/kaggle-oral-ashen/train',
            'data/raw/kaggle-oral-ashen/val',
            'data/raw/kaggle-oral-ashen/test',
        ],
    }

    print('Scanning all images for duplicates')

    # Build hash --> list of (dataset_name, filepath)
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

    # Find all duplicate groups (hash shared by 2+ files)
    duplicates = {h: files for h, files in hash_to_files.items()
                  if len(files) > 1}

    if not duplicates:
        print('No duplicates found anywhere!')
        return

    print(f'Found {len(duplicates)} duplicate image groups\n')

    # Categorize EVERY duplicate group (no sampling/slicing)
    within_orca = []          # ORCA_v1 <-> ORCA_v2 overlap (expected/known)
    orca_vs_ashen = []        # ORCA <-> ashen overlap (would be serious)
    within_ashen = []         # ashen's own internal duplicates
    other = []                # any unexpected combination

    for h, files in duplicates.items():
        datasets_present = set(f[0] for f in files)

        if datasets_present == {'ORCA_v1', 'ORCA_v2'}:
            within_orca.append((h, files))
        elif 'ashenafifasilkebede' in datasets_present and \
             ('ORCA_v1' in datasets_present or 'ORCA_v2' in datasets_present):
            orca_vs_ashen.append((h, files))
        elif datasets_present == {'ashenafifasilkebede'}:
            within_ashen.append((h, files))
        else:
            other.append((h, files))

    total_categorized = (len(within_orca) + len(orca_vs_ashen) +
                          len(within_ashen) + len(other))

    print(f'{"="*55}')
    print(f'DUPLICATE BREAKDOWN')
    print(f'{"="*55}')
    print(f'ORCA v1 <-> ORCA v2 overlap:      {len(within_orca)}')
    print(f'ORCA <-> ashen (serious!):        {len(orca_vs_ashen)}')
    print(f'Within ashen only:                {len(within_ashen)}')
    print(f'Other/unexpected combinations:    {len(other)}')
    print(f'{"-"*55}')
    print(f'Total verified:                   {total_categorized} '
          f'(should equal {len(duplicates)})')
    print(f'{"="*55}\n')

    if orca_vs_ashen:
        print('Found overlap between ORCA and ashen datasets')
        for h, files in orca_vs_ashen[:10]:
            print(f'  Duplicate group (hash={h[:8]}...):')
            for ds, fp in files:
                print(f'    [{ds}] {os.path.basename(fp)}')
    else:
        print('No overlap between ORCA and ashen — sources are independent.\n')

    if within_orca:
        print(f'\nORCA v1/v2 overlap confirmed: {len(within_orca)} images '
              f'exist in both dataset versions. Deduplicate ORCA before use')

    if within_ashen:
        print(f'\n{len(within_ashen)} internal duplicate groups within '
              f'ashenafifasilkebede itself.')

    if other:
        print(f'\n {len(other)} unexpected combination(s) — inspecting:')
        for h, files in other:
            print(f'Duplicate group (hash={h[:8]}...):')
            for ds, fp in files:
                print(f'    [{ds}] {os.path.basename(fp)}')

if __name__ == '__main__':
    check_duplicates()