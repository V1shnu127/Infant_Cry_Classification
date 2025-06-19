# This program is to count the number of .wav files present in the dataset.

import os

def count_wav_files(root_dir):
    if not os.path.exists(root_dir):
        print(f"Directory '{root_dir}' does not exist.")
        return False, 0

    wav_counts = {}
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Count .wav files in the subfolder
            wav_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
            wav_counts[subfolder] = len(wav_files)

    print(f"\nNumber of .wav files in each subfolder of '{root_dir}':")
    for subfolder, count in sorted(wav_counts.items()):
        print(f"{subfolder}: {count} .wav files")

    total_wav_files = sum(wav_counts.values())
    print(f"Total .wav files in '{root_dir}': {total_wav_files}")

    return True, total_wav_files

directories = [
    "/kaggle/input/segmented-bc2ttv-split/train",
    "/kaggle/input/segmented-bc2ttv-split/test",
    "/kaggle/input/segmented-bc2ttv-split/val"
]

overall_total = 0
for directory in directories:
    success, total = count_wav_files(directory)
    if success:
        overall_total += total

print(f"\nGrand total of .wav files across all directories: {overall_total}")
