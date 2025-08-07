import random

# File paths
input_file = "manifest.txt"
train_file = "train_manifest.txt"
val_file = "val_manifest.txt"
val_size = 150
seed = 42

# Read all lines
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Shuffle
random.seed(seed)
random.shuffle(lines)

# Split
val_lines = lines[:val_size]
train_lines = lines[val_size:]

# Write to files
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(val_file, "w", encoding="utf-8") as f:
    f.writelines(val_lines)