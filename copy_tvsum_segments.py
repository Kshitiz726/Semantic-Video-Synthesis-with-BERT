import os
import shutil

# ==========================
# CONFIG
# ==========================

SRC_ROOT = r"D:\Semantic-Video-Synthesis-with-BERT\importance_scoring\ydata-tvsum50-v1_1\tvsum_segments"
DST_DIR  = r"D:\Semantic-Video-Synthesis-with-BERT\importance_scoring\clips"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ==========================
# PREP
# ==========================

os.makedirs(DST_DIR, exist_ok=True)

scene_counter = 1

# ==========================
# WALK THROUGH FOLDERS
# ==========================

for root, dirs, files in os.walk(SRC_ROOT):
    dirs.sort()    # deterministic folder order
    files.sort()   # deterministic file order

    for file in files:
        if file.lower().endswith(VIDEO_EXTS):
            src_path = os.path.join(root, file)

            dst_name = f"scene_{scene_counter:03d}.mp4"
            dst_path = os.path.join(DST_DIR, dst_name)

            shutil.copy2(src_path, dst_path)

            print(f"Copied: {src_path} -> {dst_name}")

            scene_counter += 1

print("\n✅ Done.")
print(f"Total scenes copied: {scene_counter - 1}")
