import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from Processing_Time_Breakdown.txt
labels = [
    'Importance Scoring\n(CLIP + Models)', 
    'Shot Detection\n(TransNetV2)', 
    'Scene Segmentation\n(LGSS)', 
    'Clip Extraction', 
    'Text Embeddings',
    'Other Stages'
]

# Raw percentages
sizes = [33.3, 26.7, 10.0, 6.7, 6.7]
# Calculate remainder for "Other" (merging, refinement, fusion, etc.)
remainder = 100 - sum(sizes)
sizes.append(remainder)

# IEEE Colors (Subdued/Professional)
colors = [
    '#CC3333',  # Importance Scoring (Red - Bottleneck)
    '#003366',  # Shot Detection (Dark Blue)
    '#009966',  # Scene Segmentation (Green)
    '#FF9933',  # Clip Extraction (Orange)
    '#663399',  # Text Embeddings (Purple)
    '#999999'   # Other (Grey)
]

# Create Figure
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Create Donut
wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%', 
    startangle=140, 
    pctdistance=0.85, 
    wedgeprops={'width': 0.4, 'edgecolor': 'w'}
)

# Typography settings for IEEE style
plt.setp(texts, size=10, family='serif', weight='normal')
plt.setp(autotexts, size=9, weight='bold', color='white')

# Center Title
ax.text(0, 0, 'Total Time:\n~30 min', ha='center', va='center', fontsize=12, family='serif', weight='bold')

plt.title('Processing Time Breakdown by Stage', fontsize=14, family='serif', pad=20)
plt.tight_layout()

# Save
output_dir = Path("analysis_graphs")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "processing_time_donut.png"
plt.savefig(output_path, bbox_inches='tight')

print(f"Chart generated at: {output_path.resolve()}")
