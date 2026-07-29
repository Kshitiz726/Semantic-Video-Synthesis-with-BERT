import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Data from Processing_Time_Breakdown.txt
data = {
    'Video Duration': ['1 minute', '5 minutes', '10 minutes', '20 minutes'],
    'CPU Only': ['6 min', '30 min', '60 min', '90 min'],
    'GPU (CUDA)': ['2 min', '10 min', '20 min', '30 min'],
    'Speedup': ['3.0x', '3.0x', '3.0x', '3.0x']
}

# Create figure with dark background
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Hide axes
ax.axis('off')

# Define colors
header_bg = '#2d3748'
row_bg_1 = '#1e2533'
row_bg_2 = '#252d3d'
text_color = '#e2e8f0'
header_text = '#ffffff'
speedup_color = '#10b981'

# Table dimensions
n_rows = len(data['Video Duration']) + 1  # +1 for header
n_cols = 4
cell_height = 0.12
cell_width = 0.23
start_y = 0.75

# Draw header
header_labels = ['Video Duration', 'CPU Only', 'GPU (CUDA)', 'Speedup']
for i, label in enumerate(header_labels):
    x = 0.05 + i * cell_width
    
    # Header background
    rect = FancyBboxPatch(
        (x, start_y), cell_width - 0.01, cell_height,
        boxstyle="round,pad=0.01",
        facecolor=header_bg,
        edgecolor='#4a5568',
        linewidth=2,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    
    # Header text
    ax.text(
        x + cell_width/2, start_y + cell_height/2,
        label,
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color=header_text,
        transform=ax.transAxes
    )

# Draw data rows
for row_idx in range(len(data['Video Duration'])):
    y = start_y - (row_idx + 1) * cell_height
    bg_color = row_bg_1 if row_idx % 2 == 0 else row_bg_2
    
    # Video Duration
    rect = FancyBboxPatch(
        (0.05, y), cell_width - 0.01, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=bg_color,
        edgecolor='#374151',
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(
        0.05 + cell_width/2, y + cell_height/2,
        data['Video Duration'][row_idx],
        ha='center', va='center',
        fontsize=13,
        color=text_color,
        transform=ax.transAxes
    )
    
    # CPU Only
    rect = FancyBboxPatch(
        (0.05 + cell_width, y), cell_width - 0.01, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=bg_color,
        edgecolor='#374151',
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(
        0.05 + cell_width + cell_width/2, y + cell_height/2,
        data['CPU Only'][row_idx],
        ha='center', va='center',
        fontsize=13,
        color='#ef4444',  # Red for slower
        fontweight='bold',
        transform=ax.transAxes
    )
    
    # GPU (CUDA)
    rect = FancyBboxPatch(
        (0.05 + 2*cell_width, y), cell_width - 0.01, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=bg_color,
        edgecolor='#374151',
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(
        0.05 + 2*cell_width + cell_width/2, y + cell_height/2,
        data['GPU (CUDA)'][row_idx],
        ha='center', va='center',
        fontsize=13,
        color='#10b981',  # Green for faster
        fontweight='bold',
        transform=ax.transAxes
    )
    
    # Speedup
    rect = FancyBboxPatch(
        (0.05 + 3*cell_width, y), cell_width - 0.01, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=bg_color,
        edgecolor='#374151',
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(
        0.05 + 3*cell_width + cell_width/2, y + cell_height/2,
        data['Speedup'][row_idx],
        ha='center', va='center',
        fontsize=13,
        color=speedup_color,
        fontweight='bold',
        transform=ax.transAxes
    )

# Add title
ax.text(
    0.5, 0.92,
    'Processing Time: CPU vs GPU Comparison',
    ha='center', va='center',
    fontsize=18, fontweight='bold',
    color='#ffffff',
    transform=ax.transAxes
)

# Add subtitle
ax.text(
    0.5, 0.07,
    'Consistent 3x speedup with GPU acceleration across all video lengths',
    ha='center', va='center',
    fontsize=11,
    color='#94a3b8',
    style='italic',
    transform=ax.transAxes
)

plt.tight_layout()
plt.savefig(
    'analysis_graphs/gpu_comparison_table.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#1a1a2e',
    edgecolor='none'
)
print("✅ GPU comparison table saved to: analysis_graphs/gpu_comparison_table.png")
plt.close()
