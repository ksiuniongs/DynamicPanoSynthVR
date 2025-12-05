import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os
import re

# File paths mapping
files = {
    'RAFT-based Decoupling': 'evaluation/renderer_raft.csv',
    'Background Subtraction Decoupling': 'evaluation/renderer_bg.csv',
    'Monolithic Video-Atlas': 'evaluation/renderer_video_atlas.csv'
}

# Total processing times provided by user
raw_times = {
    'RAFT-based Decoupling': '0h 12m 15s',
    'Background Subtraction Decoupling': '0h 11m 24s',
    'Monolithic Video-Atlas': '0h 23m 39s' 
}

output_path = 'evaluation/performance_analysis_4_metrics.png'

# Custom colors
custom_colors = ['#fc8d59', '#ffffbf', '#91bfdb']

# Helper to parse time string
def parse_time_to_minutes(time_str):
    h = 0
    m = 0
    s = 0
    
    match_h = re.search(r'(\d+)h', time_str)
    if match_h:
        h = int(match_h.group(1))
    
    match_m = re.search(r'(\d+)m', time_str)
    if match_m:
        m = int(match_m.group(1))
        
    match_s = re.search(r'(\d+)s', time_str)
    if match_s:
        s = int(match_s.group(1))
        
    return h * 60 + m + s / 60

# 1. Load Data
data_list = []
keys = list(files.keys())

for algo in keys:
    path = files[algo]
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            if all(col in df.columns for col in ['fps', 'frame_ms', 'mem_used_mb']):
                temp_df = df[['fps', 'frame_ms', 'mem_used_mb']].copy()
                temp_df['Algorithm'] = algo
                data_list.append(temp_df)
        except Exception:
            pass

combined_df = pd.concat(data_list) if data_list else pd.DataFrame()

# 2. Prepare Data for Bar Charts
# Total Time
time_data = []
for algo in keys:
    time_str = raw_times.get(algo, "0m")
    minutes = parse_time_to_minutes(time_str)
    time_data.append({'Algorithm': algo, 'Total Time (min)': minutes})
time_df = pd.DataFrame(time_data)

# Memory Usage (Aggregation for Bar Chart)
# We need a separate DF for memory bar chart to work easily with seaborn barplot which aggregates by default
# But wait, seaborn barplot DOES aggregate by default (mean) and shows error bars (ci).
# So we can just use combined_df for memory barplot.

# 3. Plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
sns.set_theme(style="whitegrid")

# Define metrics
metrics = [
    {'col': 'fps', 'label': 'FPS', 'title': 'FPS Distribution', 'type': 'violin', 'data': combined_df},
    {'col': 'frame_ms', 'label': 'Frame Time (ms)', 'title': 'Frame Time Distribution', 'type': 'violin', 'data': combined_df},
    {'col': 'mem_used_mb', 'label': 'Memory Usage (MB)', 'title': 'AVG Memory Usage', 'type': 'bar', 'data': combined_df},
    {'col': 'Total Time (min)', 'label': 'Time (min)', 'title': 'Total Processing Time', 'type': 'bar', 'data': time_df}
]

for i, metric in enumerate(metrics):
    ax = axes[i]
    algo_col = 'Algorithm'
    
    if metric['type'] == 'violin':
        sns.violinplot(
            x=algo_col, 
            y=metric['col'], 
            data=metric['data'], 
            ax=ax, 
            palette=custom_colors, 
            inner="quartile",
            order=keys
        )
    elif metric['type'] == 'bar':
        # Default estimator is mean, 'errorbar' is usually confidence interval, let's keep default
        sns.barplot(
            x=algo_col, 
            y=metric['col'], 
            data=metric['data'], 
            ax=ax, 
            palette=custom_colors, 
            order=keys,
            width=0.4,
            edgecolor='black', # Add stroke
            linewidth=1.5
        )
        
        # Add labels on top of bars for Time (since it's a single value per Algo)
        # For Memory, it's an average, plotting value might be messy if CI is large, but let's try
        if metric['col'] == 'Total Time (min)':
            for idx, algo in enumerate(keys):
                # We need to find the value from the dataframe
                val = metric['data'][metric['data'][algo_col] == algo][metric['col']].values[0]
                ax.text(idx, val + (val * 0.02), f"{val:.1f}", ha='center', fontsize=10, fontweight='bold')
        elif metric['col'] == 'mem_used_mb':
             # Calculate means for labels
             means = metric['data'].groupby(algo_col)[metric['col']].mean()
             for idx, algo in enumerate(keys):
                 val = means[algo]
                 ax.text(idx, val + (val * 0.02), f"{val:.1f}", ha='center', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_title(metric['title'], fontsize=14)
    ax.set_ylabel(metric['label'], fontsize=12)
    ax.set_xlabel('') 
    ax.set_ylim(bottom=0)
    
    # Remove X-axis labels (tick labels)
    ax.set_xticks([])
    ax.set_xticklabels([])

# Create a shared legend
patches = [mpatches.Patch(color=custom_colors[i], label=keys[i], ec='black', linewidth=1) for i in range(len(keys))]
fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12, frameon=False)

plt.tight_layout()
# Adjust layout to make room for legend at the bottom - increased to 0.2
plt.subplots_adjust(bottom=0.2) 
plt.savefig(output_path)
print(f"Performance analysis saved to {output_path}")
