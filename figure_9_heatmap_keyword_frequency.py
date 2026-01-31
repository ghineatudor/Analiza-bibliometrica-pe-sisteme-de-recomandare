import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

print("[STEP 1] Loading data from merged_output.xlsx...")
df = pd.read_excel("merged_output.xlsx")
year_col = "Publication Year"
keyword_col = "Author Keywords"

print(f"  - Total rows loaded: {len(df)}")

df = df[df[keyword_col].notnull() & df[year_col].notnull()]
print(f"  - Rows after removing missing values: {len(df)}")

df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype(int)
df = df[df[year_col] >= 1980]
print(f"  - Year range: {df[year_col].min()} - {df[year_col].max()}")

print("[STEP 2] Explode and clean keywords...")

exploded = df.assign(**{
    keyword_col: df[keyword_col].str.split(r'[;,|]')
}).explode(keyword_col)

exploded[keyword_col] = (
    exploded[keyword_col]
    .str.strip()
    .str.lower()
)

exploded = exploded[exploded[keyword_col].str.len() > 0]
print(f"  - Total keyword entries after exploding: {len(exploded)}")
print(f"  - Unique keywords: {exploded[keyword_col].nunique()}")

print("[STEP 3] Select top 20 keywords...")

top_keywords = exploded[keyword_col].value_counts().nlargest(20).index
print("  - Top 20 keywords:")
for i, kw in enumerate(top_keywords, 1):
    count = exploded[exploded[keyword_col] == kw].shape[0]
    print(f"    {i}. {kw} ({count} occurrences)")

filtered = exploded[exploded[keyword_col].isin(top_keywords)]
print(f"  - Filtered rows for top 20: {len(filtered)}")

print("[STEP 4] Create pivot table for heatmap...")

heatmap_data = (
    filtered.groupby([year_col, keyword_col])
    .size()
    .unstack(fill_value=0)
)

years_min = filtered[year_col].min()
years_max = filtered[year_col].max()
years_range = range(years_min, years_max + 1)
heatmap_data = heatmap_data.reindex(years_range, fill_value=0)

print(f"  - Pivot table dimensions: {heatmap_data.shape}")
print(f"  - Years: {years_min} - {years_max} ({len(years_range)} years)")

print("[STEP 5] Generate heatmap figure...")

fig, ax = plt.subplots(figsize=(14, 8))

sns.heatmap(
    heatmap_data.T,
    cmap="YlGnBu",
    cbar_kws={
        'label': 'Frequency',
        'shrink': 0.8
    },
    ax=ax,
    linewidths=0.5,
    linecolor='white',
    annot=False,
    fmt='d'
)

ax.set_title(
    "Keyword Frequency Over Time",
    fontsize=14,
    fontweight='bold',
    pad=15
)

ax.set_xlabel(
    "Publication Year",
    fontsize=12,
    fontweight='bold',
    labelpad=10
)

ax.set_ylabel(
    "Keywords",
    fontsize=12,
    fontweight='bold',
    labelpad=10
)

years_list = list(years_range)
tick_positions = range(0, len(years_list), 5)
tick_labels = [years_list[i] for i in tick_positions]

ax.set_xticks([i + 0.5 for i in tick_positions])
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)

ax.tick_params(axis='y', labelsize=10)
plt.yticks(rotation=0)

print("[STEP 6] Save figure...")

plt.tight_layout()

plt.savefig(
    "heatmap_keywords_time.png",
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.15,
    facecolor='white',
    edgecolor='none'
)

print("[OK] Heatmap saved successfully: heatmap_keywords_time.png")
print(f"  - Dataset dimensions: {len(filtered)} rows")
print(f"  - Top 20 keywords analyzed")
print(f"  - Years: {years_min} - {years_max}")
print(f"  - Resolution: 600 DPI)")

plt.show()