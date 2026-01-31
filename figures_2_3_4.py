import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

plt.style.use("seaborn-v0_8-whitegrid")
FONT_SIZE = 12

file_path = "merged_output.xlsx"
col_citations = "Times Cited, All Databases"

df = pd.read_excel(file_path)
cit = pd.to_numeric(df[col_citations], errors="coerce").fillna(0).astype(int)

n_articles = len(cit)
mean_cit = cit.mean()
median_cit = cit.median()
max_cit = cit.max()
pct_uncited = (cit == 0).mean() * 100

threshold_top1 = cit.quantile(0.99)
total_cit = cit.sum()
top1_cit = cit[cit >= threshold_top1].sum()
share_top1 = (top1_cit / total_cit) * 100 if total_cit > 0 else 0

print("--- STATISTICS USED IN FIGURES ---")
print(f"Number of articles    : {n_articles}")
print(f"Mean citations/article: {mean_cit:.2f}")
print(f"Median citations      : {median_cit:.1f}")
print(f"Max citations         : {max_cit}")
print(f"Uncited articles (%)  : {pct_uncited:.1f}%")
print(f"Top 1% threshold (>=) : {threshold_top1}")
print(f"Top 1% citation share : {share_top1:.1f}%")
print("--------------------------------------------------")


fig, ax = plt.subplots(figsize=(7, 4.5))

bins = np.arange(0, 101, 1)

cit_clipped = cit.clip(upper=100)
ax.hist(cit_clipped, bins=bins, color="#1f77b4", edgecolor="white")

ax.set_xlabel("Number of citations", fontsize=FONT_SIZE)
ax.set_ylabel("Number of publications", fontsize=FONT_SIZE)

ax.axvline(mean_cit, color="red", linestyle="--", linewidth=1.5,
           label=f"Mean = {mean_cit:.2f}")
ax.axvline(median_cit, color="green", linestyle=":", linewidth=1.8,
           label=f"Median = {median_cit:.1f}")

ax.legend(fontsize=FONT_SIZE-1)
ax.text(0.98, 0.95,
        f"Max citations: {max_cit}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT_SIZE-1)

fig.tight_layout()
fig.savefig("figure2_citation_histogramFIN.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))

cit_sorted = np.sort(cit)
cum_probs = np.arange(1, n_articles + 1) / n_articles

ax.plot(cit_sorted, cum_probs, color="#1f77b4", linewidth=1.8)

ax.set_xlabel("Citations per article", fontsize=FONT_SIZE)
ax.set_ylabel("Cumulative proportion of articles", fontsize=FONT_SIZE)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.axvline(median_cit, color="green", linestyle=":", linewidth=1.5)

ax.text(median_cit, 0.52,
        f"50% articles: {int(median_cit)}+ citations",
        fontsize=FONT_SIZE-1,
        va="bottom", ha="left",
        color="green")

ax.set_ylim(0, 1.01)
ax.set_xlim(left=0)

fig.tight_layout()
fig.savefig("figure3a_cdf_citationsFIN.png", dpi=600)
plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 4.5))

ax.boxplot(cit, vert=False, showfliers=False)

ax.set_xlabel("Citations per article", fontsize=FONT_SIZE)
ax.set_yticks([])

text_str = (
    f"Articles: {n_articles}\n"
    f"Mean: {mean_cit:.2f}\n"
    f"Median: {median_cit:.1f}\n"
    f"Max: {max_cit}\n"
    f"Uncited articles: {pct_uncited:.1f}%\n"
    f"Top 1% (>={int(threshold_top1)} citations): {share_top1:.1f}% of all citations"
)

ax.text(0.98, 0.95, text_str,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT_SIZE-1,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white",
                  edgecolor="gray",
                  alpha=0.8))

fig.tight_layout()
fig.savefig("figure3b_boxplot_statsFIN.png", dpi=600)
plt.close(fig)

print("Figures saved as:")
print("  - figure2_citation_histogramFIN.png")
print("  - figure3a_cdf_citationsFIN.png")
print("  - figure3b_boxplot_statsFIN.png")
