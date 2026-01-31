import os
import pandas as pd
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

plt.style.use("seaborn-v0_8-whitegrid")
FONT_SIZE = 12
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PATH = "merged_output.xlsx"
COL_ABS = "Abstract"
COL_YEAR = "Publication Year"

df = pd.read_excel(FILE_PATH)

df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
df = df.dropna(subset=[COL_YEAR])
df[COL_YEAR] = df[COL_YEAR].astype(int)

sid = SentimentIntensityAnalyzer()

df["vader"] = (
    df[COL_ABS]
    .fillna("")
    .astype(str)
    .map(lambda x: sid.polarity_scores(x)["compound"])
)

overall_mean = df["vader"].mean()
overall_median = df["vader"].median()

print(f"Total abstracts: {len(df)}")
print(f"Overall mean VADER:   {overall_mean:.3f}")
print(f"Overall median VADER: {overall_median:.3f}")

mask = (df[COL_YEAR] >= 1980) & (df[COL_YEAR] <= 2025)
df_trend = df.loc[mask].copy()

sentiment_trend = (
    df_trend.groupby(COL_YEAR)["vader"]
    .agg(["mean", "median", "count"])
    .reset_index()
)

sentiment_trend.to_csv(
    os.path.join(OUTPUT_DIR, "sentiment_by_year_detailed_final.csv"),
    index=False,
)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    sentiment_trend[COL_YEAR],
    sentiment_trend["mean"],
    marker="o",
    linewidth=1.8,
    color="#1f77b4",
)

ax.set_xlabel("Publication year", fontsize=FONT_SIZE)
ax.set_ylabel("Average VADER compound score", fontsize=FONT_SIZE)

ax.axhline(
    y=overall_mean,
    color="red",
    linestyle="--",
    linewidth=1.4,
    label=f"Overall mean: {overall_mean:.3f}",
)

ax.legend(fontsize=FONT_SIZE - 1)
ax.set_xlim(sentiment_trend[COL_YEAR].min() - 0.5,
            sentiment_trend[COL_YEAR].max() + 0.5)

ax.grid(True, linestyle="--", alpha=0.6)

fig.tight_layout()
fig_path = os.path.join("figure7_sentiment_trend_vader.png")
fig.savefig(fig_path, dpi=600)
plt.close(fig)

print(f"Figure saved as: {fig_path}")
