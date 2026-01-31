import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
FONT_SIZE = 12

file_path = "merged_output.xlsx"
col_year = "Publication Year"

df = pd.read_excel(file_path)

years = pd.to_numeric(df[col_year], errors="coerce").dropna().astype(int)

pubs_per_year = years.value_counts().sort_index()

print("Last years and publication counts:")
print(pubs_per_year.tail(10))

pubs_plot = pubs_per_year[pubs_per_year.index <= 2025]
year_min = pubs_plot.index.min()
year_max = pubs_plot.index.max()

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    pubs_plot.index,
    pubs_plot.values,
    marker="o",
    linewidth=1.8,
    color="#1f77b4",
)

ax.set_xlabel("Publication year", fontsize=FONT_SIZE)
ax.set_ylabel("Number of publications", fontsize=FONT_SIZE)


ax.set_xlim(year_min - 0.5, year_max + 0.5)

for year in [2022, 2023, 2024]:
    if year in pubs_plot.index:
        ax.annotate(
            f"{year}",
            xy=(year, pubs_plot[year]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=FONT_SIZE - 2,
        )

ax.axhline(0, color="gray", linewidth=0.5)

fig.tight_layout()
fig.savefig("figure5_publication_trend.png", dpi=600)
plt.close(fig)

print("Figure 5 saved as figure5_publication_trend.png")
