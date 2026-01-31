import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from datetime import datetime

def remove_diacritics(text):
    if pd.isna(text):
        return text
    normalized = unicodedata.normalize("NFKD", str(text))
    without_diacritics = "".join(c for c in normalized if not unicodedata.combining(c))
    return without_diacritics

def normalize_text_col(val):
    if pd.isna(val):
        return val
    v = remove_diacritics(val)
    return str(v).strip().lower()

df = pd.read_excel("merged_output.xlsx")
year_col = "Publication Year"
keyword_col = "Author Keywords"

text_columns_to_normalize = [
    'Author Keywords', 'Title', 'Abstract', 'Author', 'Authors',
    'Author Full Name', 'Source Title', 'Affiliations', 'Keywords Plus'
]
for col in text_columns_to_normalize:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text_col)

print("\n" + "="*70)
print("=== CAGR FOR TOTAL PUBLICATIONS ===")
print("="*70)
print("NOTE: Year 2026 is excluded from CAGR (incomplete data - January only)")
print("="*70)

df_complete_years = df[df[year_col] <= 2025]

yearly_pubs = df_complete_years[df_complete_years[year_col].notnull()].groupby(year_col).size().reset_index(name='count')
yearly_pubs = yearly_pubs.sort_values(year_col)

print("\nPublications for first 10 years:")
print(yearly_pubs.head(10).to_string(index=False))
print("\nPublications for last 10 years (until 2025):")
print(yearly_pubs.tail(10).to_string(index=False))

first_year = yearly_pubs.iloc[0]
last_year = yearly_pubs.iloc[-1]
year_span = last_year[year_col] - first_year[year_col]

if year_span > 0 and first_year['count'] > 0:
    cagr_total = ((last_year['count'] / first_year['count']) ** (1 / year_span)) - 1
    print(f"\n{'='*70}")
    print(f"CAGR TOTAL PUBLICATIONS (1980-2025):")
    print(f"{'='*70}")
    print(f"   First year: {int(first_year[year_col])} with {int(first_year['count'])} publications")
    print(f"   Last year: {int(last_year[year_col])} with {int(last_year['count'])} publications")
    print(f"   Period: {int(year_span)} years")
    print(f"   CAGR: {cagr_total:.4f} ({cagr_total*100:.2f}%)")
    print(f"{'='*70}")
    
    recent_decade = yearly_pubs[yearly_pubs[year_col] >= 2015]
    if len(recent_decade) >= 2:
        first_recent = recent_decade.iloc[0]
        last_recent = recent_decade.iloc[-1]
        year_span_recent = last_recent[year_col] - first_recent[year_col]
        
        if year_span_recent > 0 and first_recent['count'] > 0:
            cagr_recent = ((last_recent['count'] / first_recent['count']) ** (1 / year_span_recent)) - 1
            print(f"\nCAGR LAST DECADE (2015-2025):")
            print(f"   Initial year: {int(first_recent[year_col])} with {int(first_recent['count'])} publications")
            print(f"   Final year: {int(last_recent[year_col])} with {int(last_recent['count'])} publications")
            print(f"   Period: {int(year_span_recent)} years")
            print(f"   CAGR: {cagr_recent:.4f} ({cagr_recent*100:.2f}%)")
            print(f"{'='*70}")

print("\nAVERAGE PUBLICATIONS PER DECADE (including partial 2026):")
df_with_year = df[df[year_col].notnull()].copy()
df_with_year['Decade'] = (df_with_year[year_col] // 10) * 10
decade_counts = df_with_year.groupby('Decade').size().reset_index(name='total')

decade_years = df_with_year.groupby('Decade')[year_col].agg(['min', 'max', 'nunique']).reset_index()
decade_years.columns = ['Decade', 'first_year', 'last_year', 'years_with_data']
decade_avg = decade_counts.merge(decade_years, on='Decade')
decade_avg['avg_per_year'] = decade_avg['total'] / decade_avg['years_with_data']

print(decade_avg[['Decade', 'total', 'years_with_data', 'avg_per_year']].to_string(index=False))

print("\nNOTE: Decade 2020 includes only 2020-2026 (6-7 years), not complete decade")

print("\n" + "="*70)
print("=== KEYWORD PROCESSING (including 2026 for context) ===")
print("="*70)

df_keywords = df[df[keyword_col].notnull() & df[year_col].notnull()]
exploded = df_keywords.assign(**{keyword_col: df_keywords[keyword_col].str.split(r';|,|\|')}).explode(keyword_col)

exploded[keyword_col] = (
    exploded[keyword_col]
    .str.strip()
    .str.lower()
    .apply(remove_diacritics)
)
top_keywords = exploded[keyword_col].value_counts().nlargest(100).index
exploded_top = exploded[exploded[keyword_col].isin(top_keywords)]
keyword_counts = exploded_top.groupby([year_col, keyword_col]).size().reset_index(name="count")

cagr_results = []
for kw in keyword_counts[keyword_col].unique():
    kw_years = keyword_counts[keyword_col] == kw
    kw_df = keyword_counts[kw_years].sort_values(year_col)
    if len(kw_df) >= 2:
        first, last = kw_df.iloc[0], kw_df.iloc[-1]
        year_span = last[year_col] - first[year_col]
        if first['count'] > 0 and year_span > 0:
            cagr = (last['count'] / first['count']) ** (1 / year_span) - 1
            cagr_results.append({
                "Keyword": kw, 
                "CAGR": cagr,
                "First_Year": int(first[year_col]),
                "Last_Year": int(last[year_col]),
                "First_Count": int(first['count']),
                "Last_Count": int(last['count']),
                "Years": int(year_span)
            })

cagr_df = pd.DataFrame(cagr_results)

if not cagr_df.empty:
    cagr_df = cagr_df.sort_values("CAGR", ascending=False)
    
    output_file = "ALL_CAGR_results.xlsx"
    try:
        cagr_df.to_excel(output_file, index=False)
        print(f"\nSaved {len(cagr_df)} keywords with CAGR in {output_file}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ALL_CAGR_results_{timestamp}.xlsx"
        cagr_df.to_excel(output_file, index=False)
        print(f"\nFile was locked. Saved to: {output_file}")
    
    print("\n" + "="*70)
    print("=== TOP 50 KEYWORDS BY CAGR ===")
    print("="*70)
    print(cagr_df.head(50)[['Keyword', 'CAGR', 'First_Year', 'Last_Year']].to_string(index=False))
    
    keywords_to_check = [
        'contrastive learning',
        'graph neural network',
        'self-supervised learning',
        'federated learning',
        'fairness', 
        'privacy',
        'privacy-preserving',
        'explainability',
        'deep learning',
        'collaborative filtering'
    ]
    
    print("\n" + "="*70)
    print("=== TEXT KEYWORDS VERIFICATION ===")
    print("="*70)
    
    for kw in keywords_to_check:
        match = cagr_df[cagr_df['Keyword'].str.contains(kw, case=False, na=False)]
        if not match.empty:
            print(f"\n{kw.upper()}:")
            print(match[['Keyword', 'CAGR', 'First_Year', 'Last_Year']].to_string(index=False))
            if 'contrastive learning' in kw and match.iloc[0]['CAGR'] > 1.8:
                print(f"   Confirmed in text: 187% (actual: {match.iloc[0]['CAGR']*100:.1f}%)")
            elif 'graph neural' in kw and 0.85 <= match.iloc[0]['CAGR'] <= 0.95:
                print(f"   Confirmed in text: 89% (actual: {match.iloc[0]['CAGR']*100:.1f}%)")
            elif 'self-supervised' in kw and 0.70 <= match.iloc[0]['CAGR'] <= 0.80:
                print(f"   Confirmed in text: 76% (actual: {match.iloc[0]['CAGR']*100:.1f}%)")
            elif 'federated' in kw and 0.70 <= match.iloc[0]['CAGR'] <= 0.75:
                print(f"   Confirmed in text: 72% (actual: {match.iloc[0]['CAGR']*100:.1f}%)")
        else:
            print(f"\n{kw.upper()}: NOT FOUND in top 100 keywords")
    
    top_n = min(20, len(cagr_df))
    
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(x="CAGR", y="Keyword", data=cagr_df.head(top_n), ax=ax)
    ax.set_title(f"Top {top_n} Keywords by CAGR (Author Keywords)", fontsize=28, pad=60)
    ax.set_xlabel("CAGR", fontsize=22, labelpad=15)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=19)
    ax.tick_params(axis='x', labelsize=17)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    plt.savefig("figure_8_cagr_keywords.png", dpi=600, bbox_inches='tight', pad_inches=0.13)
    plt.savefig("figure_8_cagr_keywords.tiff", dpi=600, bbox_inches='tight', pad_inches=0.13)
    plt.close()
    
    
else:
    print("\nNo CAGR results generated for keywords.")

print("\n" + "="*70)
print("=== SUMMARY FOR ARTICLE ===")
print("="*70)
print("\nVALUES TO VERIFY IN TEXT:")
print(f"\n1. CAGR total publications (1980-2025): {cagr_total*100:.1f}%")
print(f"   Verify if text states ~18.3%")
print(f"\n2. CAGR last decade (2015-2025): {cagr_recent*100:.1f}%")
print(f"   Verify if text states ~24.7%")
print("\n3. Top CAGR keywords (verify numbers from text):")
top_5_for_text = cagr_df.head(5)
for idx, row in top_5_for_text.iterrows():
    print(f"   - {row['Keyword']}: {row['CAGR']*100:.0f}%")
