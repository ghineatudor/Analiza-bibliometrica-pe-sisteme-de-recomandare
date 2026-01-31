import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import json
import re
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\nWORDCLOUD VERIFICATION AND GENERATION")
print("merged_output.xlsx")

print("Checking NLTK resources...")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("    Downloading: wordnet")
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("    Downloading: omw-1.4")
    nltk.download('omw-1.4', quiet=True)
print("NLTK ready\n")

print("LOADING DATASET\n")

file_path = "merged_output.xlsx"
print(f"Reading: {file_path}")

df = pd.read_excel(file_path)
print(f"    Total records: {len(df):,}")
print(f"    Total columns: {len(df.columns)}")
print(f"    Abstract column: 'Abstract'")

df_valid = df[df['Abstract'].notna()].copy()
print(f"    Valid abstracts: {len(df_valid):,} ({len(df_valid)/len(df)*100:.1f}%)")
print(f"    Missing abstracts: {len(df) - len(df_valid)}")

abstract_lengths = df_valid['Abstract'].str.len()
print(f"\nAbstract length statistics (characters):")
print(f"    Min: {abstract_lengths.min()} | Max: {abstract_lengths.max():,}")
print(f"    Mean: {abstract_lengths.mean():.0f} | Median: {abstract_lengths.median():.0f}")

print("\nTEXT PREPROCESSING\n")

print("Preprocessing steps:")
print("    1. Lowercase conversion")
print("    2. Special character removal")
print("    3. Tokenization")
print("    4. Stopword removal")
print("    5. Lemmatization")
print("    6. Short term filtering (< 3 characters)\n")

stop_words = set(stopwords.words('english'))

custom_stops = {
    'also', 'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
    'additionally', 'propose', 'present', 'show', 'demonstrate', 'introduce',
    'describe', 'discuss', 'analyze', 'examine', 'investigate', 'study',
    'provide', 'consider', 'suggest', 'evaluate', 'compare', 'explore',
    'paper', 'article', 'work', 'approach', 'method', 'technique', 'result',
    'experiment', 'evaluation', 'performance', 'accuracy', 'effectiveness',
    'efficiency', 'quality', 'problem', 'solution', 'issue', 'challenge',
    'use', 'used', 'using', 'make', 'made', 'making', 'take', 'taken', 'taking',
    'give', 'given', 'giving', 'find', 'found', 'finding', 'show', 'shown', 'showing',
    'different', 'various', 'several', 'many', 'much', 'more', 'most', 'less',
    'new', 'novel', 'current', 'recent', 'previous', 'existing', 'traditional',
    'based', 'propose', 'proposed', 'two', 'three', 'first', 'second',
    'may', 'can', 'could', 'would', 'should', 'must', 'need', 'able',
    'well', 'good', 'better', 'best', 'high', 'low', 'large', 'small'
}

stop_words.update(custom_stops)
print(f"    Total stopwords: {len(stop_words)}")

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return []
    
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    tokens = text.split()
    
    tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) >= 3
    ]
    
    return tokens

print(f"\nProcessing {len(df_valid):,} abstracts...")
all_tokens = []

for idx, text in enumerate(df_valid['Abstract'], 1):
    tokens = preprocess_text(text)
    all_tokens.extend(tokens)
    
    if idx % 2000 == 0:
        print(f"    Processed: {idx:,}/{len(df_valid):,} ({idx/len(df_valid)*100:.1f}%)")

print(f"Processed: {len(df_valid):,} documents")

unique_tokens = set(all_tokens)
print(f"\nVocabulary:")
print(f"    Unique terms: {len(unique_tokens):,}")
print(f"    Total occurrences: {len(all_tokens):,}")
print(f"    Average terms/document: {len(all_tokens)/len(df_valid):.1f}")

token_freq = Counter(all_tokens)
print(f"\nTop 50 most frequent terms:")
for i, (term, freq) in enumerate(token_freq.most_common(50), 1):
    print(f"   {i:2d}. {term:30s} {freq:6,}")

print("\nREFERENCE WORDCLOUD ANALYSIS\n")

reference_terms = [
    'recommendation', 'recommender', 'system', 'collaborative filtering',
    'neural network', 'deep learning', 'machine learning', 'graph',
    'personalized', 'music', 'product', 'social', 'mobile',
    'context aware', 'hybrid', 'matrix factorization', 'semantic',
    'algorithm', 'user', 'item', 'rating', 'preference', 'similarity',
    'knowledge', 'ontology', 'clustering', 'sentiment analysis',
    'reinforcement learning', 'attention mechanism', 'big data',
    'e-commerce', 'content-based filtering', 'information',
    'personalization', 'framework', 'model', 'data mining',
    'application', 'intelligent', 'prediction', 'optimization',
    'review', 'evaluation', 'cold start', 'sparsity', 'scalability',
    'privacy', 'trust'
]

print(f"Terms identified in reference wordcloud: {len(reference_terms)}")

categories = {
    'Core RS concepts': ['recommendation', 'recommender', 'collaborative filtering', 
                         'content-based filtering', 'hybrid'],
    'Algorithms': ['neural network', 'deep learning', 'machine learning', 'graph', 
                   'matrix factorization', 'reinforcement learning', 'attention mechanism'],
    'Application domains': ['music', 'product', 'social', 'mobile', 'e-commerce'],
    'Techniques': ['personalized', 'context aware', 'semantic', 'sentiment analysis'],
    'Components': ['user', 'item', 'rating', 'preference', 'similarity', 'knowledge']
}

print("\nMain categories:")
for cat, terms in categories.items():
    print(f"    - {cat}: {', '.join(terms[:5])}")

print("\nGENERATE COMPARATIVE WORDCLOUDS (TEMPORAL)\n")

period1_start, period1_end = 1995, 2010
period2_start, period2_end = 2011, 2024

if 'Publication Year' not in df_valid.columns:
    print("WARNING: Column 'Publication Year' not found!")
    print("Available columns:", df_valid.columns.tolist())
    year_col = None
    for col in ['Year', 'PY', 'Publication_Year']:
        if col in df_valid.columns:
            year_col = col
            break
    if year_col:
        df_valid['Publication Year'] = df_valid[year_col]
    else:
        print("ERROR: Cannot identify year column!")
        exit()

df_period1 = df_valid[
    (df_valid['Publication Year'] >= period1_start) & 
    (df_valid['Publication Year'] <= period1_end)
].copy()

df_period2 = df_valid[
    (df_valid['Publication Year'] >= period2_start) & 
    (df_valid['Publication Year'] <= period2_end)
].copy()

print(f"Period 1 ({period1_start}-{period1_end}): {len(df_period1):,} documents")
print(f"Period 2 ({period2_start}-{period2_end}): {len(df_period2):,} documents")

print(f"\nProcessing Period 1...")
tokens_period1 = []
for idx, text in enumerate(df_period1['Abstract'], 1):
    tokens = preprocess_text(text)
    tokens_period1.extend(tokens)
    if idx % 500 == 0:
        print(f"    {idx:,}/{len(df_period1):,}")

text_period1 = ' '.join(tokens_period1)
print(f"Unique terms Period 1: {len(set(tokens_period1)):,}")

print(f"\nProcessing Period 2...")
tokens_period2 = []
for idx, text in enumerate(df_period2['Abstract'], 1):
    tokens = preprocess_text(text)
    tokens_period2.extend(tokens)
    if idx % 1000 == 0:
        print(f"    {idx:,}/{len(df_period2):,}")

text_period2 = ' '.join(tokens_period2)
print(f"Unique terms Period 2: {len(set(tokens_period2)):,}")

wordcloud_params = {
    'width': 2400,
    'height': 1600,
    'background_color': 'white',
    'max_words': 200,
    'relative_scaling': 0.5,
    'min_font_size': 10,
    'colormap': 'viridis',
    'random_state': 42,
    'collocations': False
}

print("\nGenerating WordClouds...")
wordcloud1 = WordCloud(**wordcloud_params).generate(text_period1)
wordcloud2 = WordCloud(**wordcloud_params).generate(text_period2)

print("\nTop 20 terms - Period 1:")
freq1 = Counter(tokens_period1)
for i, (term, freq) in enumerate(freq1.most_common(20), 1):
    print(f"   {i:2d}. {term:25s} {freq:6,}")

print("\nTop 20 terms - Period 2:")
freq2 = Counter(tokens_period2)
for i, (term, freq) in enumerate(freq2.most_common(20), 1):
    print(f"   {i:2d}. {term:25s} {freq:6,}")

print("\nSAVING COMPARATIVE WORDCLOUDS\n")

output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(24, 16), dpi=600)
ax1.imshow(wordcloud1, interpolation='bilinear')
ax1.set_title(f'Word Cloud: {period1_start}-{period1_end} (Collaborative Filtering Era)', 
              fontsize=40, pad=20)
ax1.axis('off')
plt.tight_layout(pad=0)
plot1_path = output_dir / f'wordcloud_period1_{period1_start}-{period1_end}.png'
plt.savefig(plot1_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

fig2, ax2 = plt.subplots(figsize=(24, 16), dpi=600)
ax2.imshow(wordcloud2, interpolation='bilinear')
ax2.set_title(f'Word Cloud: {period2_start}-{period2_end} (Deep Learning Era)', 
              fontsize=40, pad=20)
ax2.axis('off')
plt.tight_layout(pad=0)
plot2_path = output_dir / f'wordcloud_period2_{period2_start}-{period2_end}.png'
plt.savefig(plot2_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 16), dpi=600)

ax1.imshow(wordcloud1, interpolation='bilinear')
ax1.set_title(f'{period1_start}-{period1_end}\n(Collaborative Filtering Era)', 
              fontsize=36, pad=20, weight='bold')
ax1.axis('off')

ax2.imshow(wordcloud2, interpolation='bilinear')
ax2.set_title(f'{period2_start}-{period2_end}\n(Deep Learning Era)', 
              fontsize=36, pad=20, weight='bold')
ax2.axis('off')

plt.tight_layout(pad=2)
plot_combined_path = output_dir / 'wordcloud_comparative_temporal.png'
plt.savefig(plot_combined_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"WordCloud Perioada 1: {plot1_path}")
print(f"WordCloud Perioada 2: {plot2_path}")
print(f"WordCloud Comparativ: {plot_combined_path}")

print("\nTEMPORAL DIFFERENCE ANALYSIS\n")

top_period1 = set([term for term, _ in freq1.most_common(50)])
top_period2 = set([term for term, _ in freq2.most_common(50)])

emerging_terms = top_period2 - top_period1
declining_terms = top_period1 - top_period2
persistent_terms = top_period1 & top_period2

print(f"Persistent terms (both periods): {len(persistent_terms)}")
print(f"    Examples: {', '.join(list(persistent_terms)[:10])}")

print(f"\nEmerging terms (predominant in {period2_start}-{period2_end}): {len(emerging_terms)}")
print(f"    Examples: {', '.join(list(emerging_terms)[:10])}")

print(f"\nDeclining terms (predominant in {period1_start}-{period1_end}): {len(declining_terms)}")
print(f"    Examples: {', '.join(list(declining_terms)[:10])}")

comparative_stats = {
    'periods': {
        'period1': {'start': period1_start, 'end': period1_end, 'documents': len(df_period1)},
        'period2': {'start': period2_start, 'end': period2_end, 'documents': len(df_period2)}
    },
    'top_terms_period1': [{'term': t, 'freq': f} for t, f in freq1.most_common(50)],
    'top_terms_period2': [{'term': t, 'freq': f} for t, f in freq2.most_common(50)],
    'comparative_analysis': {
        'persistent_terms': list(persistent_terms),
        'emerging_terms': list(emerging_terms),
        'declining_terms': list(declining_terms)
    }
}

json_comp_path = output_dir / 'wordcloud_comparative_analysis.json'
with open(json_comp_path, 'w', encoding='utf-8') as f:
    json.dump(comparative_stats, f, indent=2, ensure_ascii=False)

print(f"\nComparative analysis saved: {json_comp_path}")

print("\nCOMPLETE - COMPARATIVE WORDCLOUDS GENERATED")
