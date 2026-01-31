import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

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
print("[OK] NLTK ready\n")


file_path = "merged_output.xlsx"
print(f"[*] Reading file: {file_path}")

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"[!] ERROR: File '{file_path}' not found.")
    print("    Please make sure the Excel file is in the same folder.")
    exit()

print(f"    Total records: {len(df)}")
print(f"    Abstract column: 'Abstract'")

df_valid = df[df['Abstract'].notna()].copy()
print(f"    Valid abstracts: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")
print(f"    Missing abstracts: {len(df) - len(df_valid)}")

abstract_lengths = df_valid['Abstract'].str.len()
print(f"\n[*] Abstract length statistics (characters):")
print(f"    Min: {abstract_lengths.min()} | Max: {abstract_lengths.max()}")
print(f"    Mean: {int(abstract_lengths.mean())} | Median: {int(abstract_lengths.median())}")

print("\n" + "="*70)
print("                              TEXT PREPROCESSING")
print("="*70 + "\n")

print("[*] Processing in progress...")
print("    1. Lowercase conversion")
print("    2. Non-text character removal")
print("    3. Tokenization and stopword removal")
print("    4. Lemmatization\n")

stop_words = set(stopwords.words('english'))
custom_stops = {
    'also', 'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
    'additionally', 'propose', 'present', 'show', 'demonstrate', 'introduce',
    'describe', 'discuss', 'analyze', 'examine', 'investigate', 'study',
    'paper', 'article', 'work', 'approach', 'method', 'technique', 'result',
    'experiment', 'evaluation', 'performance', 'accuracy', 'effectiveness',
    'use', 'used', 'using', 'provide', 'based', 'propose', 'proposed'
}
stop_words.update(custom_stops)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return []
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    tokens = text.split()
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    
    return tokens

processed_docs = []
total_docs = len(df_valid)

for idx, text in enumerate(df_valid['Abstract'], 1):
    tokens = preprocess_text(text)
    processed_docs.append(tokens)
    if idx % 2000 == 0:
        print(f"    Progress: {idx}/{total_docs} ({idx/total_docs*100:.1f}%)")

print(f"[OK] Processed: {len(processed_docs)} documents")

min_tokens = 5
valid_docs = [doc for doc in processed_docs if len(doc) >= min_tokens]
print(f"[!] Documents excluded (too short): {len(processed_docs) - len(valid_docs)}")

all_tokens = [token for doc in valid_docs for token in doc]
print(f"\n[*] Vocabulary:")
print(f"    Unique terms: {len(set(all_tokens))}")
print(f"    Total words: {len(all_tokens)}")


print("\n" + "="*70)
print("                         LDA MODEL TRAINING")
print("="*70 + "\n")

processed_texts = [' '.join(doc) for doc in valid_docs]

print("[*] Vectorization (CountVectorizer)...")
vectorizer = CountVectorizer(
    max_features=3000,
    min_df=5,
    max_df=0.8,
    token_pattern=r'\b[a-z]+\b'
)

dtm = vectorizer.fit_transform(processed_texts)
print(f"    DTM Matrix: {dtm.shape[0]} docs x {dtm.shape[1]} terms")

print(f"\n[*] Initializing LDA (5 Topics)...")
n_topics = 5


lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=20,
    learning_method='batch',
    learning_offset=50.,
    doc_topic_prior=0.025,  
    topic_word_prior=0.1,    
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("[*] Training in progress (may take 1-2 minutes)...")
lda_model.fit(dtm)

log_likelihood = lda_model.score(dtm)
perplexity = lda_model.perplexity(dtm)
print(f"\n[OK] Training completed.")
print(f"    Log-Likelihood: {log_likelihood:.2f}")
print(f"    Perplexity: {perplexity:.2f}")

print("\n" + "="*70)
print("                         EXTRACTING RESULTS")
print("="*70 + "\n")

doc_topic_dist = lda_model.transform(dtm)
topic_proportions = doc_topic_dist.mean(axis=0) * 100

feature_names = vectorizer.get_feature_names_out()
n_top_words = 10
topics_data = {}

print("Top words per topic:")
for topic_idx, topic in enumerate(lda_model.components_, 1):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = [topic[i] for i in top_indices]
    
    topics_data[topic_idx] = {
        'words': top_words,
        'weights': top_weights,
        'proportion': topic_proportions[topic_idx - 1]
    }
    
    print(f"    Topic {topic_idx} ({topic_proportions[topic_idx-1]:.1f}%): {', '.join(top_words)}")

output_dir = Path('rezultate_lda')
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(n_topics, 1, figsize=(12, 15))
fig.suptitle('LDA Topic Modeling Results', fontsize=16)

for i in range(1, n_topics + 1):
    ax = axes[i-1]
    words = topics_data[i]['words']
    weights = topics_data[i]['weights']
    
    y_pos = np.arange(len(words))
    ax.barh(y_pos, weights, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f"Topic {i} ({topics_data[i]['proportion']:.1f}%)")

plt.tight_layout()
plot_path = output_dir / 'lda_plot.png'
plt.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white')


csv_data = []
for i in range(1, n_topics + 1):
    csv_data.append({
        'Topic': f'Topic {i}',
        'Proportion': f"{topics_data[i]['proportion']:.2f}%",
        'Terms': ', '.join(topics_data[i]['words'])
    })
pd.DataFrame(csv_data).to_csv(output_dir / 'lda_summary.csv', index=False)

json_path = output_dir / 'lda_results.json'
with open(json_path, 'w') as f:
    json_ready_data = {}
    for k, v in topics_data.items():
        json_ready_data[k] = {
            'words': v['words'],
            'weights': [float(x) for x in v['weights']],
            'proportion': float(v['proportion'])
        }
    json.dump(json_ready_data, f, indent=2)

print("\n" + "="*70)
print("                         COMPLETED")
print("="*70)
print(f"Files saved in folder: {output_dir}")
print(f"1. lda_plot.png")
print(f"2. lda_summary.csv")
print(f"3. lda_results.json")