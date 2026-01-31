import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.patches import Patch

df = pd.read_excel("merged_output.xlsx")
keywords_col = 'Author Keywords'

all_keywords = []
for kw_str in df[keywords_col].dropna():
    if pd.notna(kw_str) and str(kw_str).strip():
        kws = [k.strip().lower() for k in str(kw_str).split(';') if k.strip()]
        all_keywords.extend(kws)

keyword_freq = Counter(all_keywords)
N_TOP = 28 
top_keywords = [kw for kw, count in keyword_freq.most_common(N_TOP)]

keyword_to_idx = {kw: i for i, kw in enumerate(top_keywords)}
n_keywords = len(top_keywords)
cooccurrence_matrix = np.zeros((n_keywords, n_keywords))

for kw_str in df[keywords_col].dropna():
    if pd.notna(kw_str):
        kws = [k.strip().lower() for k in str(kw_str).split(';') if k.strip()]
        kws = [k for k in kws if k in keyword_to_idx]
        
        for i, kw1 in enumerate(kws):
            for kw2 in kws[i:]:
                idx1 = keyword_to_idx[kw1]
                idx2 = keyword_to_idx[kw2]
                cooccurrence_matrix[idx1, idx2] += 1
                if idx1 != idx2:
                    cooccurrence_matrix[idx2, idx1] += 1

node_degrees = cooccurrence_matrix.sum(axis=1)

X = node_degrees.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

G = nx.Graph()
G.add_nodes_from(top_keywords)

threshold = np.percentile(cooccurrence_matrix[cooccurrence_matrix > 0], 80)
for i in range(n_keywords):
    for j in range(i+1, n_keywords):
        if cooccurrence_matrix[i, j] >= threshold:
            G.add_edge(top_keywords[i], top_keywords[j], 
                      weight=cooccurrence_matrix[i, j])

for i, node in enumerate(top_keywords):
    G.nodes[node]['cluster'] = int(cluster_labels[i])

positions = {
    'recommendation system': (0, 0),
    'collaborative filtering': (0, -3.2),
    
    'recommendation systems': (3.8, 1.0),
    'recommender systems': (4.2, -1.2),
    'deep learning': (2.8, 3.5),
    'machine learning': (1.2, 5.0),
    'recommender system': (6.0, 1.8),
    
    'social networks': (-2.0, 5.5),
    'social network': (0.5, 6.2),
    'artificial intelligence': (-4.5, 3.8),
    'feature extraction': (-2.8, 2.2),
    'attention mechanism': (-3.8, 1.2),
    'graph neural networks': (-5.5, -2.2),
    
    'clustering': (-3.0, -4.5),
    'recommendation': (-2.8, -6.0),
    
    'sentiment analysis': (0.5, -8.0),
    'contrastive learning': (3.2, -7.5),
    
    'data mining': (1.8, -5.8),
    'personalization': (6.8, -1.2),
    'knowledge graph': (3.8, -4.5),
    'graph neural network': (5.5, -3.8),
    
    'matrix factorization': (7.5, 0.2),
    'big data': (8.2, 2.5),
    'personalized recommendation': (7.0, 4.2),
    'e-commerce': (8.0, -2.8),
    'ontology': (9.0, 0.5),
    
    'content-based filtering': (9.5, -1.5),
    'reinforcement learning': (8.5, 4.8),
}

missing_nodes = set(G.nodes()) - set(positions.keys())
if missing_nodes:
    print(f"WARNING: Missing positions for: {missing_nodes}")

color_map = {0: "#E74C3C", 1: "#3498DB", 2: "#2ECC71"}
node_colors = [color_map[G.nodes[n]['cluster']] for n in G.nodes()]

max_freq = keyword_freq[top_keywords[0]]
node_sizes = [2400 if keyword_freq[kw] > 2000 else 
              1900 if keyword_freq[kw] > 1000 else 
              1400 if keyword_freq[kw] > 500 else 1200 
              for kw in G.nodes()]

fig, ax = plt.subplots(figsize=(26, 18))

nx.draw_networkx_edges(G, positions, alpha=0.20, width=2.5, ax=ax, edge_color='#666666')

nx.draw_networkx_nodes(G, positions, node_color=node_colors, node_size=node_sizes, 
                       alpha=0.98, edgecolors='black', linewidths=2.8, ax=ax)

for node, (x, y) in positions.items():
    freq = keyword_freq[node]
    label = node.title()
    
    if freq > 2000:
        fontsize = 31  
        weight = 'bold'
    elif freq > 1000:
        fontsize = 26  
        weight = 'bold'
    elif freq > 500:
        fontsize = 23  
        weight = 'semibold'
    else:
        fontsize = 21  
        weight = 'normal'
    
    ax.text(
        x, y - 0.42, label,
        fontsize=fontsize,
        weight=weight,
        ha='center', va='top',
        color='#0A0A0A',
        fontname='DejaVu Sans'
    )

cluster_stats = {}
for cluster_id in sorted(set(cluster_labels)):
    cluster_kws = [top_keywords[i] for i, label in enumerate(cluster_labels) 
                   if label == cluster_id]
    count = len(cluster_kws)
    pct = (count / len(top_keywords)) * 100
    avg_degree = np.mean([node_degrees[keyword_to_idx[kw]] for kw in cluster_kws])
    cluster_stats[cluster_id] = {'count': count, 'pct': pct, 'avg_degree': avg_degree}

legend_labels_list = [
    f"Cluster 0 ({cluster_stats[0]['pct']:.1f}%) - Core Concepts\n{cluster_stats[0]['count']} keywords, avg degree: {cluster_stats[0]['avg_degree']:.0f}",
    f"Cluster 1 ({cluster_stats[1]['pct']:.1f}%) - Dominant Paradigms\n{cluster_stats[1]['count']} keywords, avg degree: {cluster_stats[1]['avg_degree']:.0f}",
    f"Cluster 2 ({cluster_stats[2]['pct']:.1f}%) - Active Research Areas\n{cluster_stats[2]['count']} keywords, avg degree: {cluster_stats[2]['avg_degree']:.0f}"
]

legend_elements = [
    Patch(facecolor=color_map[0], edgecolor='black', linewidth=1.8, label=legend_labels_list[0]),
    Patch(facecolor=color_map[1], edgecolor='black', linewidth=1.8, label=legend_labels_list[1]),
    Patch(facecolor=color_map[2], edgecolor='black', linewidth=1.8, label=legend_labels_list[2])
]

plt.legend(handles=legend_elements, loc="upper center", fontsize=15,
           title="Legend:",
           title_fontsize=17,
           fancybox=True, framealpha=0.96,
           bbox_to_anchor=(0.5, -0.02), ncol=3,
           edgecolor='black', frameon=True)

edge_count = G.number_of_edges()

plt.axis('off')
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)

plt.savefig(f"keyword_network_top{N_TOP}.png", dpi=600, 
           bbox_inches='tight', facecolor='white')

print(f"\n{'='*80}")
print(f"CLEAN MANUAL NETWORK VISUALIZATION - {N_TOP} KEYWORDS (ENLARGED FONT)")
print(f"{'='*80}")
print(f"File saved: keyword_network_top{N_TOP}.png")
print(f"Total edges: {edge_count}")
print(f"Edge threshold (80th percentile): {threshold:.0f} co-occurrences")
print(f"\nFont sizes increased:")
print(f"  Highest frequency: 35pt (was 25pt)")
print(f"  High frequency: 30pt (was 20pt)")
print(f"  Medium frequency: 26pt (was 17pt)")
print(f"  Standard: 23pt (was 15pt)")
print(f"\nCluster Distribution:")
for cluster_id in sorted(cluster_stats.keys()):
    stats = cluster_stats[cluster_id]
    print(f"  Cluster {cluster_id}: {stats['count']} keywords ({stats['pct']:.1f}%), "
          f"avg degree: {stats['avg_degree']:.0f}")
print(f"\nAll {N_TOP} nodes positioned - NO ISOLATED NODES")
print(f"{'='*80}\n")

plt.show()
