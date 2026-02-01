import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Colors
color_teal = '#2E8B8B'
color_orange = '#E85D3C'
color_yellow = '#F5A623'
color_blue = '#3A7CA5'

def get_box_edges(x, y, width, height, offset=0.13):
    return {
        "top": (x, y + height/2 + offset),
        "bottom": (x, y - height/2 - offset),
        "left": (x - width/2 - offset, y),
        "right": (x + width/2 + offset, y),
        "center": (x, y)
    }

def create_box(ax, x, y, width, height, text, color, fontsize=18, bold=True):
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, 
        edgecolor='black', 
        linewidth=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight=weight, wrap=True)

def create_line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2.5)

# Box dimensions
w_start, h_start = 2, 0.8
w_data, h_data = 4.5, 1
w_orange, h_orange = 3.8, 1
w_yellow, h_yellow = 3.5, 0.8
w_network_yellow, h_network_yellow = 3.5, 0.8
w_blue, h_blue = 2.8, 0.8
w_custom_blue, h_custom_blue = 4, 0.8

# Start and Data Collection
start_edges = get_box_edges(5, 18.5, w_start, h_start)
create_box(ax, 5, 18.5, w_start, h_start, 'Start', color_teal, fontsize=20)
data_edges = get_box_edges(5, 16.5, w_data, h_data)
create_box(ax, 5, 16.5, w_data, h_data, 'Data Collection and Filtering\n15,944 Records (1980-2026)', color_teal, fontsize=20)
create_line(ax, start_edges['bottom'][0], start_edges['bottom'][1], data_edges['top'][0], data_edges['top'][1])

# Orange branching boxes
gr_edges = get_box_edges(2.5, 13.5, w_orange, h_orange)
net_edges = get_box_edges(7.5, 13.5, w_orange, h_orange)
create_box(ax, 2.5, 13.5, w_orange, h_orange, 'Graphical Representations\nand Data Visualization', color_orange, fontsize=18)
create_box(ax, 7.5, 13.5, w_orange, h_orange, 'Create Network\nDiagrams', color_orange, fontsize=18)
create_line(ax, data_edges['bottom'][0], data_edges['bottom'][1], gr_edges['top'][0], gr_edges['top'][1])
create_line(ax, data_edges['bottom'][0], data_edges['bottom'][1], net_edges['top'][0], net_edges['top'][1])

# Yellow (sub orange)
viz_edges = get_box_edges(2.5, 11.8, w_yellow, h_yellow)
illus_edges = get_box_edges(7.5, 11.8, w_network_yellow, h_network_yellow)
create_box(ax, 2.5, 11.8, w_yellow, h_yellow, 'Visualize Trends\nand Evolution', color_yellow, fontsize=18)
create_box(ax, 7.5, 11.8, w_network_yellow, h_network_yellow, 'Illustrate Relationships\n(e.g., Sankey Diagram)', color_yellow, fontsize=18)
create_line(ax, gr_edges['bottom'][0], gr_edges['bottom'][1], viz_edges['top'][0], viz_edges['top'][1])
create_line(ax, net_edges['bottom'][0], net_edges['bottom'][1], illus_edges['top'][0], illus_edges['top'][1])

# Blue - sentiment analysis, complet separate!
space = 0.9  # spațiu mare între boxuri, ajustează dacă vrei mai mic/mare
x_left = 5 - w_blue/2 - space/2
x_right = 5 + w_blue/2 + space/2
Y_blue = 9.8

vader_edges = get_box_edges(x_left, Y_blue, w_blue, h_blue)
textblob_edges = get_box_edges(x_right, Y_blue, w_blue, h_blue)
create_box(ax, x_left, Y_blue, w_blue, h_blue, 'Sentiment Analysis\nUsing VADER', color_blue, fontsize=18)
create_box(ax, x_right, Y_blue, w_blue, h_blue, 'Sentiment Analysis\nUsing TextBlob', color_blue, fontsize=18)

create_line(ax, viz_edges['bottom'][0], viz_edges['bottom'][1], vader_edges['top'][0], vader_edges['top'][1])
create_line(ax, illus_edges['bottom'][0], illus_edges['bottom'][1], textblob_edges['top'][0], textblob_edges['top'][1])

# Central blue
custom_edges = get_box_edges(5, 8.3, w_custom_blue, h_custom_blue)
create_box(ax, 5, 8.3, w_custom_blue, h_custom_blue, 'Sentiment Analysis Using\nCustom Word Lists', color_blue, fontsize=18)
create_line(ax, vader_edges['bottom'][0], vader_edges['bottom'][1], custom_edges['top'][0], custom_edges['top'][1])
create_line(ax, textblob_edges['bottom'][0], textblob_edges['bottom'][1], custom_edges['top'][0], custom_edges['top'][1])

# Vertical galben (final)
compare_edges = get_box_edges(5, 6.6, w_yellow, h_yellow)
model_edges = get_box_edges(5, 4.9, w_orange, h_yellow)
extract_edges = get_box_edges(5, 3.2, w_yellow, h_yellow)
create_box(ax, 5, 6.6, w_yellow, h_yellow, 'Compare Sentiment\nResults', color_yellow, fontsize=18)
create_box(ax, 5, 4.9, w_orange, h_yellow, 'Integrate with Topic\nModeling (LDA)', color_yellow, fontsize=18)
create_box(ax, 5, 3.2, w_yellow, h_yellow, 'Extract Insights\nand Trends', color_yellow, fontsize=18)

create_line(ax, custom_edges['bottom'][0], custom_edges['bottom'][1], compare_edges['top'][0], compare_edges['top'][1])
create_line(ax, compare_edges['bottom'][0], compare_edges['bottom'][1], model_edges['top'][0], model_edges['top'][1])
create_line(ax, model_edges['bottom'][0], model_edges['bottom'][1], extract_edges['top'][0], extract_edges['top'][1])

end_edges = get_box_edges(5, 1.5, w_start, h_start)
create_box(ax, 5, 1.5, w_start, h_start, 'End', color_teal, fontsize=20)
create_line(ax, extract_edges['bottom'][0], extract_edges['bottom'][1], end_edges['top'][0], end_edges['top'][1])

plt.tight_layout(pad=0.3)
plt.savefig('research_workflow_flowchart.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
plt.show()
