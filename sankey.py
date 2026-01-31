import os
import sys
import io
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from nltk.stem import WordNetLemmatizer
import nltk

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.makedirs("plots", exist_ok=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

def generate_sankey_html():
    print("=" * 60)
    print("STEP 1: Generating Sankey Diagram HTML")
    print("=" * 60 + "\n")
    
    file_path = "merged_output.xlsx"
    data = pd.read_excel(file_path)

    expected_cols = ['Author Keywords', 'Publication Year', 'Research Areas']
    missing_cols = [col for col in expected_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    synonym_mapping = {
        "generative artificial intelligence": "generative ai",
        "gen ai": "generative ai",
        "genai": "generative ai",
        "generative a.i.": "generative ai",
        "gen-ai": "generative ai"
    }

    def clean_keyword_list(keyword_list):
        if isinstance(keyword_list, list):
            cleaned_list = []
            for keyword in keyword_list:
                cleaned_keyword = ' '.join([lemmatizer.lemmatize(word) for word in keyword.lower().split()])
                for phrase, standard in synonym_mapping.items():
                    cleaned_keyword = cleaned_keyword.replace(phrase, standard)
                cleaned_list.append(cleaned_keyword.strip())
            return cleaned_list
        else:
            return keyword_list

    keywords_data = data[['Author Keywords', 'Publication Year', 'Research Areas']].copy()

    keywords_data['Keywords'] = keywords_data['Author Keywords'].astype(str).str.split(';')
    keywords_data['Keywords'] = keywords_data['Keywords'].apply(clean_keyword_list)
    keywords_data = keywords_data.explode('Keywords')

    keywords_data['Research Areas'] = keywords_data['Research Areas'].astype(str).str.split(';')
    keywords_data = keywords_data.explode('Research Areas')

    keywords_data['Keywords'] = keywords_data['Keywords'].str.strip()
    keywords_data['Research Areas'] = keywords_data['Research Areas'].str.strip()
    keywords_data = keywords_data.dropna(subset=['Keywords', 'Research Areas', 'Publication Year'])

    genai_terms = [
        "generative ai",
        "generative model",
        "diffusion model",
        "large language model",
        "llm",
        "gpt",
        "stable diffusion",
        "text-to-image",
        "image generation"
    ]

    pattern = '|'.join(genai_terms)
    genai_data = keywords_data[keywords_data['Keywords'].str.contains(pattern, case=False, na=False)]
    genai_data = genai_data[genai_data['Publication Year'].between(2023, 2025)]

    print("Available years in filtered data:")
    print(genai_data['Publication Year'].value_counts().sort_index())
    print(f"\nTotal articles: {len(genai_data)}")

    grouped = genai_data.groupby(
        ['Keywords', 'Publication Year', 'Research Areas']
    ).size().reset_index(name='Count')

    print("\nDistribution by year after aggregation:")
    print(grouped.groupby('Publication Year')['Count'].sum().sort_index())

    def simplify_keyword(kw):
        kw_lower = kw.lower()
        if any(term in kw_lower for term in ['generative ai', 'genai', 'gen ai']):
            return 'generative ai'
        elif any(term in kw_lower for term in ['large language model', 'llm']):
            return 'large language models'
        elif 'diffusion' in kw_lower:
            return 'diffusion models'
        elif 'generative model' in kw_lower:
            return 'generative models'
        else:
            return kw

    grouped['Keywords'] = grouped['Keywords'].apply(simplify_keyword)

    grouped = grouped.groupby(
        ['Keywords', 'Publication Year', 'Research Areas']
    )['Count'].sum().reset_index()

    top_keywords = grouped.groupby('Keywords')['Count'].sum().nlargest(4).index
    top_research_areas = grouped.groupby('Research Areas')['Count'].sum().nlargest(10).index

    filtered = grouped[
        (grouped['Keywords'].isin(top_keywords)) &
        (grouped['Research Areas'].isin(top_research_areas))
    ]

    print("\nFinal distribution (after filtering top keywords and domains):")
    print(filtered.groupby('Publication Year')['Count'].sum().sort_index())

    keywords = filtered['Keywords'].unique().tolist()
    years = [2023, 2024, 2025]
    areas = filtered['Research Areas'].unique().tolist()
    labels = keywords + [str(y) for y in years] + areas

    keyword_colors = ["#3498db"] * len(keywords)
    year_colors = {
        2023: "#e91e63",
        2024: "#9b59b6",
        2025: "#f39c12"
    }
    year_color_list = [year_colors[y] for y in years]

    area_colors = ["#2ecc71", "#e67e22", "#1abc9c", "#34495e", "#16a085",
                   "#27ae60", "#2980b9", "#8e44ad", "#c0392b", "#d35400",
                   "#7f8c8d", "#95a5a6", "#bdc3c7", "#ecf0f1", "#e8daef"]
    area_color_list = area_colors[:len(areas)]

    node_colors = keyword_colors + year_color_list + area_color_list

    source, target, value = [], [], []

    for _, row in filtered.iterrows():
        kw_idx = labels.index(row['Keywords'])
        year_idx = labels.index(str(int(row['Publication Year'])))
        area_idx = labels.index(row['Research Areas'])

        source.append(kw_idx)
        target.append(year_idx)
        value.append(row['Count'])

        source.append(year_idx)
        target.append(area_idx)
        value.append(row['Count'])

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20,
            thickness=15,
            line=dict(color="white", width=1),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(150,150,150,0.15)"
        )
    ))

    fig.update_layout(
        font=dict(size=12, family="Arial"),
        height=700,
        width=1400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    output_path = "plots/sankey_genai_evolution_optimized.html"
    fig.write_html(output_path)
    print(f"\n[OK] HTML diagram saved to: {output_path}")
    
    return output_path


def render_sankey_to_png():
    print("\n" + "=" * 60)
    print("STEP 2: Rendering Sankey to PNG (600 DPI)")
    print("=" * 60 + "\n")
    
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    
    html_file = Path("plots/sankey_genai_evolution_optimized.html").absolute()
    output_file = Path("plots/sankey_genai_evolution_optimized_600dpi.png").absolute()
    
    print("Rendering Sankey at full resolution...\n")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    width, height = 2400, 1600
    chrome_options.add_argument(f"--window-size={width},{height}")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        file_url = f"file:///{html_file}".replace("\\", "/")
        print(f"[1/6] Loading HTML...")
        driver.get(file_url)
        
        print(f"[2/6] Waiting for Plotly render...")
        time.sleep(3)
        
        print(f"[3/6] Adjusting window size...")
        driver.set_window_size(width, height)
        time.sleep(1)
        
        print(f"[4/6] Maximizing Plotly diagram...")
        driver.execute_script("""
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
            document.body.style.width = '100vw';
            document.body.style.height = '100vh';
            
            var plotDiv = document.querySelector('div[id]');
            if (plotDiv) {
                plotDiv.style.width = '100vw';
                plotDiv.style.height = '100vh';
                plotDiv.style.margin = '0';
                plotDiv.style.padding = '0';
                
                var layout = plotDiv.data ? plotDiv.layout : null;
                if (layout) {
                    layout.margin = {l: 50, r: 50, t: 20, b: 20};
                    Plotly.relayout(plotDiv, layout);
                }
            }
        """)
        
        time.sleep(2)
        
        print(f"[5/6] Taking screenshot...")
        driver.save_screenshot(str(output_file))
        
        print(f"[6/6] Optimizing PNG...")
        
        try:
            from PIL import Image
            img = Image.open(output_file)
            original_size = img.size
            
            img_array = img.convert('RGB')
            width_px, height_px = img_array.size
            
            left, top, right, bottom = 0, 0, width_px, height_px
            
            for x in range(width_px):
                is_white = True
                for y in range(height_px):
                    r, g, b = img_array.getpixel((x, y))
                    if (r, g, b) != (255, 255, 255):
                        is_white = False
                        break
                if not is_white:
                    left = x
                    break
            
            for x in range(width_px - 1, -1, -1):
                is_white = True
                for y in range(height_px):
                    r, g, b = img_array.getpixel((x, y))
                    if (r, g, b) != (255, 255, 255):
                        is_white = False
                        break
                if not is_white:
                    right = x
                    break
            
            for y in range(height_px):
                is_white = True
                for x in range(width_px):
                    r, g, b = img_array.getpixel((x, y))
                    if (r, g, b) != (255, 255, 255):
                        is_white = False
                        break
                if not is_white:
                    top = y
                    break
            
            for y in range(height_px - 1, -1, -1):
                is_white = True
                for x in range(width_px):
                    r, g, b = img_array.getpixel((x, y))
                    if (r, g, b) != (255, 255, 255):
                        is_white = False
                        break
                if not is_white:
                    bottom = y
                    break
            
            padding = 30
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(width_px, right + padding)
            bottom = min(height_px, bottom + padding)
            
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_file, 'PNG', optimize=True, dpi=(600, 600))
            
            new_size = cropped.size
            print(f"\n[OK] Cropped from {original_size} to {new_size}")
            
        except ImportError:
            print("\n(PIL not available, skipping crop optimization)")
        
        if output_file.exists():
            size_kb = output_file.stat().st_size / 1024
            
            print(f"\n{'='*60}")
            print(f"SUCCESS - SANKEY DIAGRAM COMPLETE!")
            print(f"{'='*60}")
            print(f"PNG File: {output_file.name}")
            print(f"Location: plots/")
            print(f"Resolution: ~{width}x{height} pixels (cropped)")
            print(f"DPI: 600")
            print(f"File size: {size_kb:.1f} KB")
            print(f"{'='*60}\n")
            
            return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        driver.quit()


if __name__ == "__main__":
    generate_sankey_html()
    render_sankey_to_png()
