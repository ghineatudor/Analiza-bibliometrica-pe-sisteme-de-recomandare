from pathlib import Path
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def render_sankey_fullscreen():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    
    html_file = Path("plots/sankey_genai_evolution_optimized.html").absolute()
    output_file = Path("plots/sankey_genai_evolution_optimized_600dpi.png").absolute()
    
    print("Rendering Sankey at full resolution (no wasted space)...\n")
    
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
            // Remove all margins and padding from document
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.overflow = 'hidden';
            document.body.style.width = '100vw';
            document.body.style.height = '100vh';
            
            // Find and maximize Plotly container
            var plotDiv = document.querySelector('div[id]');
            if (plotDiv) {
                console.log('Found plot div:', plotDiv.id);
                plotDiv.style.width = '100vw';
                plotDiv.style.height = '100vh';
                plotDiv.style.margin = '0';
                plotDiv.style.padding = '0';
                
                // Adjust Plotly layout margins
                var layout = plotDiv.data ? plotDiv.layout : null;
                if (layout) {
                    layout.margin = {l: 50, r: 50, t: 20, b: 20};
                    Plotly.relayout(plotDiv, layout);
                }
            }
        """)
        
        time.sleep(2)
        
        print(f"[5/6] Cropping excess whitespace...")
        driver.save_screenshot(str(output_file))
        
        print(f"[6/6] Optimizing PNG...")
        
        try:
            from PIL import Image
            img = Image.open(output_file)
            original_size = img.size
            
            img_array = img.convert('RGB')
            
            pixels = list(img_array.getdata())
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
            print(f"\n✓ Cropped from {original_size} to {new_size}")
            
        except ImportError:
            print("\n(PIL not available, skipping crop optimization)")
        
        if output_file.exists():
            size_kb = output_file.stat().st_size / 1024
            
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS - FULL SIZE SANKEY!")
            print(f"{'='*60}")
            print(f"File: {output_file.name}")
            print(f"Location: plots/")
            print(f"Resolution: ~{width}x{height} pixels (cropped)")
            print(f"DPI equivalent: ~600 DPI")
            print(f"File size: {size_kb:.1f} KB")
            print(f"\nThe diagram now fills the entire image!")
            print(f"{'='*60}\n")
            
            return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        driver.quit()

if __name__ == "__main__":
    render_sankey_fullscreen()
