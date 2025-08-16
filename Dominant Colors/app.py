from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from scipy.cluster.vq import kmeans, whiten
import matplotlib
matplotlib.use('Agg')  # For servers without a display
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_dominant_colors(image_path, k=3):
    image = mpimg.imread(image_path)
    
    # Separate channels
    r, g, b = [], [], []
    for row in image:
        for temp_r, temp_g, temp_b in row:
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
    
    # Original std deviations
    r_std_orig = np.std(r)
    g_std_orig = np.std(g)
    b_std_orig = np.std(b)
    
    # Whiten channels
    r_w = whiten(r)
    g_w = whiten(g)
    b_w = whiten(b)
    
    # DataFrame
    df = pd.DataFrame({'r': r_w, 'g': g_w, 'b': b_w})
    
    # KMeans clustering
    cluster_centers, _ = kmeans(df[['r', 'g', 'b']], k)
    
    # Un-whiten & normalize for display
    colors = []
    for scaled_r, scaled_g, scaled_b in cluster_centers:
        colors.append((
            scaled_r * r_std_orig / 255,
            scaled_g * g_std_orig / 255,
            scaled_b * b_std_orig / 255
        ))
    
    # Save palette as an image
    plt.imshow([colors])
    plt.axis('off')
    palette_path = os.path.join(app.config['UPLOAD_FOLDER'], 'palette.png')
    plt.savefig(palette_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return palette_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            palette_path = get_dominant_colors(filepath, k=3)
            return render_template('result.html', image=file.filename, palette='palette.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
