from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import os
import uuid
import pandas as pd
import requests
from bs4 import BeautifulSoup
from preprocessing import preprocess_data
from model_runner import run_model
from utils import generate_confusion_matrix_plot
from eda_report import generate_eda_summary, generate_correlation_plot
from fpdf import FPDF
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

data_cache = {}  # Stores uploaded DataFrames in memory

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Learn Page
@app.route('/learn')
def learn():
    return render_template('learn.html')
@app.route('/quiz')
def quiz():
    return render_template('quiz.html')
@app.route('/aboutus')
def aboutus():
    return render_template('about.html')

# Scraper UI Page
@app.route('/scrape')
def scraper():
    return render_template('scraper.html')

# Upload dataset and return EDA + Correlation plot
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['dataset']
    uid = str(uuid.uuid4())
    filename = uid + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_csv(filepath) if file.filename.endswith('.csv') else pd.read_excel(filepath)
    data_cache[uid] = df

    eda = generate_eda_summary(df)
    corr_path = generate_correlation_plot(df, uid)

    return jsonify({
        "uid": uid,
        "columns": list(df.columns),
        "preview": eda["head"],
        "eda": {
            "shape": eda["shape"],
            "dtypes": eda["dtypes"],
            "missing": eda["missing_values"],
            "describe": eda["describe"],
            "tail": eda["tail"],
            "corr_path": f"/{corr_path}" if corr_path else None
        }
    })

# Train selected model on uploaded dataset
@app.route('/train', methods=['POST'])
def train():
    uid = request.form['uid']
    target = request.form['target']
    algorithm = request.form['algorithm']

    df = data_cache.get(uid)
    if df is None:
        return jsonify({"error": "Data not found"}), 400

    try:
        X_train, X_test, y_train, y_test = preprocess_data(df, target)
        model, acc, y_pred = run_model(X_train, X_test, y_train, y_test, algorithm)
        cm_path = generate_confusion_matrix_plot(y_test, y_pred, uid)

        return jsonify({
            "accuracy": round(acc * 100, 2),
            "confusion_matrix": url_for('serve_plot', filename=os.path.basename(cm_path))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Web scraper endpoint


@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url')

    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Title and meta description
        title = soup.title.string if soup.title else "No title"
        description_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc = description_tag['content'] if description_tag else "No meta description"

        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        meta_keywords = keywords_tag['content'] if keywords_tag else "No keywords found"

        # First paragraph
        first_para_tag = soup.find('p')
        first_paragraph = first_para_tag.get_text(strip=True) if first_para_tag else "No paragraph found"

        # Headings (h1 to h6)
        headings = {f"H{i}": [h.text.strip() for h in soup.find_all(f'h{i}')] for i in range(1, 7)}

        # Hyperlinks
        links = [a['href'] for a in soup.find_all('a', href=True)]

        # Image sources
        images = [img['src'] for img in soup.find_all('img', src=True)]

        return jsonify({
            "title": title,
            "description": meta_desc,
            "keywords": meta_keywords,
            "first_paragraph": first_paragraph,
            "headings": headings,
            "links": links,
            "images": images
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Serve static confusion matrix or correlation plots
@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
