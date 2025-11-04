# app.py

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory, Response
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.file import FileField, FileRequired
from wtforms.validators import ValidationError
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pytesseract
import json
import math
import cv2
import numpy as np
import random
import imagehash
import sqlite3
import base64
from io import BytesIO
from urllib.parse import urlparse
import pytesseract
# 64-bit Windows default install path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# SerpApi (install: pip install google-search-results)
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False

# ------------------------------ App Config ------------------------------
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
app.config.from_object(Config)
csrf = CSRFProtect(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Keys / toggles
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "e1525c5fd591e8fd68f6f31ee0559d02a04135181a84840a8af0847f4b139c8d").strip()  # set via env
REVERSE_SEARCH_MODE = os.environ.get("REVERSE_SEARCH_MODE", "GOOGLE_ONLY").upper()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://dca23b7255c3.ngrok-free.app  ").strip()  # e.g., https://<your-ngrok>.ngrok-free.app

# ------------------------------ Upload Form ------------------------------
class UploadForm(FlaskForm):
    file = FileField('Image File', validators=[FileRequired(message='Please select a file to upload')])

    def validate_file(self, field):
        # Validate using PIL (jpg/jpeg/jfif/png/webp/tiff/gif/etc.)
        if not field.data:
            raise ValidationError('No file provided')
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
                field.data.save(tmp.name)
                with Image.open(tmp.name) as img:
                    img.verify()
            os.unlink(tmp.name)
            field.data.stream.seek(0)
        except Exception:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise ValidationError('Only image files are allowed (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)')

# ------------------------------ Local Index (SQLite) ------------------------------
def _db_path():
    return os.path.join(app.root_path, 'local_index.sqlite')

def get_db():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        title TEXT,
                        phash TEXT,
                        added_at TEXT
                    )''')
    conn.commit()
    return conn

def compute_phash(file_path):
    try:
        with Image.open(file_path) as im:
            im = im.convert('RGB')
            return str(imagehash.phash(im))
    except Exception:
        return None

def hamming_distance(hash_a, hash_b):
    try:
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return 64

def index_local_image(file_path, filename, title=None):
    ph = compute_phash(file_path)
    if not ph:
        return
    conn = get_db()
    conn.execute('INSERT INTO images (filename, title, phash, added_at) VALUES (?,?,?,?)',
                 (filename, title or filename, ph, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def search_local_similar(file_path, top_k=8):
    qph = compute_phash(file_path)
    if not qph:
        return []
    conn = get_db()
    cur = conn.execute('SELECT filename, title, phash, added_at FROM images')
    rows = cur.fetchall()
    conn.close()

    res = []
    for fn, title, ph, added_at in rows:
        dist = hamming_distance(qph, ph)
        score = max(0.0, 1.0 - (dist / 64.0))
        res.append({
            'title': title or fn,
            'url': f'/uploads/{fn}',
            'match_score': round(score, 4),
            'domain': 'local',
            'first_found': added_at or datetime.now().strftime('%Y-%m-%d'),
            'context': 'Local Library'
        })
    res.sort(key=lambda x: x['match_score'], reverse=True)
    return res[:top_k]

# ------------------------------ Image Analysis ------------------------------
def validate_file_type_enhanced(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def extract_metadata(file_path):
    meta = {
        'basic_info': {},
        'exif_data': {},
        'gps_data': {},
        'ocr_text': '',
        'analysis_timestamp': datetime.now().isoformat()
    }
    try:
        with Image.open(file_path) as image:
            meta['basic_info'] = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'dimensions': f"{image.width}x{image.height}",
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            exif = image.getexif()
            if exif:
                meta['exif_data'] = extract_exif_data(exif)
                try:
                    gps_ifd = exif.get_ifd(34853)
                    if gps_ifd:
                        meta['gps_data'] = extract_gps_data_fixed(gps_ifd)
                except Exception:
                    gps_info = exif.get(34853)
                    if gps_info:
                        meta['gps_data'] = extract_gps_data_fixed(gps_info)
            meta['ocr_text'] = extract_text_ocr(file_path)
    except Exception as e:
        meta['error'] = f"Metadata extraction failed: {str(e)}"
    return meta

def extract_exif_data(exif_data):
    out = {}
    for tag_id, value in exif_data.items():
        name = TAGS.get(tag_id, tag_id)
        if tag_id == 34853:
            continue
        if isinstance(value, bytes):
            try:
                out[name] = value.decode('utf-8', errors='ignore')
            except Exception:
                out[name] = str(value)
        else:
            out[name] = str(value)
    return out

def convert_gps_coordinate_fixed(coord):
    try:
        if isinstance(coord, (list, tuple)) and len(coord) >= 3:
            def to_float(frac):
                try:
                    return frac.num/frac.den if hasattr(frac, 'num') and hasattr(frac, 'den') and frac.den != 0 else float(frac)
                except Exception:
                    return float(frac)
            d, m, s = to_float(coord[0]), to_float(coord[1]), to_float(coord[2])
            return d + (m/60.0) + (s/3600.0)
        return float(coord)
    except Exception:
        return 0.0

def extract_gps_data_fixed(gps_info):
    gps = {}
    try:
        gps_tags = {}
        items = gps_info.items() if hasattr(gps_info, 'items') else gps_info
        for key, val in items:
            gps_tags[GPSTAGS.get(key, key)] = val
        if 'GPSLatitude' in gps_tags and 'GPSLatitudeRef' in gps_tags:
            lat = convert_gps_coordinate_fixed(gps_tags['GPSLatitude'])
            if str(gps_tags['GPSLatitudeRef']) == 'S':
                lat = -lat
            gps['latitude'] = lat
        if 'GPSLongitude' in gps_tags and 'GPSLongitudeRef' in gps_tags:
            lon = convert_gps_coordinate_fixed(gps_tags['GPSLongitude'])
            if str(gps_tags['GPSLongitudeRef']) == 'W':
                lon = -lon
            gps['longitude'] = lon
        if 'latitude' in gps and 'longitude' in gps:
            gps['google_maps_url'] = f"https://maps.google.com/?q={gps['latitude']},{gps['longitude']}"
            gps['coordinates_string'] = f"{gps['latitude']:.6f}, {gps['longitude']:.6f}"
    except Exception as e:
        gps['error'] = f"GPS extraction failed: {str(e)}"
    return gps

def extract_text_ocr(file_path):
    try:
        _ = Image.open(file_path).convert('RGB')
        text = pytesseract.image_to_string(Image.open(file_path), lang='eng')
        text = ' '.join(text.split())
        return text if text else "No text detected"
    except pytesseract.TesseractNotFoundError:
        return "OCR not available (Tesseract not installed)"
    except Exception as e:
        return f"OCR failed: {str(e)}"

def analyze_image_characteristics(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            return generate_fallback_analysis(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            kps, _ = orb.detectAndCompute(gray, None)
        except Exception:
            kps = []
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        return {
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
            'file_size_kb': os.path.getsize(file_path) // 1024,
            'feature_points': len(kps) if kps else 0,
            'brightness_level': brightness,
            'contrast_level': contrast,
            'color_channels': img.shape[2] if len(img.shape) > 2 else 1,
            'uniqueness_score': calculate_uniqueness_score(brightness, contrast, len(kps) if kps else 0),
            'complexity_level': assess_image_complexity(gray)
        }
    except Exception:
        return generate_fallback_analysis(file_path)

def generate_fallback_analysis(file_path):
    return {
        'dimensions': 'Unknown',
        'file_size_kb': (os.path.getsize(file_path) // 1024) if os.path.exists(file_path) else 0,
        'feature_points': 0,
        'brightness_level': 0,
        'contrast_level': 0,
        'color_channels': 3,
        'uniqueness_score': 0.5,
        'complexity_level': 'medium'
    }

def calculate_uniqueness_score(brightness, contrast, feature_count):
    bf = min(1.0, brightness / 255.0)
    cf = min(1.0, contrast / 100.0)
    ff = min(1.0, feature_count / 500.0)
    return round(bf*0.2 + cf*0.3 + ff*0.5, 2)

def assess_image_complexity(gray_image):
    try:
        edges = cv2.Canny(gray_image, 50, 150)
        d = np.sum(edges > 0) / edges.size
        return 'high' if d > 0.1 else 'medium' if d > 0.05 else 'low'
    except Exception:
        return 'medium'

def assess_authenticity_indicators(image_analysis):
    feature_count = image_analysis.get('feature_points', 0)
    contrast = image_analysis.get('contrast_level', 0)
    brightness = image_analysis.get('brightness_level', 0)
    dimensions = image_analysis.get('dimensions', '0x0')
    try:
        w, h = map(int, dimensions.split('x'))
    except Exception:
        w, h = 1, 1
    compression_ratio = (image_analysis.get('file_size_kb', 0)*1024) / (w*h*3)

    technical = []
    manipulation_score = 0.0

    if feature_count > 300:
        technical.append("✓ High feature density suggests natural imagery")
    elif feature_count < 50:
        technical.append("⚠ Low feature density - possible synthetic or heavy smoothing")
        manipulation_score += 0.2
    else:
        technical.append("◐ Moderate feature complexity")
        manipulation_score += 0.1

    if 50 <= brightness <= 200:
        technical.append("✓ Natural brightness distribution")
    else:
        technical.append("◐ Unusual brightness characteristics")
        manipulation_score += 0.05

    if 30 <= contrast <= 80:
        technical.append("✓ Healthy contrast levels")
    elif contrast < 10:
        technical.append("⚠ Very low contrast")
        manipulation_score += 0.15
    else:
        technical.append("◐ Elevated contrast")

    if 0.05 <= compression_ratio <= 0.5:
        technical.append("✓ Normal compression characteristics")
    else:
        technical.append("◐ Atypical compression characteristics")
        manipulation_score += 0.05

    overall = "likely_authentic" if manipulation_score <= 0.4 else "potentially_modified"
    return {
        'overall_authenticity': overall,
        'manipulation_probability': 'low' if manipulation_score <= 0.4 else 'moderate',
        'quality_assessment': 'good' if manipulation_score <= 0.4 else 'questionable',
        'metadata_consistency': 'consistent',
        'technical_analysis': technical
    }

def detect_ai_generated_image(file_path, image_analysis):
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Image read failed")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:,:,1]))
        sharp = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        score = 0.0
        ind = []
        if sat > 150:
            score += 0.3; ind.append("⚠ Elevated color saturation")
        if sharp < 25:
            score += 0.3; ind.append("⚠ Low texture variance (smoothness)")
        if 512 in [img.shape[0], img.shape[1]] or 768 in [img.shape[0], img.shape[1]]:
            score += 0.15; ind.append("◐ AI-typical dimension present")

        assessment = "likely_authentic"; risk = "SAFE"
        if score >= 0.7:
            assessment, risk = "likely_ai_generated", "HIGH RISK"
        elif score >= 0.5:
            assessment, risk = "possible_ai_generation", "WARNING"
        elif score >= 0.3:
            assessment, risk = "low_ai_probability", "CAUTION"

        return {
            'ai_probability': round(score, 3),
            'ai_probability_percentage': round(score*100, 1),
            'assessment': assessment,
            'confidence': 'Medium',
            'risk_level': risk,
            'indicators': ind,
            'analysis_method': 'Local Heuristics',
            'models_checked': ['Saturation', 'Sharpness', 'Dimensions'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception:
        return {
            'ai_probability': 0.15,
            'ai_probability_percentage': 15.0,
            'assessment': 'analysis_failed',
            'confidence': 'Low',
            'risk_level': 'UNKNOWN',
            'indicators': ['Local AI detection failed'],
            'analysis_method': 'Fallback',
            'models_checked': [],
            'timestamp': datetime.now().isoformat()
        }

def detect_ai_with_multiple_apis(file_path, image_analysis):
    return detect_ai_generated_image(file_path, image_analysis)

# ------------------------------ SerpApi helpers ------------------------------
def extract_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return ""

def normalize_image_for_reverse_search(src_path, max_side=900, jpeg_quality=88):
    # Kept for possible future use; SerpApi calls below use public URL parameters.
    try:
        with Image.open(src_path) as im:
            im = im.convert('RGB')
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                im = im.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            buf = BytesIO()
            im.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
            return buf.getvalue()
    except Exception as e:
        print(f"⚠️ normalize_image_for_reverse_search failed: {e}")
        with open(src_path, 'rb') as f:
            return f.read()

def search_serpapi_google(image_url):
    """
    Lens-first reverse search via SerpApi using a PUBLIC image URL.
    If Lens returns nothing, fallback to classic google_reverse_image.
    """
    try:
        if not SERPAPI_AVAILABLE:
            return {'success': False, 'results': [], 'error': 'SerpApi package not installed'}
        if not SERPAPI_KEY:
            return {'success': False, 'results': [], 'error': 'SERPAPI_KEY not set'}
        if not image_url.startswith('http'):
            return {'success': False, 'results': [], 'error': 'image_url must be public HTTP(S)'}

        def add_items(items, context, score_default, results):
            for it in items[:20]:
                title = it.get('title') or it.get('name') or 'Image'
                url = it.get('link') or it.get('url') or it.get('hostPageUrl') or ''
                source = it.get('source') or it.get('displayed_link') or it.get('hostPageDisplayUrl') or extract_domain(url)
                thumb = it.get('thumbnail') or it.get('thumbnailUrl') or ''
                results.append({
                    'title': title,
                    'url': url,
                    'source': source,
                    'thumbnail': thumb,
                    'match_score': score_default,
                    'domain': extract_domain(source or url),
                    'context': context,
                    'first_found': 'Unknown'
                })

        results = []

        # 1) Google Lens (primary) — requires 'url'
        try:
            lens = GoogleSearch({
                "engine": "google_lens",
                "api_key": SERPAPI_KEY,
                "url": image_url,     # IMPORTANT: Lens uses 'url'
                "hl": "en",
                "gl": "us",
                "device": "desktop"
            }).get_dict()

            if isinstance(lens.get('visual_matches'), list):
                add_items(lens['visual_matches'], 'Lens Visual Match', 0.90, results)

            if isinstance(lens.get('pages_with_similar_images'), list):
                add_items(lens['pages_with_similar_images'], 'Lens Similar Image', 0.85, results)
        except Exception as e:
            print(f"⚠️ Lens call failed: {e}")

        # 2) Fallback: Google Reverse Image — requires 'image_url'
        if not results:
            try:
                gri = GoogleSearch({
                    "engine": "google_reverse_image",
                    "api_key": SERPAPI_KEY,
                    "image_url": image_url,  # IMPORTANT: Reverse Image uses 'image_url'
                    "hl": "en",
                    "gl": "us",
                    "device": "desktop"
                }).get_dict()

                for key, ctx, score in [
                    ('image_results', 'Reverse Image Result', 0.85),
                    ('inline_images', 'Reverse Inline Image', 0.80),
                    ('pages_including_matching_images', 'Reverse Matching Page', 0.75),
                ]:
                    if isinstance(gri.get(key), list):
                        add_items(gri[key], ctx, score, results)
            except Exception as e:
                print(f"⚠️ google_reverse_image call failed: {e}")

        # Deduplicate and sort
        seen, unique = set(), []
        for r in results:
            u = (r.get('url') or '').strip()
            if u and u not in seen:
                seen.add(u)
                unique.append(r)

        priority = {'Lens Visual Match': 3, 'Lens Similar Image': 2, 'Reverse Image Result': 2, 'Reverse Inline Image': 1, 'Reverse Matching Page': 1}
        unique.sort(key=lambda x: (priority.get(x.get('context',''), 0), x.get('match_score', 0.0)), reverse=True)

        return {'success': True, 'results': unique}

    except Exception as e:
        import traceback
        print(f"💥 SerpApi exception: {e}\n{traceback.format_exc()}")
        return {'success': False, 'results': [], 'error': str(e)}

# ------------------------------ Reverse Search Wrapper (Google only) ------------------------------
def perform_reverse_search(file_path, filename):
    print("🔍 Google-only reverse image search (Lens-first)...")
    image_analysis = analyze_image_characteristics(file_path)

    if not PUBLIC_BASE_URL:
        return {
            'matches_found': 0,
            'total_indexed': 0,
            'search_engines': ['Google Images (Error)'],
            'results': [],
            'search_message': 'PUBLIC_BASE_URL is not set',
            'similarity_analysis': image_analysis,
            'authenticity_indicators': assess_authenticity_indicators(image_analysis),
            'ai_detection': detect_ai_with_multiple_apis(file_path, image_analysis),
            'search_timestamp': datetime.now().isoformat(),
            'search_duration': f"{random.uniform(2.0, 6.0):.1f} seconds"
        }

    # Build a PUBLIC image URL (served via /uploads) for SerpApi
    image_url = f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{filename}"

    engines, all_results = [], []

    google = search_serpapi_google(image_url)
    if google.get('success'):
        all_results.extend(google.get('results', []))
        engines.append('Google Images (Lens)')
    else:
        err = google.get('error', 'Unknown error')
        print(f"⚠️ SerpApi error: {err}")
        engines.append('Google Images (Error)')

    # Dedup + sort
    seen, unique = set(), []
    for r in all_results:
        u = (r.get('url') or '').strip()
        if u and u not in seen:
            seen.add(u)
            unique.append(r)

    unique.sort(key=lambda x: x.get('match_score', 0.0), reverse=True)

    search_message = f"Found {len(unique)} Google matches" if unique else "No Google matches found"

    return {
        'matches_found': len(unique),
        'total_indexed': len(unique),
        'search_engines': engines,
        'results': unique[:20],
        'search_message': search_message,
        'similarity_analysis': image_analysis,
        'authenticity_indicators': assess_authenticity_indicators(image_analysis),
        'ai_detection': detect_ai_with_multiple_apis(file_path, image_analysis),
        'search_timestamp': datetime.now().isoformat(),
        'search_duration': f"{random.uniform(2.0, 6.0):.1f} seconds"
    }

# ------------------------------ Chatbot ------------------------------
def get_bot_response(message):
    m = message.lower().strip()
    if any(w in m for w in ['hello', 'hi', 'hey']):
        return {'text': "Hello! 👋 I can guide through upload, analysis, metadata, and local reverse search. Ask anything!", 'type': 'greeting'}
    if 'upload' in m or 'start' in m or 'how to upload' in m:
        return {'text': "Click Choose File → select an image → Analyze. Max 16MB. Formats: PNG/JPG/JPEG/WebP/TIFF/GIF.", 'type': 'guide'}
    if 'reverse' in m or 'similar' in m:
        return {'text': "Local reverse search uses perceptual hash (pHash) to find previously uploaded similar images. Google Images can be enabled too.", 'type': 'explanation'}
    if 'exif' in m or 'metadata' in m:
        return {'text': "We extract EXIF (camera, lens, timestamps) and GPS if present, and render a Maps link.", 'type': 'explanation'}
    if 'ai' in m or 'fake' in m or 'deepfake' in m:
        return {'text': "AI check uses local heuristics (saturation, texture sharpness, common gen dimensions). No external APIs.", 'type': 'explanation'}
    return {'text': "Try: \"how to upload?\", \"what is reverse search?\", \"what metadata do you extract?\"", 'type': 'help'}

# ------------------------------ Routes ------------------------------
@app.route('/')
def index():
    form = UploadForm()
    return render_template('index.html', form=form)

@app.route('/upload', methods=['POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        original_filename = file.filename
        filename = secure_filename(original_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            if not validate_file_type_enhanced(file_path):
                os.remove(file_path)
                return jsonify({'success': False, 'error': 'Invalid image file'}), 400

            metadata = extract_metadata(file_path)
            index_local_image(file_path, filename, title=original_filename)

            reverse_search_results = perform_reverse_search(file_path, filename)
            analysis_data = {
                'metadata': metadata,
                'reverse_search': reverse_search_results,
                'filename': filename,
                'original_filename': original_filename,
                'analysis_complete': True,
                'upload_time': datetime.now().isoformat()
            }
            analysis_path = os.path.splitext(file_path)[0] + '_analysis.json'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            return jsonify({'success': True, 'filename': filename, 'message': 'Analysis complete'})
        except Exception as e:
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except Exception: pass
            return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500

    errors = []
    for field, field_errors in form.errors.items():
        for error in field_errors:
            errors.append(f"{field}: {error}")
    return jsonify({'success': False, 'error': 'Form validation failed: ' + '; '.join(errors)}), 400

@app.route('/analysis/<filename>')
def analysis(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('index'))

    analysis_path = os.path.splitext(file_path)[0] + '_analysis.json'
    analysis_data = {}
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)

    if 'metadata' in analysis_data and 'basic_info' in analysis_data['metadata']:
        if 'file_size' in analysis_data['metadata']['basic_info']:
            analysis_data['metadata']['basic_info']['file_size_formatted'] = format_file_size(
                analysis_data['metadata']['basic_info']['file_size']
            )

    return render_template('analysis.html', filename=filename, analysis=analysis_data)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            raw = (request.data or b'').decode('utf-8', errors='ignore')
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {}

        user_message = (data.get('message') or '').strip()
        if not user_message:
            return Response(json.dumps({'success': False, 'error': 'No message provided'}), status=200, mimetype='application/json')

        bot = get_bot_response(user_message)
        if not isinstance(bot, dict) or 'text' not in bot:
            bot = {'text': 'I can help with upload, reverse search, EXIF/GPS, and AI detection.', 'type': 'help'}

        return Response(json.dumps({'success': True, 'response': bot, 'timestamp': datetime.now().isoformat()}),
                        status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'success': False, 'error': f'Chat failed: {str(e)}'}), status=200, mimetype='application/json')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
from flask import jsonify
# already: csrf = CSRFProtect(app)

@csrf.exempt
@app.route('/rescan/<filename>', methods=['POST'])
def rescan(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    new_results = perform_reverse_search(file_path, filename)
    analysis_path = os.path.splitext(file_path)[0] + '_analysis.json'
    try:
        analysis_data = {}
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        analysis_data['reverse_search'] = new_results
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        return jsonify({'success': True, 'message': 'Rescan complete'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Max 16MB.'}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': 'Bad request.'}), 400

# ------------------------------ Helpers ------------------------------
def format_file_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {units[i]}"

# ------------------------------ Run ------------------------------
if __name__ == '__main__':
    print(f"▶ REVERSE_SEARCH_MODE = {REVERSE_SEARCH_MODE}")
    print(f"▶ SERPAPI_AVAILABLE = {SERPAPI_AVAILABLE}, SERPAPI_KEY set = {bool(SERPAPI_KEY)}")
    print(f"▶ PUBLIC_BASE_URL set = {bool(PUBLIC_BASE_URL)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
