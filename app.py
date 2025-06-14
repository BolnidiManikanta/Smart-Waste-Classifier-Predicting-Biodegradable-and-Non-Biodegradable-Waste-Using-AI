from flask import Flask, request, render_template, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = 'waste-classifier-secret-key'  # Required for flash messages

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create a directory for product images if it doesn't exist
PRODUCT_IMAGES_FOLDER = 'static/product_images'
os.makedirs(PRODUCT_IMAGES_FOLDER, exist_ok=True)

# Load model with error handling
model = None
try:
    model = tf.keras.models.load_model('waste_model.keras')
except Exception as e:
    try:
        # Try with compile=False
        model = tf.keras.models.load_model('waste_model.keras', compile=False)
    except Exception as e2:
        print(f"Error loading model: {e2}")
        traceback.print_exc()
        model = None

# Class labels and waste information
class_names = ['(BT) Body Tissue or Organ', '(GE) Glass equipment-packaging 551', '(ME) Metal equipment -packaging',
               '(OW) Organic wastes', '(PE) Plastic equipment-packaging', '(PP) Paper equipment-packaging',
               '(SN) Syringe needles', 'Gauze', 'Gloves', 'Mask', 'Syringe', 'Tweezers']

# Updated waste_info dictionary with product images
waste_info = {
    '(BT) Body Tissue or Organ': ('Biodegradable', [
        {'name': 'Compost', 'image': 'compost.jpg'},
        {'name': 'Fertilizer', 'image': 'fertilizer.jpg'}
    ], 'Requires special handling due to biohazard risks.'),
    
    '(GE) Glass equipment-packaging 551': ('Non-Biodegradable', [
        {'name': 'Recycled glass', 'image': 'recycled_glass.jpg'},
        {'name': 'Tiles', 'image': 'glass_tiles.jpg'}
    ], 'Glass can be recycled indefinitely without loss of quality.'),
    
    '(ME) Metal equipment -packaging': ('Non-Biodegradable', [
        {'name': 'Scrap metal reuse', 'image': 'scrap_metal.jpg'},
        {'name': 'Construction materials', 'image': 'metal_construction.jpg'}
    ], 'Metal recycling saves 75% of energy compared to making new metal.'),
    
    '(OW) Organic wastes': ('Biodegradable', [
        {'name': 'Manure', 'image': 'manure.jpg'},
        {'name': 'Biogas', 'image': 'biogas.jpg'}
    ], 'Can be composted to create nutrient-rich soil.'),
    
    '(PE) Plastic equipment-packaging': ('Non-Biodegradable', [
        {'name': 'Recycled plastic items', 'image': 'recycled_plastic.jpg'}
    ], 'Takes 400+ years to decompose naturally.'),
    
    '(PP) Paper equipment-packaging': ('Biodegradable', [
        {'name': 'Recycled paper', 'image': 'recycled_paper.jpg'},
        {'name': 'Insulation', 'image': 'paper_insulation.jpg'}
    ], 'Can be recycled 5-7 times before fibers become too short.'),
    
    '(SN) Syringe needles': ('Non-Biodegradable', [
        {'name': 'Reprocessed industrial tools', 'image': 'recycled_metal_tools.jpg'}
    ], 'Must be disposed of in sharps containers to prevent injuries.'),
    
    'Gauze': ('Biodegradable', [
        {'name': 'Medical cotton', 'image': 'medical_cotton.jpg'}
    ], 'Should be properly sterilized before disposal.'),
    
    'Gloves': ('Non-Biodegradable', [
        {'name': 'Recycled plastic', 'image': 'recycled_plastic_items.jpg'}
    ], 'Consider using biodegradable alternatives when possible.'),
    
    'Mask': ('Non-Biodegradable', [
        {'name': 'Recycled fibers', 'image': 'recycled_fibers.jpg'}
    ], 'Cut ear loops before disposal to prevent wildlife entanglement.'),
    
    'Syringe': ('Non-Biodegradable', [
        {'name': 'Industrial plastic reuse', 'image': 'industrial_plastic.jpg'}
    ], 'Remove needles before disposal in appropriate containers.'),
    
    'Tweezers': ('Non-Biodegradable', [
        {'name': 'Metal recycling', 'image': 'metal_recycling.jpg'}
    ], 'Can be sterilized and reused multiple times.'),
}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(file_path):
    """Process image and return prediction"""
    try:
        print(f"Starting prediction for {file_path}")
        
        # Verify file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
            
        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        pred = model.predict(img_array, verbose=0)  # Added verbose=0 to reduce console output
        
        # Check if prediction has expected shape
        if pred.shape[0] == 0 or pred.shape[1] != len(class_names):
            print(f"Unexpected prediction shape: {pred.shape}, expected: (1, {len(class_names)})")
            return None
            
        class_index = np.argmax(pred)
        
        if class_index >= len(class_names):
            print(f"Class index out of bounds: {class_index}, max: {len(class_names)-1}")
            return None
            
        confidence = float(pred[0][class_index]) * 100  # Convert to percentage
        prediction = class_names[class_index]
        
        waste_type, products, tips = waste_info.get(prediction, ("Unknown", [], "No information available"))
        
        # Check if product images exist and use placeholder if not
        for product in products:
            product_image_path = os.path.join(PRODUCT_IMAGES_FOLDER, product['image'])
            if not os.path.exists(product_image_path):
                product['image'] = 'placeholder.jpg'  # Use a placeholder image if the product image doesn't exist
                
        # Print biodegradable tips and potential products
        print(f"\n=== WASTE CLASSIFICATION RESULT ===")
        print(f"Prediction: {prediction}")
        print(f"Waste Type: {waste_type}")
        print(f"Disposal Tips: {tips}")
        print(f"Potential Products:")
        for product in products:
            print(f"- {product['name']}")
        print(f"==============================\n")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'waste_type': waste_type,
            'products': products,
            'tips': tips
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the application"""
    result = None
    image_path = ''

    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            flash('Model is not available. Please contact the administrator.', 'error')
            return redirect(request.url)
            
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(file_path)
                
                # Get prediction
                result = get_prediction(file_path)
                if result:
                    image_path = '/' + file_path.replace('\\', '/')  # Ensure correct path format for web
                else:
                    flash('Error processing image.', 'error')
                    
            except Exception as e:
                print(f"File save error: {e}")
                traceback.print_exc()
                flash('Error saving file', 'error')
                
        else:
            flash(f'Allowed file types are: {", ".join(ALLOWED_EXTENSIONS)}', 'error')

    # Pass the current year to the template
    current_year = datetime.now().year
    return render_template('index.html', result=result, image_path=image_path, year=current_year)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large (max 16MB)', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    print(f"Server error: {e}")
    traceback.print_exc()
    flash('Server error occurred', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)