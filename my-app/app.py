import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
load_model = keras.models.load_model
image = keras.preprocessing.image
from PIL import Image
import logging


# Configure environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with FIXED template path
app = Flask(__name__, template_folder='frontend/tempates', static_folder='frontend/static')  
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------
# Configuration and Model Loading
# ---------------------------------------------------------------
MODEL_PATHS = {
    'efficientnet': 'models/EfficientNetV2B0.keras',
    'mobilenet': 'models/animal_classifier_mobilenet.keras',
    'densenet': 'models/best_densenet_model1.keras'
}

INPUT_SIZES = {
    'efficientnet': (512, 512),
    'mobilenet': (224, 224),
    'densenet': (224, 224)
}


ANIMAL_CLASSES = ['Hargila Bok', 'Lama', 'Rabbit', 'ass', 'bear', 'camel', 'camel bird', 'cat', 'cow', 'crocodile', 'deer', 'dog', 'elephant', 'gayal', 'giraffe', 'goat', 'hippopotamus', 'horse', 'kalo bok', 'kangaru', 'lion', 'monkey', 'panda', 'peacock', 'porcupine', 'rhinoceros', 'sheep', 'squirrel', 'tiger', 'zebra']

ANIMAL_DETAILS = {
    'gayal': {
        'scientific': 'Bos frontalis',
        'habitat': 'Hilly forests in South and Southeast Asia',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'A large domesticated bovine found in Northeast India and nearby regions.'
    },
    'giraffe': {
        'scientific': 'Giraffa camelopardalis',
        'habitat': 'African savannas and woodlands',
        'diet': 'Herbivore',
        'status': 'Vulnerable',
        'description': 'Tallest land animals with long necks adapted for browsing tree leaves.'
    },
    'kangaru': {
        'scientific': 'Macropus spp.',
        'habitat': 'Australian grasslands and forests',
        'diet': 'Herbivore',
        'status': 'Least Concern',
        'description': 'Marsupials known for their powerful hind legs and pouches for carrying young.'
    },
    'lion': {
        'scientific': 'Panthera leo',
        'habitat': 'African savannas, grasslands',
        'diet': 'Carnivore',
        'status': 'Vulnerable',
        'description': 'Apex predators known for social behavior and distinctive manes in males.'
    },
    'rhinoceros': {
        'scientific': 'Rhinocerotidae',
        'habitat': 'Grasslands, savannas, and tropical forests',
        'diet': 'Herbivore',
        'status': 'Endangered',
        'description': 'Large, thick-skinned herbivores known for their prominent horns.'
    },
    'sheep': {
        'scientific': 'Ovis aries',
        'habitat': 'Grasslands and pastures',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Domesticated ruminants raised for wool, meat, and milk.'
    },
    'tiger': {
        'scientific': 'Panthera tigris',
        'habitat': 'Tropical forests, grasslands',
        'diet': 'Carnivore',
        'status': 'Endangered',
        'description': 'Solitary big cats known for their striped coats and powerful builds.'
    },
    'zebra': {
        'scientific': 'Equus quagga',
        'habitat': 'African savannas and grasslands',
        'diet': 'Herbivore',
        'status': 'Near Threatened',
        'description': 'Known for black and white stripes and strong social structure.'
    },
    'camel': {
        'scientific': 'Camelus dromedarius',
        'habitat': 'Deserts and arid regions',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Adapted to survive in extreme heat and arid environments.'
    },
    'camel bird': {
        'scientific': 'Struthio camelus',
        'habitat': 'African savannas and deserts',
        'diet': 'Omnivore',
        'status': 'Least Concern',
        'description': 'Also known as ostrich, the largest living bird incapable of flight.'
    },
    'bear': {
        'scientific': 'Ursidae',
        'habitat': 'Forests, mountains, tundra',
        'diet': 'Omnivore',
        'status': 'Varies by species',
        'description': 'Large mammals with powerful builds and omnivorous diets.'
    },
    'panda': {
        'scientific': 'Ailuropoda melanoleuca',
        'habitat': 'Mountain forests in China',
        'diet': 'Herbivore (mainly bamboo)',
        'status': 'Vulnerable',
        'description': 'Distinctive black-and-white bears that primarily eat bamboo.'
    },
    'Rabbit': {
        'scientific': 'Oryctolagus cuniculus',
        'habitat': 'Grasslands, forests, and meadows',
        'diet': 'Herbivore',
        'status': 'Least Concern',
        'description': 'Small mammals known for long ears and rapid reproduction.'
    },
    'cat': {
        'scientific': 'Felis catus',
        'habitat': 'Domesticated, worldwide',
        'diet': 'Carnivore',
        'status': 'Domesticated',
        'description': 'Popular pets known for agility and independence.'
    },
    'cow': {
        'scientific': 'Bos taurus',
        'habitat': 'Pastures and farmlands',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Domesticated cattle raised for milk, meat, and labor.'
    },
    'crocodile': {
        'scientific': 'Crocodylidae',
        'habitat': 'Freshwater rivers, lakes, wetlands',
        'diet': 'Carnivore',
        'status': 'Varies by species',
        'description': 'Large aquatic reptiles known for powerful jaws and stealth.'
    },
    'deer': {
        'scientific': 'Cervidae',
        'habitat': 'Forests, grasslands, and wetlands',
        'diet': 'Herbivore',
        'status': 'Varies by species',
        'description': 'Graceful mammals with antlers found in many regions.'
    },
    'dog': {
        'scientific': 'Canis lupus familiaris',
        'habitat': 'Domesticated, worldwide',
        'diet': 'Omnivore',
        'status': 'Domesticated',
        'description': 'Loyal and intelligent companions bred for various purposes.'
    },
    'elephant': {
        'scientific': 'Loxodonta africana / Elephas maximus',
        'habitat': 'Forests, savannas',
        'diet': 'Herbivore',
        'status': 'Endangered',
        'description': 'Largest land animals with trunks used for communication and feeding.'
    },
    'goat': {
        'scientific': 'Capra aegagrus hircus',
        'habitat': 'Hills, farmlands',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Hardy domesticated animals known for milk, meat, and agility.'
    },
    'Hargila Bok': {
        'scientific': 'Leptoptilos dubius',
        'habitat': 'Wetlands and marshes of South Asia',
        'diet': 'Scavenger',
        'status': 'Endangered',
        'description': 'Also known as greater adjutant stork, a rare scavenging bird.'
    },
    'hippopotamus': {
        'scientific': 'Hippopotamus amphibius',
        'habitat': 'Rivers and lakes in Africa',
        'diet': 'Herbivore',
        'status': 'Vulnerable',
        'description': 'Large semi-aquatic mammals known for their size and aggressive nature.'
    },
    'horse': {
        'scientific': 'Equus ferus caballus',
        'habitat': 'Grasslands, domesticated regions',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Domesticated mammals used for transport, work, and sport.'
    },
    'kalo bok': {
        'scientific': 'Milvus migrans (likely local name for black kite)',
        'habitat': 'Open fields, forests, and urban areas',
        'diet': 'Carnivore (scavenger)',
        'status': 'Least Concern',
        'description': 'A bird of prey known for scavenging and widespread distribution.'
    },
    'Lama': {
        'scientific': 'Lama glama',
        'habitat': 'Andes Mountains, South America',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Domesticated camelid used as pack animals and for wool.'
    },
    'peacock': {
        'scientific': 'Pavo cristatus',
        'habitat': 'Forests and farmlands',
        'diet': 'Omnivore',
        'status': 'Least Concern',
        'description': 'Colorful birds known for extravagant tail feathers in males.'
    },
    'porcupine': {
        'scientific': 'Hystricidae',
        'habitat': 'Forests, deserts, and grasslands',
        'diet': 'Herbivore',
        'status': 'Least Concern',
        'description': 'Rodents covered in sharp quills used for defense.'
    },
    'ass': {
        'scientific': 'Equus africanus asinus',
        'habitat': 'Deserts and dry regions',
        'diet': 'Herbivore',
        'status': 'Domesticated',
        'description': 'Domesticated working animals used for transport and farming.'
    },
    'squirrel': {
        'scientific': 'Sciuridae',
        'habitat': 'Forests, urban areas, and parks',
        'diet': 'Omnivore',
        'status': 'Least Concern',
        'description': 'Small agile rodents with bushy tails, often found in trees.'
    },
    'monkey': {
        'scientific': 'Primates',
        'habitat': 'Forests and tropical regions',
        'diet': 'Omnivore',
        'status': 'Varies by species',
        'description': 'Intelligent, social primates with great agility and curiosity.'
    }
}

# Global variables to store the loaded models
models = {}

def load_models():
    """Load all three models with error handling"""
    global models
    try:
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                models[model_name] = load_model(model_path, compile=False)
                logger.info(f"Successfully loaded {model_name} model from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"All models loaded successfully. Total models: {len(models)}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise RuntimeError("Model loading failed") from e

# ---------------------------------------------------------------
# FIXED Helper Functions
# ---------------------------------------------------------------
def preprocess_image_for_model(img_file, target_size):
    """Preprocess uploaded image for a specific model - MATCHES KAGGLE CODE"""
    try:
        
        img_file.stream.seek(0)
        
        
        img = Image.open(img_file.stream)
        img = img.convert('RGB')  
        img = img.resize(target_size, Image.Resampling.LANCZOS)  
        
        
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError("Invalid image format") from e

def get_ensemble_predictions(img_file):
    """Get predictions from all models and create ensemble prediction - MATCHES KAGGLE LOGIC"""
    try:
        predictions = {}
        
        
        for model_name, model in models.items():
            target_size = INPUT_SIZES[model_name]
            img_array = preprocess_image_for_model(img_file, target_size)
            
           
            pred = model.predict(img_array, verbose=0)[0]
            predictions[model_name] = pred
            
            logger.info(f"{model_name} prediction shape: {pred.shape}, max prob: {np.max(pred):.4f}")
        
        
        ensemble_pred = (predictions['efficientnet'] + predictions['mobilenet'] + predictions['densenet']) / 3
        
        logger.info(f"Ensemble prediction - max prob: {np.max(ensemble_pred):.4f}, predicted class: {np.argmax(ensemble_pred)}")
        
        return predictions, ensemble_pred
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {str(e)}")
        raise

def get_top_predictions(predictions, model_confidences=None, top_k=5):
    """Get top k predictions with confidence scores"""
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    results = []
    
    for idx in top_indices:
        animal_name = ANIMAL_CLASSES[idx]
        result = {
            'animal': animal_name,
            'confidence': float(predictions[idx]),
            'details': ANIMAL_DETAILS.get(animal_name, {})
        }
        
        
        if model_confidences:
            result['model_confidences'] = {
                model: float(conf[idx]) for model, conf in model_confidences.items()
            }
        
        results.append(result)
    
    return results




# ---------------------------------------------------------------
# Application Routes
# ---------------------------------------------------------------
@app.route('/')
def home():
    """Serve the main application page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return jsonify({'error': 'Template not found. Make sure index.html is in frontend/templates/ folder'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    

    try:
        
        if not models:

            return jsonify({'error': 'Models not loaded'}), 500
        
        logger.info("Making predictions with ensemble models...")
        
        model_predictions, ensemble_pred = get_ensemble_predictions(file)
        
        
        top_predictions = get_top_predictions(
            ensemble_pred, 
            model_confidences=model_predictions,
            top_k=5
        )
        
        
        logger.info(f"Top prediction: {top_predictions[0]['animal']} with confidence {top_predictions[0]['confidence']:.4f}")
        logger.info(f"Top 3 predictions: {[(p['animal'], f'{p['confidence']:.3f}') for p in top_predictions[:3]]}")
        
        
        response = {
            'success': True,
            'top_prediction': top_predictions[0],
            'all_predictions': top_predictions,
            'model_info': {
                'models_used': list(models.keys()),
                'total_classes': len(ANIMAL_CLASSES),
                'ensemble_method': 'average'
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"Image processing error: {str(e)}")
        return jsonify({'error': 'Invalid image format'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500



@app.route('/animal-details/<animal_name>', methods=['GET'])
def get_animal_details(animal_name):
    """Get detailed information about a specific animal"""
    try:
        animal_name = animal_name.lower()
        if animal_name in ANIMAL_DETAILS:
            return jsonify({
                'success': True,
                'animal': animal_name,
                'details': ANIMAL_DETAILS[animal_name]
            })
        else:
            return jsonify({'error': 'Animal not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching animal details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/search-animals', methods=['GET'])
def search_animals():
    """Search for animals by name or characteristics"""
    try:
        query = request.args.get('query', '').lower().strip()
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        results = []
        for animal_name, details in ANIMAL_DETAILS.items():
            # Search Fields in animal name, scientific name, and description
            search_fields = [
                animal_name.lower(),
                details.get('scientific', '').lower(),
                details.get('description', '').lower(),
                details.get('habitat', '').lower(),
                details.get('diet', '').lower()
            ]
            
            if any(query in field for field in search_fields):
                results.append({
                    'animal': animal_name,
                    'details': details
                })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        models_loaded = len(models)
        model_status = {name: 'loaded' for name in models.keys()} if models else 'not loaded'
        
        return jsonify({
            'status': 'healthy' if models else 'unhealthy',
            'models_loaded': models_loaded,
            'model_status': model_status,
            'total_classes': len(ANIMAL_CLASSES),
            'class_order': ANIMAL_CLASSES  
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# ---------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.route('/animal-details')
def animal_details():
    """Serve the animal details encyclopedia page"""
    return render_template('animalDetails.html')

# ---------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------
if __name__ == '__main__':
    try:
        
        load_models()
        
        # Configure Flask settings
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
        
        logger.info("Starting Flask application...")
        logger.info(f"Models loaded: {list(models.keys())}")
        logger.info(f"Class order: {ANIMAL_CLASSES}")
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1)