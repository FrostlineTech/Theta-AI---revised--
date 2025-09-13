from flask import Flask, request, jsonify, render_template, url_for, session, Response, make_response
from flask_wtf.csrf import CSRFProtect, generate_csrf, validate_csrf
import sys
import os
import secrets
import re
import time
import logging
import html
import datetime
import json
import random
from pathlib import Path

# Import definition templates and other improvements
from src.interface.definitions import TECHNICAL_DEFINITIONS, IDENTITY_ANSWERS, HALLUCINATION_PRONE_TOPICS, SAFETY_RESPONSES

# Add project root to path to import modules
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from src.interface.theta_interface import ThetaInterface

class CustomFlask(Flask):
    def get_send_file_max_age(self, name):
        # Override to set longer cache timeout for static files
        return 31536000  # 1 year in seconds

app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'),
            static_url_path='/static')
            
# Security settings
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a random secret key
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching in development
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # CSRF token valid for 1 hour

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Set up logging
log_path = os.path.join(project_root, 'logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, f"theta_ui_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('theta_ui')

# Add console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)

# Request count for monitoring
request_counts = {}

# Define patterns to detect potential XSS or command injection
MALICIOUS_PATTERNS = [
    r'<script[^>]*>',
    r'javascript:',
    r'onerror=',
    r'onclick=',
    r'onload=',
    r'eval\(',
    r'\bexec\b',
    r'\bsystem\b',
    r'\bos\.\w+\b',
    r'\bsubprocess\.\w+\b',
]

# Compile patterns for efficiency
MALICIOUS_REGEX = re.compile('|'.join(MALICIOUS_PATTERNS), re.IGNORECASE)

# Initialize Theta interface
print("Initializing Theta AI...")

# Look for the best performing model (epoch 26) first
best_model_path = os.path.join(project_root, "models/theta_checkpoint_epoch_26")
final_model_path = os.path.join(project_root, "models/theta_final")

# Use best model if available, otherwise fallback to alternatives
if os.path.exists(best_model_path):
    model_path = best_model_path
    print(f"Using best performing model: {best_model_path} (Epoch 26)")
else:
    # Fallback to finding latest checkpoint or final model
    latest_checkpoint = None
    latest_epoch = 0
    
    # Check for checkpoint folders
    models_dir = os.path.join(project_root, "models")
    for item in os.listdir(models_dir):
        if item.startswith("theta_checkpoint_epoch_"):
            try:
                epoch = int(item.split("_")[-1])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = os.path.join(models_dir, item)
            except ValueError:
                continue
    
    # If a checkpoint was found, use it
    if latest_checkpoint:
        model_path = latest_checkpoint
        print(f"Using latest checkpoint model: {latest_checkpoint} (Epoch {latest_epoch})")
    else:
        # Otherwise use final model
        model_path = final_model_path
        print("Using final model")

# Initialize with the selected model
theta = ThetaInterface(model_path=model_path)
print("Theta AI initialized successfully!")

# Security headers middleware
@app.after_request
def add_security_headers(response):
    # Content Security Policy - Allow inline scripts for now
    response.headers['Content-Security-Policy'] = "default-src 'self'; "\
        "script-src 'self' 'unsafe-inline'; "\
        "style-src 'self' 'unsafe-inline'; "\
        "img-src 'self'; "\
        "font-src 'self'; "\
        "connect-src 'self'; "\
        "frame-src 'none'; "\
        "object-src 'none'; "\
        "base-uri 'self'"
    
    # Additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    
    return response

@app.route('/')
def home():
    # Log the request
    client_ip = request.remote_addr
    logger.info(f"Home page accessed from {client_ip}")
    
    # Track request counts for rate monitoring
    if client_ip not in request_counts:
        request_counts[client_ip] = {'count': 1, 'first_request': time.time()}
    else:
        request_counts[client_ip]['count'] += 1
        # Check for unusual activity (more than 60 requests per minute)
        elapsed = time.time() - request_counts[client_ip]['first_request']
        if elapsed < 60 and request_counts[client_ip]['count'] > 60:
            logger.warning(f"Possible abuse detected from {client_ip}: {request_counts[client_ip]['count']} requests in {elapsed:.1f} seconds")
    
    # Pass static URLs to template to avoid Jinja2 template issues with JavaScript
    static_url_css = url_for('static', filename='styles.css')
    static_url_logo = url_for('static', filename='theta-symbol.png')
    
    # Generate CSRF token for the form
    csrf_token = generate_csrf()
    
    return render_template('index.html', 
                           static_url_css=static_url_css,
                           static_url_logo=static_url_logo,
                           csrf_token=csrf_token)

def sanitize_input(text):
    """Sanitize user input to prevent XSS and command injection"""
    # Log if we find potential malicious patterns but still allow the text
    if MALICIOUS_REGEX.search(text):
        logger.warning(f"Potential suspicious pattern in input: {text[:100]}")
    
    # Escape HTML entities to prevent XSS in responses
    return html.escape(text)

@app.route('/ask', methods=['POST'])
@csrf.exempt  # Exempt the API from CSRF for now, we'll validate manually
def ask():
    try:
        # Get client IP for logging
        client_ip = request.remote_addr
        
        # Get CSRF token from request
        csrf_token = request.json.get('csrf_token')
        # For now, log but don't block if token is missing - helps with debugging
        if not csrf_token:
            logger.warning(f"Missing CSRF token from {client_ip}")
        # Only validate if a token was provided
        elif csrf_token:
            try:
                validate_csrf(csrf_token)
            except Exception as e:
                logger.warning(f"Invalid CSRF token from {client_ip}: {str(e)}")
                return jsonify({'error': 'Invalid CSRF token'}), 403
            
        # Get and validate question
        raw_question = request.json.get('question')
        if not raw_question:
            logger.info(f"Empty question from {client_ip}")
            return jsonify({'error': 'No question provided'}), 400
        
        # Sanitize input
        question = sanitize_input(raw_question)
        
        # Log the question (without PII)
        truncated_question = question[:100] + '...' if len(question) > 100 else question
        logger.info(f"Question from {client_ip}: {truncated_question}")
        
        # Clean and normalize the question
        question_lower = question.lower().strip()
        term = None  # Define term outside conditional blocks for later use
        
        # Check for identity-related questions
        identity_related = any(term in question_lower for term in [
            "who created", "who made", "who built", "founder", "ceo", "owner", 
            "who are you", "who is theta", "what are you", "tell me about yourself",
            "developed by", "made by", "built by", "creator", "developers"
        ])
        
        # Check for definition/explanation questions
        is_definition_question = any(pattern in question_lower for pattern in [
            "what is", "what's", "what are", "define", "explain", "tell me about", 
            "what does", "meaning of", "definition of", "stands for"
        ])
        
        # Check for exact identity questions in our templates
        for key, response in IDENTITY_ANSWERS.items():
            if question_lower == key or question_lower.startswith(key + "?"):
                logger.info(f"Using identity template for {client_ip}")
                answer = response
                # Log the response length and return
                logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                return jsonify({'answer': answer})
        
        # Check for technical definitions
        is_definition_question = any(pattern in question_lower for pattern in [
            "what is", "what's", "what are", "define", "explain", "tell me about", 
            "what does", "meaning of", "definition of", "stands for"
        ])
        
        # Extract the term being defined
        if is_definition_question:
            # Try to extract the term after common patterns
            for pattern in ["what is", "what's", "what does", "define", "explain", "tell me about"]:
                if pattern in question_lower:
                    # Get everything after the pattern
                    term_part = question_lower.split(pattern, 1)[1].strip().rstrip('?')
                    # If it contains "mean" or "stand for", handle specially
                    if "mean" in term_part:
                        term = term_part.split("mean")[0].strip()
                    elif "stand for" in term_part:
                        term = term_part.split("stand for")[0].strip()
                    else:
                        term = term_part
                    break
            
            # If we extracted a term, check if we have a definition
            if term:
                # Clean up term (remove articles like "a", "an", "the")
                term = re.sub(r'^(a|an|the)\s+', '', term).strip()
                # Check in our technical definitions
                if term in TECHNICAL_DEFINITIONS:
                    logger.info(f"Using definition template for '{term}' from {client_ip}")
                    answer = TECHNICAL_DEFINITIONS[term]
                    # Log the response length and return
                    logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                    return jsonify({'answer': answer})
        
        # Check for hallucination-prone topics
        if any(topic in question_lower for topic in HALLUCINATION_PRONE_TOPICS):
            # For these topics, sometimes directly return a safety response
            if random.random() < 0.7:  # 70% chance to use safety response
                safety_keys = list(SAFETY_RESPONSES.keys())
                logger.info(f"Using safety response for hallucination-prone topic from {client_ip}")
                answer = SAFETY_RESPONSES[random.choice(safety_keys)]
                # Log the response length and return
                logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
                return jsonify({'answer': answer})
        
        # If we reached here, no templates matched - generate answer using the model
        answer = theta.answer_question(question)
        
        # Post-process validation
        hallucination_markers = [
            "[Source]", "[source]", "https://", "www.", "(link)", "[link]",
            "[citation", "Source:", "+", "++", "====", "*.*", ".com", ".org",
            "https://www", "$", "@", "<![CDATA[", "]]", "//", "<!DOCTYPE", "<html", "<script"
        ]
        
        if any(marker in answer for marker in hallucination_markers):
            logger.warning(f"Detected potential hallucination markers in response to {client_ip}")
            # Fall back to a safer response
            if is_definition_question and term:
                answer = f"I don't have a specific definition for '{term}' in my knowledge base. This term might be specialized or outside my training data."
            else:
                answer = f"I don't have specific information about '{question.strip()}' in my knowledge base. I'm trained on cybersecurity, software development, and IT concepts by Frostline Solutions."
        
        # Check if the response is unusually long for a simple question
        if len(question.split()) < 5 and len(answer.split()) > 100:
            logger.warning(f"Unusually long answer to short question from {client_ip}")
            answer = "I'd like to provide a concise answer to your question, but I'm not confident I have the specific information you're looking for. Could you provide more details or ask in a different way?"
        
        # Log the response length
        logger.info(f"Generated answer of {len(answer)} characters for {client_ip}")
        
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500  # Don't expose specific error details

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    print("Starting Theta AI Web Interface...")
    print("Access the interface at http://localhost:5000")
    # Set to False for production
    app.run(host='0.0.0.0', port=5000, debug=False)  # Accessible on local network
