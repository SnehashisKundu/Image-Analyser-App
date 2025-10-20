import os
import json
from PIL import Image
from datetime import datetime
import base64

import numpy as np
import tensorflow as tf
import streamlit as st
from dotenv import load_dotenv

# Import Gemini functionality from separate module
from gemini import get_gemini_solution, test_gemini_connection

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="🌿 Plant Disease Classifier", page_icon="🌱", layout="wide")
st.markdown("""
<style>
button[kind="primary"] {
    background-color: #34c759 !important;
    color: white !important;
}
/* Custom style for the Gemini advice box */
.gemini-advice-box {
    background-color: #f0fdf4; /* Light green background */
    border-left: 5px solid #2e7d32; /* Darker green border */
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* Subtle shadow */
}
.gemini-advice-box h4 {
    color: #1b5e20; /* Even darker green for heading */
    margin-bottom: 15px;
    font-size: 1.4em;
    display: flex;
    align-items: center;
}
.gemini-advice-box h4 svg { /* Style for potential SVG icons if Gemini generates them */
    margin-right: 10px;
}
.gemini-advice-box p {
    color: #333;
    font-size: 1.05em;
    line-height: 1.7;
}
.gemini-advice-box ul {
    list-style-type: none; /* Remove default bullet points */
    padding-left: 0;
}
.gemini-advice-box li {
    margin-bottom: 10px;
    padding-left: 25px; /* Indent list items */
    position: relative;
    color: #444;
}
.gemini-advice-box li::before {
    content: '•'; /* Custom bullet point */
    color: #34c759; /* Green bullet point */
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -1em;
    position: absolute;
    left: 0;
}
</style>
""", unsafe_allow_html=True)

# 🔐 Gemini AI is now configured in gemini.py module

# ----------------------------
# 🧠 Model & Metadata Loading
# ----------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "plant_disease_prediction_model (1).h5"
model_path = os.path.join(working_dir, model_filename)

model = None
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.warning(f"Could not load model at {model_path}: {e}")

class_indices = {}
class_indices_path = os.path.join(working_dir, "class_indices.json")
if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
    except Exception as e:
        st.warning(f"Failed to load class_indices.json: {e}")
else:
    st.warning(f"class_indices.json not found at {class_indices_path}")

# ----------------------------
# 🪴 Helper Functions
# ----------------------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    if isinstance(image_path, Image.Image):
        img = image_path
    else:
        img = Image.open(image_path)

    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def pil_image_to_base64(pil_img, fmt='PNG'):
    """Convert a PIL image to a base64 string (no header)."""
    from io import BytesIO
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    byte_im = buf.getvalue()
    import base64
    return base64.b64encode(byte_im).decode('utf-8')

def _analyze_pixel(r, g, b):
    import colorsys
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    is_green = (0.25 <= h <= 0.45) and s > 0.25 and v > 0.15
    is_yellow_plant = (0.12 <= h <= 0.25) and s > 0.3 and v > 0.3
    is_brown_plant = (0.05 <= h <= 0.12) and s > 0.2 and (0.2 <= v <= 0.8)
    is_skin = (0.02 <= h <= 0.1) and (0.15 <= s <= 0.8) and (0.3 <= v <= 0.95)
    is_document = s < 0.05 and v > 0.9
    is_text = s < 0.1 and (v < 0.2 or v > 0.8)
    return is_green or is_yellow_plant or is_brown_plant, is_skin, is_document, is_text

def is_likely_plant_image(image):
    """Check if image likely contains plant material using improved color analysis"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    plant_pixels = 0
    skin_pixels = 0
    document_pixels = 0
    text_pixels = 0

    for y in range(0, height, 6):
        for x in range(0, width, 6):
            r, g, b = img_array[y, x][:3]
            is_plant, is_skin, is_document, is_text = _analyze_pixel(r, g, b)
            if is_plant:
                plant_pixels += 1
            if is_skin:
                skin_pixels += 1
            if is_document:
                document_pixels += 1
            elif is_text:
                text_pixels += 1

    sampled_pixels = (height // 6) * (width // 6)
    plant_ratio = plant_pixels / max(sampled_pixels, 1)
    skin_ratio = skin_pixels / max(sampled_pixels, 1)
    text_ratio = text_pixels / max(sampled_pixels, 1)
    document_ratio = document_pixels / max(sampled_pixels, 1)

    if document_ratio > 0.3 or text_ratio > 0.4 or skin_ratio > 0.15:
        return False
    if plant_ratio > 0.12:
        return True
    return False

def predict_image_topk(model, image_input, class_indices, k=3):
    preprocessed = load_and_preprocess_image(image_input)
    preds = model.predict(preprocessed)
    probs = preds[0]

    if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
        probs = tf.nn.softmax(probs).numpy()

    try:
        idxs = sorted(class_indices.keys(), key=lambda x: int(x))
        idx_to_name = [class_indices[i] for i in idxs]
    except Exception:
        idx_to_name = list(class_indices.values())

    topk_idx = np.argsort(probs)[::-1][:k]
    topk = [(idx_to_name[i] if i < len(idx_to_name) else str(i), float(probs[i])) for i in topk_idx]
    return topk

def save_corrected_sample(pil_image, label_name, dest_root=None):
    if dest_root is None:
        dest_root = os.path.join(working_dir, 'collected')
    safe_label = label_name.replace('/', '_').strip()
    dest_dir = os.path.join(dest_root, safe_label)
    os.makedirs(dest_dir, exist_ok=True)
    import time, uuid
    fn = f"img_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    dest_path = os.path.join(dest_dir, fn)
    pil_image.save(dest_path)
    return dest_path

# ----------------------------
# 🌾 Gemini Integration is now handled in gemini.py module
# ----------------------------

# ----------------------------
# 🌿 Streamlit UI
# ----------------------------
# Initialize session states at the top of the app
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'gemini_advice' not in st.session_state:
    st.session_state['gemini_advice'] = None
if 'advice_for_prediction' not in st.session_state:
    st.session_state['advice_for_prediction'] = None
if 'advice_history' not in st.session_state:
    st.session_state['advice_history'] = []

st.markdown("""
    <h1 style='text-align: center; color: #34c759;'>
        🌿 Plant Disease Classifier
    </h1>
    <p style='text-align: center; color: gray;'>
        Upload a plant leaf and let AI diagnose it for you 🍃
    </p>
    <hr>
""", unsafe_allow_html=True)

# Professional sidebar - only show content when there are valid predictions
with st.sidebar:
    st.header("🕒 Recent Predictions")
    
    if len(st.session_state['history']) == 0:
        st.info("No plant analyses yet.\n\nUpload a plant image to get started! 🌱")
    else:
        clear_clicked = st.button("🗑️ Clear All History", key="sidebar_clear_all_btn")
        if clear_clicked:
            st.session_state['history'].clear()
            st.success("All history cleared!")
            st.rerun()

        for i, item in enumerate(st.session_state['history'][-5:][::-1]):
            col_left, col_right = st.columns([4, 1])
            with col_left:
                st.write(f"🌿 {item}")
            with col_right:
                if st.button("❌", key=f"sidebar_del_{i}", help="Remove this entry"):
                    idx_to_remove = len(st.session_state['history']) - 1 - i
                    st.session_state['history'].pop(idx_to_remove)
                    st.rerun()

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
    except Exception:
        st.error("Uploaded file couldn't be opened as an image.")
        image = None
    # show columns for preview + controls regardless of whether image opened
    col1, col2 = st.columns([1, 2])
    with col1:
        if image is not None:
            try:
                b64 = pil_image_to_base64(image.resize((960, 540)), fmt='JPEG')
                preview_html = f"""
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="width: 100%; max-width: 760px; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.45); margin-bottom: 8px;">
                        <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:auto; display:block;" />
                    </div>
                    <div style="color: #9ca3af; font-size: 13px; margin-top: 4px; text-align:center;">Uploaded Plant Image</div>
                </div>
                """
                st.markdown(preview_html, unsafe_allow_html=True)
            except Exception:
                # fallback small thumbnail
                st.image(image.resize((180, 180)))

    with col2:
        if model is None:
            st.error("Model not loaded. Please ensure the .h5 file is present.")
        elif not class_indices:
            st.error("class_indices.json is missing or empty.")
        else:
            if st.button('Classify', key="classify_btn"):
                with st.spinner('Analyzing the leaf... 🌿'):
                    try:
                        topk = predict_image_topk(model, uploaded_image, class_indices, k=3)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        topk = []

                if topk:
                    predicted_label = topk[0][0]
                    confidence = topk[0][1]
                    
                    # Check if this actually looks like a plant using image analysis
                    if not is_likely_plant_image(image):
                        # Clear previous predictions when plant detection fails
                        if 'last_prediction' in st.session_state:
                            del st.session_state['last_prediction']
                        if 'gemini_advice' in st.session_state:
                            st.session_state['gemini_advice'] = None
                        if 'advice_for_prediction' in st.session_state:
                            st.session_state['advice_for_prediction'] = None
                        
                        # Detailed analysis to determine what type of non-plant image this is
                        img_array = np.array(image)
                        import colorsys
                        skin_pixels = 0
                        document_pixels = 0
                        total_checked = 0
                        
                        for y in range(0, img_array.shape[0], 12):
                            for x in range(0, img_array.shape[1], 12):
                                r, g, b = img_array[y, x][:3] / 255.0
                                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                                
                                # Skin tone detection (broader for ID cards)
                                if (0.02 <= h <= 0.1) and (0.15 <= s <= 0.8) and (0.3 <= v <= 0.95):
                                    skin_pixels += 1
                                
                                # Document detection (white backgrounds, low saturation)
                                if s < 0.05 and v > 0.9:
                                    document_pixels += 1
                                    
                                total_checked += 1
                        
                        skin_ratio = skin_pixels / max(total_checked, 1)
                        document_ratio = document_pixels / max(total_checked, 1)
                        is_human_photo = skin_ratio > 0.12
                        is_document = document_ratio > 0.25
                        
                        st.error("🤔 This doesn't look like a plant leaf! Please upload a clear image of a plant leaf for disease analysis.")
                        
                        if is_human_photo:
                            st.markdown("""
                            <div style="background: linear-gradient(90deg, #ff6b6b, #feca57); padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4 style="color: white; margin: 0;">😄 Human Detected!</h4>
                            <p style="color: white; margin: 5px 0 0 0;">Nice try! I can see this is a photo of a person, not a plant! 
                            While you look great, I'm specifically trained for plant diseases, not human health checkups! 🌿👨‍⚕️</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif is_document:
                            st.markdown("""
                            <div style="background: linear-gradient(90deg, #ff6b6b, #feca57); padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4 style="color: white; margin: 0;">📄 Document Detected!</h4>
                            <p style="color: white; margin: 5px 0 0 0;">I can see this is an ID card, document, or text-based image! 
                            While I'm pretty smart, I can only diagnose plant diseases, not analyze documents! 🌿📋</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(90deg, #ff6b6b, #feca57); padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4 style="color: white; margin: 0;">😄 Funny Detection Alert!</h4>
                            <p style="color: white; margin: 5px 0 0 0;">I see you uploaded something that's not a plant. 
                            While I'm flattered you think I can diagnose everything, I'm specifically trained for plant diseases! 🌿</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.info("💡 **Please upload**: A clear photo of a plant leaf with visible disease symptoms")
                        st.markdown("😂")
                    else:
                        st.session_state['last_prediction'] = predicted_label

                        if st.button("💾 Save this sample for retraining", key="save_sample_btn"):
                            path = save_corrected_sample(image, predicted_label)
                            st.info(f"Saved under: `{path}`")

                        st.session_state['history'].append(predicted_label)



# Professional Plant Care Analysis Dashboard
if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
    
    current_prediction = st.session_state['last_prediction']
    disease_name = current_prediction.replace('_', ' ').replace('(', '').replace(')', '')
    
    # Professional Dashboard Container
    st.markdown("""<div style="margin: 30px 0;"></div>""", unsafe_allow_html=True)
    
    # Disease Analysis Card
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; 
                    padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: #10b981; color: white; width: 50px; height: 50px; 
                            border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                            margin-right: 16px; font-size: 24px;">🔬</div>
                <div>
                    <h2 style="color: #1f2937; margin: 0; font-size: 22px; font-weight: 700;">Disease Analysis Result</h2>
                    <p style="color: #6b7280; margin: 4px 0 0 0; font-size: 14px;">AI-Powered Plant Health Assessment</p>
                </div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; 
                        padding: 20px; border-left: 4px solid #10b981;">
                <h3 style="color: #1f2937; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">{disease_name}</h3>
                <p style="color: #6b7280; margin: 0; font-size: 14px;">Confidence Level: High • Analysis Complete</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get AI Analysis Button
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; 
                    padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); text-align: center;">
            <div style="background: #3b82f6; color: white; width: 60px; height: 60px; 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                        margin: 0 auto 16px auto; font-size: 28px;">🤖</div>
            <h3 style="color: #1f2937; margin: 0 0 8px 0; font-size: 16px; font-weight: 600;">AI Analysis</h3>
            <p style="color: #6b7280; margin: 0 0 20px 0; font-size: 12px;">Get detailed treatment recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-generate treatment plan if not already generated for this prediction
        if (not st.session_state['gemini_advice'] or 
            st.session_state['advice_for_prediction'] != current_prediction):
            
            with st.spinner("🔄 Generating treatment plan..."):
                try:
                    advice = get_gemini_solution(current_prediction)
                    
                    # More robust error checking
                    if advice and len(advice.strip()) > 10 and not advice.startswith("❌"):
                        st.session_state['gemini_advice'] = advice
                        st.session_state['advice_for_prediction'] = current_prediction
                        
                        # Add to history
                        history_item = {
                            'prediction': current_prediction,
                            'advice': advice,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        if history_item not in st.session_state['advice_history']:
                            st.session_state['advice_history'].insert(0, history_item)
                            st.session_state['advice_history'] = st.session_state['advice_history'][:5]
                        
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Optional: Add manual refresh button for new analysis
        if st.button("🔄 Generate New Analysis", key="refresh_analysis_button", 
                    help="Generate fresh treatment recommendations", use_container_width=True):
            with st.spinner("🔄 Generating new analysis..."):
                try:
                    advice = get_gemini_solution(current_prediction)
                    
                    if advice and len(advice.strip()) > 10 and not advice.startswith("❌"):
                        st.session_state['gemini_advice'] = advice
                        st.session_state['advice_for_prediction'] = current_prediction
                        
                        # Add to history
                        history_item = {
                            'prediction': current_prediction,
                            'advice': advice,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state['advice_history'].insert(0, history_item)
                        st.session_state['advice_history'] = st.session_state['advice_history'][:5]
                        
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Professional Analysis Results
    if (st.session_state['gemini_advice'] and 
        st.session_state['advice_for_prediction'] == current_prediction):
        
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; 
                    padding: 30px; margin: 20px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; margin-bottom: 25px;">
                <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; 
                            width: 55px; height: 55px; border-radius: 12px; 
                            display: flex; align-items: center; justify-content: center; 
                            margin-right: 16px; font-size: 26px; box-shadow: 0 4px 12px rgba(16,185,129,0.3);">📋</div>
                <div>
                    <h2 style="color: #1f2937; margin: 0; font-size: 24px; font-weight: 700;">Treatment Analysis Report</h2>
                    <p style="color: #6b7280; margin: 4px 0 0 0; font-size: 14px;">Comprehensive Care Recommendations</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Parse and format advice with progress bars and metrics
        advice_lines = st.session_state['gemini_advice'].split('\n')
        formatted_advice = ""
        
        for line in advice_lines:
            if line.strip():
                # Add professional formatting
                if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                    formatted_advice += f'<div style="margin: 15px 0; padding: 15px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">'
                    formatted_advice += f'<p style="color: #1f2937; margin: 0; font-weight: 600; line-height: 1.6;">{line}</p>'
                    formatted_advice += '</div>'
                elif '**' in line:
                    # Bold headers
                    formatted_line = line.replace('**', '<strong>').replace('**', '</strong>')
                    formatted_advice += f'<p style="color: #374151; margin: 8px 0; font-weight: 500; line-height: 1.6;">{formatted_line}</p>'
                else:
                    formatted_advice += f'<p style="color: #4b5563; margin: 8px 0; line-height: 1.6;">{line}</p>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0fdf4, #ecfdf5); 
                    padding: 25px; border-radius: 10px; border: 1px solid #d1fae5;">
            {formatted_advice}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col_action1, col_action2, col_action3 = st.columns([1, 1, 1])
        with col_action2:
            if st.button("🔄 Generate New Report", key="professional_clear_btn", use_container_width=True):
                st.session_state['gemini_advice'] = None
                st.session_state['advice_for_prediction'] = None
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis History (if exists)
    if st.session_state['advice_history']:
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; 
                    padding: 25px; margin: 20px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: #6366f1; color: white; width: 45px; height: 45px; 
                            border-radius: 10px; display: flex; align-items: center; justify-content: center; 
                            margin-right: 12px; font-size: 20px;">📚</div>
                <h3 style="color: #1f2937; margin: 0; font-size: 18px; font-weight: 600;">Analysis History</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for i, item in enumerate(st.session_state['advice_history'][:3]):
            short_name = item['prediction'].split('___')[-1].replace('_', ' ')
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; 
                        border-radius: 8px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <div style="flex: 1;">
                        <p style="color: #1f2937; margin: 0; font-weight: 600; font-size: 14px;">{short_name}</p>
                        <p style="color: #6b7280; margin: 2px 0 0 0; font-size: 12px;">{item['timestamp']}</p>
                    </div>
                    <div style="color: #10b981; font-weight: 600; font-size: 12px;">✓ Complete</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)