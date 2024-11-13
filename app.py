import torch
import cv2
import numpy as np
import sqlite3
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import streamlit as st
from PIL import Image
import io
import uuid
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel

# Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Object Detection with Faster R-CNN", page_icon=":guardsman:", layout="wide")

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    st.sidebar.text("Using GPU")
else:
    st.sidebar.text("Using CPU acceleration")

# Load the Faster R-CNN model with ResNet50 backbone
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model = model.to(device)
model.eval()

# Load CLIP and GPT-2 models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize SQLite database
conn = sqlite3.connect('detections.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    label TEXT,
    score REAL,
    box TEXT,
    image BLOB
)''')
conn.commit()

# Generate unique user session ID
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())
user_id = st.session_state['user_id']

# Define function to automatically export full database to CSV
def export_database():
    c.execute("SELECT * FROM detections")
    rows = c.fetchall()
    df = pd.DataFrame(rows, columns=['ID', 'User_ID', 'Label', 'Score', 'Box', 'Image', 'Description'])
    csv_path = 'detection_data_all_users.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

# Define object detection function
def detect_objects(image_np, threshold=0.5):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]["boxes"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            results.append((label, score, (x1, y1, x2, y2)))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"Label: {label}, Score: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image_np, results

# Save results to database if not duplicate
def save_results_to_db(results, image_np):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    
    for label, score, box in results:
        try:
            # Check for existing entry
            c.execute("SELECT id FROM detections WHERE user_id = ? AND label = ? AND score = ? AND box = ?", 
                      (user_id, str(label), float(score), str(box)))
            exists = c.fetchone()
            
            # Insert only if entry does not exist
            if not exists:
                _, img_encoded = cv2.imencode('.jpg', image_np)
                image_binary = img_encoded.tobytes()
                c.execute("INSERT INTO detections (user_id, label, score, box, image, description) VALUES (?, ?, ?, ?, ?, ?)", 
                          (user_id, str(label), float(score), str(box), image_binary))
        except sqlite3.Error as e:
            st.error(f"An error occurred: {e}")
    
    conn.commit()
    conn.close()
    export_database()

# Define the caption generation function
def generate_enhanced_caption(image):
    # Process image and extract CLIP features
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Initial prompt for GPT-2 based on a general descriptive request
    initial_prompt = "This image shows a scene with detailed description: "

    # Encode the initial prompt for GPT-2
    input_ids = gpt_tokenizer.encode(initial_prompt, return_tensors="pt")
    initial_caption_ids = gpt_model.generate(
        input_ids,
        max_length=70,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )
    initial_caption = gpt_tokenizer.decode(initial_caption_ids[0], skip_special_tokens=True)

    # Refine the caption by feeding it back into GPT-2
    input_ids = gpt_tokenizer.encode(initial_caption, return_tensors="pt")
    refined_caption_ids = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)
    refined_caption = gpt_tokenizer.decode(refined_caption_ids[0], skip_special_tokens=True)

    return refined_caption

# Streamlit UI
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4B9CD3;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">Object Detection with Faster R-CNN Using CUDA</p>', unsafe_allow_html=True)

st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run object detection
    output_image, detection_results = detect_objects(image_np, threshold=confidence_threshold)
    save_results_to_db(detection_results, output_image)

    # Generate the enhanced caption using CLIP and GPT-2
    refined_caption = generate_enhanced_caption(image)
    
    # Display results in Streamlit
    st.markdown(f"<h3 style='color:#28a745;'>Generated Caption:</h3>", unsafe_allow_html=True)
    st.write(f"**{refined_caption}**")

    # Display images with styling
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Processed Image with Bounding Boxes", use_column_width=True)

# Display past detections only for the current user
st.sidebar.subheader("Your Past Detections")
c.execute("SELECT label, score, box, image FROM detections WHERE user_id = ?", (user_id,))
rows = c.fetchall()

for label, score, box, image_data in rows:
    st.sidebar.write(f"**Label:** {label}, **Score:** {score:.2f}, **Box:** {box}")
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    st.sidebar.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Detection Image", use_column_width=True)
