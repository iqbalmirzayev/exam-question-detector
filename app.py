# app.py faylının içindəkilər (Statistika bölməsi əlavə edilmiş versiya)

import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter # <--- STATİSTİKA ÜÇÜN YENİ İMPORT

# --- FUNKSİYALAR ---

def load_roboflow_model():
    """Gizli məlumatları istifadə edərək Roboflow modelini yükləyir."""
    api_key = st.secrets["ROBOFLOW_API_KEY"]
    workspace = st.secrets["ROBOFLOW_WORKSPACE"]
    project = st.secrets["ROBOFLOW_MODEL"]
    version = st.secrets["ROBOFLOW_VERSION"]
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    model = project.version(version).model
    return model

def draw_predictions(image, predictions):
    """
    Şəklin üzərinə təsbit edilən obyektlərin çərçivələrini və etiketlərini çəkir.
    (Azərbaycan hərflərini dəstəkləyən Pillow kitabxanası ilə yenilənib)
    """
    output_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(output_image_rgb)
    draw = ImageDraw.Draw(pil_image)

    try:
        font_path = "fonts/Roboto-Medium.ttf"
        font_size = 15
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        st.error(f"Şrift faylı tapılmadı: {font_path}. 'fonts' qovluğunu və içindəki .ttf faylını yoxlayın.")
        font = ImageFont.load_default()

    label_map = {
        "questions": {"name": "qapalı sual", "color": (0, 255, 0)},
        "questions_o": {"name": "açıq sual", "color": (0, 165, 255)},
    }
    default_style = {"name": "digər", "color": (255, 0, 0)}

    for pred in predictions['predictions']:
        x, y, width, height = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        confidence = pred['confidence']
        raw_label = pred['class']

        style = label_map.get(raw_label, default_style)
        display_name = style["name"]
        color = style["color"]

        x1, y1 = x - width // 2, y - height // 2
        x2, y2 = x + width // 2, y + height // 2
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text = f"{display_name} ({confidence:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_height - 3), text, font=font, fill=(0, 0, 0))

    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return final_image

# --- STREAMLIT TƏTBİQİNİN ƏSAS HİSSƏSİ ---

st.set_page_config(page_title="Roboflow Obyekt Təsbiti", page_icon="🤖", layout="wide")
st.title("📄 Exam Question Detector")
st.write("Roboflow modelinizi test etmək üçün bir şəkil yükləyin.")

try:
    model = load_roboflow_model()
except Exception as e:
    st.error(f"Model yüklənərkən xəta baş verdi: {e}")
    st.stop()

with st.sidebar:
    st.header("Tənzimləmələr")
    uploaded_file = st.file_uploader("Bir şəkil seçin...", type=["jpg", "jpeg", "png"])
    confidence_threshold = st.slider("Güvən Eşiyi", 0.0, 1.0, 0.5, 0.05)
    submit_button = st.button("Obyektləri Təsbit Et")

if uploaded_file is not None and submit_button:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_to_predict = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal Şəkil")
        st.image(image_to_predict, channels="BGR", use_container_width=True)

    with st.spinner('Obyektlər təsbit edilir...'):
        predictions = model.predict(image_to_predict, confidence=confidence_threshold).json()
        output_image = draw_predictions(image_to_predict, predictions)
        with col2:
            st.subheader("Təsbit Edilmiş Obyektlər")
            st.image(output_image, channels="BGR", use_container_width=True)
            
    # --- YENİ HİSSƏ: STATİSTİKA BÖLMƏSİ ---
    st.subheader("Statistika")
    
    total_objects = len(predictions['predictions'])

    if total_objects > 0:
        # Etiketləri saymaq üçün Counter istifadə edirik
        all_labels = [p['class'] for p in predictions['predictions']]
        stats = Counter(all_labels)
        
        # Hər tipdən neçə dənə olduğunu alırıq
        qapali_count = stats.get("questions", 0)
        aciq_count = stats.get("questions_o", 0)
        
        # Mətn hissələrini hazırlayırıq
        summary_text = f"**Cəmi {total_objects} obyekt tapıldı:**"
        details_list = []
        if qapali_count > 0:
            details_list.append(f"{qapali_count} qapalı sual")
        if aciq_count > 0:
            details_list.append(f"{aciq_count} açıq sual")
        
        # Mətnləri vergül ilə birləşdiririk
        summary_details = ", ".join(details_list)
        
        # Nəticəni markdown formatında yazdırırıq
        st.markdown(f"{summary_text} {summary_details}.")
    else:
        st.write("Təyin edilən güvən eşiyi ilə heç bir obyekt tapılmadı.")
    
elif uploaded_file is None:
    st.info("Zəhmət olmasa, başlamaq üçün soldakı paneldən bir şəkil yükləyin.")