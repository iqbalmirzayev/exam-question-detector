
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import requests
import io

LABEL_MAP = {
    "questions": {"name": "qapalı sual", "color": (0, 255, 0)},
    "questions_o": {"name": "açıq sual", "color": (0, 165, 255)},
}
DEFAULT_STYLE = {"name": "digər", "color": (255, 0, 0)}
BACKEND_API_URL = "https://b350b8a2287d.ngrok-free.app/detect"

@st.cache_resource
def load_roboflow_model():
    st.write("Roboflow modeli yüklənir...")
    try:
        from roboflow import Roboflow
        api_key = st.secrets["ROBOFLOW_API_KEY"]
        workspace = st.secrets["ROBOFLOW_WORKSPACE"]
        project = st.secrets["ROBOFLOW_MODEL"]
        version = st.secrets["ROBOFLOW_VERSION"]
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        model = project_obj.version(version).model
        st.write("Roboflow modeli yükləndi.")
        return model
    except ImportError:
        st.error("Roboflow kitabxanası tapılmadı. Zəhmət olmasa quraşdırın: pip install roboflow")
        return None
    except KeyError:
        st.error("Roboflow API açarları Streamlit secrets-də tapılmadı. Zəhmət olmasa yoxlayın.")
        return None
    except Exception as e:
        st.error(f"Roboflow modeli yüklənərkən xəta baş verdi: {e}")
        return None

def draw_roboflow_predictions(original_image, predictions):
    output_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(output_image_rgb)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("fonts/Roboto-Medium.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    original_height, original_width, _ = original_image.shape
    x_scale = original_width / 640.0
    y_scale = original_height / 640.0
    for pred in predictions.get('predictions', []):
        center_x, center_y = pred['x'] * x_scale, pred['y'] * y_scale
        width, height = pred['width'] * x_scale, pred['height'] * y_scale
        confidence, raw_label = pred['confidence'], pred['class']
        style = LABEL_MAP.get(raw_label, DEFAULT_STYLE)
        display_name, color = style["name"], style["color"]
        x1, y1, x2, y2 = center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{display_name} ({confidence:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 5, y1], fill=color)
        draw.text((x1 + 5, y1 - text_height - 7), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_backend_predictions(original_image, detections):
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("fonts/Roboto-Medium.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det['box_coordinates']
        raw_label = det['class_name']
        confidence = det['confidence_score']
        style = LABEL_MAP.get(raw_label, DEFAULT_STYLE)
        display_name, color = style["name"], style["color"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{display_name} ({confidence:.2f})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_bg_y1 = y1 - text_height - 10
        if text_bg_y1 < 0:
            text_bg_y1 = y1 + 5
        draw.rectangle([x1, text_bg_y1, x1 + text_width + 10, text_bg_y1 + text_height + 10], fill=color)
        draw.text((x1 + 5, text_bg_y1 + 5), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

st.set_page_config(page_title="Model Test Platforması", page_icon="🤖", layout="wide")
st.title("📄 Exam Question Detector")
st.markdown("""
### Hansı Modeli Seçməli?

- **Roboflow (Server) ☁️:** Bu, Roboflow platformasında düzəltdiyimiz modeldir. Sürətlidir və yüksək doğruluq faizi göstərir. 

- **Lokal (API-Backend) 💻:**  Bu, Google Colab-da təlim etdirdiyimiz modelimizdir. Bu seçim edildikdə, bütün analiz kənar bir xidmətə müraciət etmədən, birbaşa bizim serverimizdə çalışır. Həmçinin siz lokalınızda çalışdıra bilərsiniz. Digər modeldən müəyyən qədər gec nəticə göstərə bilər. 
---

### Ən Yaxşı Nəticəni Almaq Üçün Bələdçi 💡
Yüklədiyiniz şəklin keyfiyyəti, modelin nə qədər dəqiq işləməsinə birbaşa təsir edir. Ən yaxşı nəticə üçün bu sadə qaydalara əməl etməyə çalışın:

- **📸 Düz Bucaq Altında Çəkin:** Şəkli, sanki bir skanerdən çıxmış kimi, tam yuxarıdan çəkin. Yandan və ya bucaq altında çəkilmiş şəkillər obyektlərin formasını pozur və modelin onları tanımasını çətinləşdirir.
- **⬆️ Düzgün İstiqamətdə Yükləyin:** Şəklin dik (portret) vəziyyətdə olduğundan və yazının düz oxunduğundan əmin olun. Başıaşağı və ya yana çevrilmiş şəkillər modeli çaşdıracaq.
- **☀️ Aydın İşıqda Çəkin:** Kölgəsiz və aydın bir mühitdə çəkilmiş şəkillər ən yaxşı nəticəni verir. Mətnin rəngi ilə fonun rəngi nə qədər aydın seçilərsə, model o qədər yaxşı işləyər.
- **⚙️ Ölçü Məsələsini Bizə Həvalə Edin:** Modelin ən kritik tələbi, analizin **mütləq 640x640 piksel** ölçülü şəkil üzərində aparılmasıdır. Narahat olmayın, bu hissəni biz avtomatik edirik! Siz sadəcə mümkün olan ən keyfiyyətli şəkli yükləyin, proqramımız onu model üçün ideal ölçüyə özü gətirəcək.
""")
st.divider()

with st.sidebar:
    st.header("Tənzimləmələr")
    model_choice = st.radio(
        "Hansı modeli istifadə edirsiniz?",
        ('Roboflow (Server)', 'Lokal (API-Backend)'),
        horizontal=True
    )
    st.divider()
    
    if model_choice == 'Roboflow (Server)':
        default_confidence = 0.50
    else:
        default_confidence = 0.65

    confidence_threshold = st.slider(
        "Confidence Level", 0.0, 1.0, default_confidence, 0.05
    )

    uploaded_file = st.file_uploader("Bir şəkil seçin...", type=["jpg", "jpeg", "png"])
    submit_button = st.button("Obyektləri Təsbit Et")

#
if uploaded_file is not None and submit_button:
    original_image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal Şəkil")
        st.image(original_image, channels="BGR", use_container_width=True)

    with st.spinner(f"'{model_choice}' ilə obyektlər təsbit edilir..."):
        all_predictions = []
        output_image = original_image.copy()

        if model_choice == 'Roboflow (Server)':
            model = load_roboflow_model()
            if model:
                image_for_model = cv2.resize(original_image, (640, 640), interpolation=cv2.INTER_AREA)
                predictions_json = model.predict(image_for_model, confidence=confidence_threshold).json()
                all_predictions = predictions_json.get('predictions', [])
                output_image = draw_roboflow_predictions(original_image, predictions_json)
        
        else: 
            try:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                payload = {'confidence': confidence_threshold}
                response = requests.post(BACKEND_API_URL, files=files, data=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    backend_detections = data.get('detections', [])
                    output_image = draw_backend_predictions(original_image, backend_detections)
                    for det in backend_detections:
                        all_predictions.append({'class': det['class_name'], 'confidence': det['confidence_score']})
                else:
                    st.error(f"Backend xətası (Kod: {response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend-ə qoşulmaq mümkün olmadı! Zəhmət olmasa, `uvicorn main:app --reload` əmrinin işlədiyindən əmin olun.")
            except Exception as e:
                st.error(f"Gözlənilməz bir xəta baş verdi: {e}")

        with col2:
            st.subheader("Nəticə")
            st.image(output_image, channels="BGR", use_container_width=True)
            
    st.subheader("Statistika")
    total_objects = len(all_predictions)
    if total_objects > 0:
        stats = Counter(p['class'] for p in all_predictions)
        summary_text = f"**Cəmi {total_objects} obyekt tapıldı:**"
        details_list = [f"{count} {LABEL_MAP.get(name, DEFAULT_STYLE)['name']}" for name, count in stats.items()]
        st.markdown(f"{summary_text} {', '.join(details_list)}.")
    else:
        st.warning("Bu güvən eşiyi ilə heç bir obyekt tapılmadı.")
    
elif not submit_button:
     st.info("Başlamaq üçün soldakı paneldən bir şəkil yükləyin, modeli seçin və 'Obyektləri Təsbit Et' düyməsinə basın.")