import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image


# -----------------
# KONFİQURASİYA
# -----------------

# Model faylının adı. Bu skriptlə eyni qovluqda olmalıdır.
MODEL_PATH = "best.pt"

# FastAPI tətbiqini yaratmaq
app = FastAPI(
    title="Lokal Obyekt Tanıma API",
    description="Bu API, şəkilləri qəbul edir və YOLOv8 modeli ilə obyektləri tanıyaraq nəticələri JSON formatında qaytarır.",
    version="1.0.0"
)

# -----------------
# MODELİN YÜKLƏNMƏSİ
# -----------------

# Modeli və sinif adlarını qlobal dəyişəndə saxlayaq
model = None
class_names = []

@app.on_event("startup")
def load_model():
    """
    API işə düşən zaman modeli yaddaşa yükləyən funksiya.
    Bu, hər sorğuda modeli yenidən yükləməyin qarşısını alır və performansı artırır.
    """
    global model, class_names
    try:
        model = YOLO(MODEL_PATH)
        class_names = model.names
        print("--------- Model Uğurla Yükləndi ---------")
        print(f"Model faylı: {MODEL_PATH}")
        print(f"Tanınan siniflər ({len(class_names)} ədəd): {class_names}")
        print("-----------------------------------------")
    except Exception as e:
        print(f"!!!!!!!!!! XƏTA !!!!!!!!!!!")
        print(f"Model yüklənərkən xəta baş verdi: {e}")
        print(f"Zəhmət olmasa '{MODEL_PATH}' faylının mövcud və düzgün olduğundan əmin olun.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        model = None

# -----------------
# API ENDPOINTLƏRİ
# -----------------

@app.get("/", tags=["Əsas"])
def read_root():
    """
    API-nin işlək olduğunu yoxlamaq üçün əsas endpoint.
    """
    if model is None:
        return {"status": "Xəta", "message": "Model yüklənə bilmədi. Server loglarına baxın."}
    return {"status": "İşləyir", "message": "Real model ilə işləyən backend hazırdır. /docs ünvanına gedərək test edə bilərsiniz."}


@app.post("/detect", tags=["Analiz"])
async def detect_objects_from_image(file: UploadFile = File(..., description="Analiz üçün şəkil faylı."), confidence: float = Form(0.25, description="Model üçün minimum güvən eşiyi (0.0-1.0).")):
    """
    Şəkil qəbul edir, onu `best.pt` modeli ilə analiz edir və
    tapılan obyektlərin məlumatlarını (koordinatlar, ad, əminlik faizi) JSON formatında qaytarır.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model mövcud deyil və ya yüklənməyib."})

    # Gələn faylın şəkil olub-olmadığını yoxlamaq üçün sadə validasiya
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Yalnız şəkil faylları qəbul edilir (image/jpeg, image/png, etc.)."})

    # Addım 1: Şəkli bayt formatında oxumaq və PIL obyektinə çevirmək
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Şəkil faylı oxuna bilmir və ya xarabdır."})

    # Addım 2: Modeli şəkil üzərində işə salmaq
    results = model(image, conf=confidence)

    # Addım 3: Nəticələri emal edərək standart JSON formatına gətirmək
    formatted_detections = []
    # results[0].boxes.data -> [x1, y1, x2, y2, confidence, class_id]
    for box in results[0].boxes.data:
        box_data = box.cpu().numpy().tolist() # GPU-da işləyirsə CPU-ya çəkmək üçün .cpu()
        x1, y1, x2, y2, confidence, class_id = box_data
        
        formatted_detections.append({
            "box_coordinates": [int(x1), int(y1), int(x2), int(y2)],
            "class_name": class_names[int(class_id)],
            "confidence_score": round(float(confidence), 4) # Nəticəni 4 rəqəm dəqiqliklə yuvarlaqlaşdıraq
        })

    return JSONResponse(content={
        "filename": file.filename,
        "detections_count": len(formatted_detections),
        "detections": formatted_detections
    })