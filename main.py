import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "best.pt"

app = FastAPI(
    title="Lokal Obyekt Tanıma API",
    description="Bu API, şəkilləri qəbul edir və YOLOv8 modeli ilə obyektləri tanıyaraq nəticələri JSON formatında qaytarır.",
    version="1.0.0"
)


origins = [
    "http://localhost",
    "http://localhost:8501", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

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

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Yalnız şəkil faylları qəbul edilir (image/jpeg, image/png, etc.)."})

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Şəkil faylı oxuna bilmir və ya xarabdır."})

    results = model(image, conf=confidence)

    formatted_detections = []
    for box in results[0].boxes.data:
        box_data = box.cpu().numpy().tolist()
        x1, y1, x2, y2, confidence, class_id = box_data
        
        formatted_detections.append({
            "box_coordinates": [int(x1), int(y1), int(x2), int(y2)],
            "class_name": class_names[int(class_id)],
            "confidence_score": round(float(confidence), 4) 
        })

    return JSONResponse(content={
        "filename": file.filename,
        "detections_count": len(formatted_detections),
        "detections": formatted_detections
    })