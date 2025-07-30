# 📄 Exam Question Detector

This project is a web-based application designed to detect and classify different types of questions within an uploaded image of an exam paper. It leverages machine learning models to identify "open-ended" and "closed-ended" questions.

The application features a user-friendly interface built with Streamlit and a powerful backend powered by FastAPI and YOLOv8.

## ✨ Key Features

- **Dual Model Support:** Choose between two different detection models:
    - ☁️ **Roboflow Model:** A cloud-hosted model known for its speed and high accuracy.
    - 💻 **Local Backend Model:** A self-hosted YOLOv8 model that runs locally via a FastAPI server, ensuring data privacy.
- **Interactive Interface:** Upload an image and instantly see the detected questions highlighted with bounding boxes.
- **Confidence Threshold Adjustment:** Fine-tune the model's sensitivity by setting a confidence level for detections.
- **Detection Statistics:** View a summary of the detected objects, including the total count and a breakdown by question type.
- **User Guidance:** Includes helpful tips on how to capture and upload images to achieve the best detection results.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Object Detection:** Ultralytics YOLOv8
- **Cloud ML Platform:** Roboflow
- **Core Libraries:** OpenCV, Pillow, Requests

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` and `venv`
- For Debian/Ubuntu-based systems, you may need to install an additional system library.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/exam-question-detector.git
    cd exam-question-detector
    ```
    *(Replace the URL with the actual repository URL.)*

2.  **(For Debian/Ubuntu) Install system dependencies:**
    This project requires `libgl1-mesa-glx` for OpenCV to function correctly.
    ```bash
    sudo apt-get update && sudo apt-get install -y $(cat packages.txt)
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

4.  **Install Python packages:**
    > **Note:** The `requirements.txt` file is encoded in UTF-16, which can cause issues with `pip` on some systems. If the command below fails, please convert the file to UTF-8 first.
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Set up Roboflow API Keys:**
    To use the cloud-based Roboflow model, you need to provide your API credentials. Create a file at `.streamlit/secrets.toml` in the project root and add your keys:
    ```toml
    # .streamlit/secrets.toml
    ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
    ROBOFLOW_WORKSPACE = "YOUR_ROBOFLOW_WORKSPACE_ID"
    ROBOFLOW_MODEL = "YOUR_ROBOFLOW_MODEL_ID"
    ROBOFLOW_VERSION = "YOUR_ROBOFLOW_VERSION_NUMBER" # This should be a number, not a string
    ```

## ▶️ How to Run

After completing the installation, you can run the application. This requires two separate terminals if you intend to use the local backend model.

> **Note:** Make sure you have activated the virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) in each terminal before running the commands.

**1. Start the Backend Server (FastAPI):**

If you want to use the local YOLOv8 model, run the following command in your first terminal to start the API server:

```bash
uvicorn main:app --reload
```

You should see a confirmation that the server is running and the model has been loaded. You can access the API documentation at `http://127.0.0.1:8000/docs`.

**2. Start the Frontend Application (Streamlit):**

In your second terminal, run the following command to launch the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the application's user interface. From there, you can select the model ("Lokal" for the backend you just started, or "Roboflow" if you have set up the API keys), upload an image, and start detecting questions.

## 🤖 Model Performance

The application provides two models with different performance characteristics.

### ☁️ Roboflow Model (RF-DETR - Medium)

This is a cloud-based model known for high performance and fast inference times.

- **mAP@50:** 95.9%
- **Precision:** 91.9%
- **Recall:** 94.0%

### 💻 Local Model (RT-DETR-L)

This model runs on the local machine via the FastAPI backend. It was trained for 120 epochs over 2.46 hours on a Tesla T4 GPU.

- **Model Size:** 66.2 MB
- **Inference Speed:** ~31.4ms per image (on Tesla T4)

#### Validation Results (`best.pt`)

| Class         | Precision | Recall | mAP@50 | mAP@50-95 |
|:--------------|:----------|:-------|:-------|:----------|
| **all**       | 0.902     | 0.844  | 0.874  | 0.747     |
| `questions`   | 0.936     | 0.912  | 0.932  | 0.834     |
| `questions_o` | 0.868     | 0.776  | 0.816  | 0.660     |

## 📁 Project Structure

```
.
├── .streamlit/
│   └── secrets.toml      # (Optional) API keys for Roboflow model
├── fonts/
│   └── Roboto-Medium.ttf # Font for drawing labels on images
├── app.py                # Main script for the Streamlit frontend
├── best.pt               # Trained YOLOv8 model for the local backend
├── main.py               # FastAPI script for serving the local backend
├── packages.txt          # System-level dependencies (for Debian/Ubuntu)
└── requirements.txt      # Python package dependencies
```
