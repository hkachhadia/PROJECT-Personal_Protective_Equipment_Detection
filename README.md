# Personal Protective Equipment (PPE) Detection System

An AI-powered computer vision application that detects Personal Protective Equipment (PPE) to help monitor safety compliance at construction sites. The system uses a trained YOLO object detection model and a Flask web application to provide a user-friendly interface for PPE detection.

## Features

- **AI-Based Detection:** Uses a trained YOLO model to detect PPE in images or video streams supported by the application.
- **Web Interface:** Provides a web interface built with Flask, HTML, and CSS.
- **Real-Time Monitoring:** Includes a monitoring page for viewing detection results, depending on the configured camera or video source.
- **Dashboard:** Displays detection-related information through the application's dashboard.
- **Modular Structure:** Keeps the application code, trained model, templates, and stylesheets organized separately.

## Technologies Used

- Python
- Flask
- YOLO / Ultralytics
- OpenCV
- NumPy
- Pandas
- HTML and CSS
- Jupyter Notebook

## Project Structure

```text
6th SEM PROJECT - PPE Detection/
├── models/
│   └── best.pt
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── base.html
│   ├── dashboard_table.html
│   ├── home.html
│   ├── monitor.html
│   └── technology.html
├── app.py
├── project6.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python installed on your system
- Git
- A compatible camera or video source if required by the monitoring functionality

### 1. Clone the Repository

```bash
git clone https://github.com/hkachhadia/PROJECT-Personal_Protective_Equipment_Detection.git
cd PROJECT-Personal_Protective_Equipment_Detection
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can activate the environment using Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the Trained Model

The application uses the trained model located at:

```text
models/best.pt
```

Make sure this file exists after cloning the repository. If the model is distributed separately in a future version, download it and place it in the `models` directory.

### 5. Download the Dataset

The dataset is not included in this GitHub repository because of its size.

**Dataset download:** https://drive.google.com/file/d/1ahcNjx2n6hHR5yF1Op4WyFPNM2wG5K8E/view?usp=drive_link

After downloading, extract the dataset and place it in the location expected by the notebook or training code. Dataset paths may need to be adjusted to match your local setup.

### 6. Run the Application

From the project root, execute:

```bash
python app.py
```

Open the local URL printed in the terminal by Flask, commonly:

```text
http://127.0.0.1:5000
```

Use the application's navigation to access the available pages and monitoring features.

## Model and Notebook

- **Trained model:** `models/best.pt`
- **Notebook:** `project6.ipynb`
- **Application entry point:** `app.py`

The notebook contains the project's experimental or development workflow. Review its dataset paths and setup instructions before running its cells.

## Important Notes

- The dataset is maintained separately from this repository.
- Do not upload private information, credentials, environment files, or unnecessary training outputs.
- Detection performance depends on the trained model, image quality, lighting, camera placement, and the PPE categories represented in the training data.
- This system is intended to assist safety monitoring; it should not replace established workplace safety procedures or human supervision.

## Future Improvements

- Improve detection accuracy across different construction environments.
- Add more detailed detection analytics and reporting.
- Improve monitoring and alert functionality.
- Support easier dataset configuration and model updates.

## Author

**Het Kachhadia**

GitHub: [@hkachhadia](https://github.com/hkachhadia)

