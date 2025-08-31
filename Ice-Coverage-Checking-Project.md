







# Complete Mermaid Diagram for Ice Coverage Checking Project
```mermaid
flowchart TD
    %% Start of the process
    Start([Start Ice Coverage Checking Project]) --> Setup[Project Setup Phase]
   
    %% Setup Phase
    subgraph SetupPhase [Initial Setup]
        direction TB
        S1[Create Folder Structure]
        S2[Install Python Dependencies<br>pip install -r requirements.txt]
        S3[Configure Azure OpenAI<br>Create OpenAI Resource & Get Credentials]
        S4[Prepare Models<br>Fine-tune YOLOv8 for Crate Detection & Safety Detection]
        S5[Prepare Input Video<br>Place surveillance_video.mp4 in input/]
        S6[Set Environment Variables<br>Add Azure credentials to .env file]
    end
   
    Setup --> SetupPhase
    SetupPhase --> RunMain[Execute Main Script: python main.py]
   
    %% Detection Phase
    RunMain --> Detect[Object Detection Phase]
   
    subgraph DetectPhase [Crate & Safety Detection]
        direction TB
        D1[Load Video Frames from input/surveillance_video.mp4]
        D2[Detect Crates with YOLOv8]
        D3[Track Crates Across Frames using SORT]
        D4[Detect Workers & Safety Gear with YOLOv8]
        D5[Track Workers Across Frames using SORT]
        D6[Handle Detection Errors & Empty Frames]
    end
   
    Detect --> DetectPhase
    DetectPhase --> Analyze[Analysis Phase]
   
    %% Analysis Phase
    subgraph AnalyzePhase [Ice Coverage & Safety Analysis]
        direction TB
        A1[Crop Detected Crate Regions]
        A2[Encode Crop as Base64 for Azure OpenAI]
        A3[Query Azure OpenAI GPT-4o for Ice Coverage Percentage]
        A4[Apply Threshold & Stability Check for Alerts]
        A5[Associate Safety Gear to Workers using IoU]
        A6[Check Worker Safety Compliance]
        A7[Handle API Errors & Fallbacks]
    end
   
    Analyze --> AnalyzePhase
    AnalyzePhase --> Annotate[Annotation Phase]
   
    %% Annotation Phase
    subgraph AnnotatePhase [Frame Annotation]
        direction TB
        T1[Draw Bounding Boxes for Crates]
        T2[Add Text for Coverage Percentage & Alerts]
        T3[Draw Bounding Boxes for Workers & Gear]
        T4[Add Worker Compliance Labels]
        T5[Write Annotated Frames to Output Video]
    end
   
    Annotate --> AnnotatePhase
    AnnotatePhase --> Output[Save Annotated Video to output/annotated_output.mp4]
   
    %% Validation Phase
    Output --> Validate[Validation Phase]
   
    subgraph ValidatePhase [Output Validation]
        direction TB
        V1[Play Annotated Video]
        V2[Check Alert Accuracy & Annotations]
        V3[Verify Coverage Estimates Match Visuals]
        V4[Check Safety Compliance Labels]
        V5[Evaluate Metrics: mAP, Precision, Recall, IoU]
    end
   
    Validate --> ValidatePhase
    ValidatePhase --> Decision{Is Performance Satisfactory?}
   
    %% Decision Paths
    Decision -->|No| Adjust[Adjust Models or Thresholds]
    Adjust --> UpdateModels[Retrain YOLO or Tune Prompts]
    UpdateModels --> RunMain
   
    Decision -->|Yes| Success[Success! Process Complete]
    Success --> FinalOutput[Final Output: Annotated Video with Alerts & Compliance Checks]
   
    %% Azure Services Connection
    AzureOpenAI[Azure OpenAI Service (GPT-4o)]
   
    A3 -.-> AzureOpenAI
   
    %% External Files
    EnvFile[.env Environment Variables]
    InputFile[surveillance_video.mp4 Input Video]
   
    S6 -.-> EnvFile
    S5 -.-> InputFile
   
    %% Styling
    style Start fill:#4CAF50,color:white
    style Success fill:#4CAF50,color:white
    style FinalOutput fill:#4CAF50,color:white
    style AzureOpenAI fill:#2196F3,color:white
   
    classDef phase fill:#E1F5FE,stroke:#01579B,stroke-width:2px
    class SetupPhase,DetectPhase,AnalyzePhase,AnnotatePhase,ValidatePhase phase
```
****
# Complete Scenario of the Ice Coverage Checking Project
## Overall Scenario (What the Project Does):
The requirements (what you need to set it up), and finally how to achieve the solution (step-by-step implementation and execution). This project is designed to automate the monitoring of ice coverage on shrimp crates in a surveillance video, using computer vision for detection and Azure OpenAI for analysis, to ensure quality control in a loading process. It also includes worker safety compliance checks as a bonus feature.
#### 1. Project Scenario: What It Does and Why
- **Problem it Solves**: In seafood processing, shrimps in crates must be adequately covered with ice to maintain freshness and prevent spoilage during loading and transport. Manual inspection is labor-intensive, error-prone, and not scalable for real-time monitoring. Additionally, ensuring workers wear proper safety gear (coats, gloves, hats) is crucial for compliance. This project automates both by:
  - Detecting and tracking shrimp crates in a 5-minute surveillance video.
  - Analyzing the top layer of each crate for ice coverage using AI vision.
  - Triggering alerts for insufficient coverage (e.g., <80% ice).
  - Detecting workers, tracking them, associating safety gear via IoU, and flagging non-compliance.
- **Use Case Example**: In a warehouse, workers load shrimp crates onto carts. The system processes the video in near real-time, flags under-iced crates for immediate correction, and logs safety violations. This ensures product quality, reduces waste, and enhances safety.
- **Key Workflow**:
  1. Input: Surveillance video (e.g., `surveillance_video.mp4`).
  2. Processing: Detect/track crates & workers → Crop regions → Analyze with Azure OpenAI → Associate gear & check compliance → Annotate and alert.
  3. Output: Annotated video with bounding boxes, coverage percentages, alerts, and compliance labels.
- **Benefits**: Real-time alerts prevent quality issues, improves efficiency, scalable to longer videos or live streams, and integrates safety checks.
- **Limitations in Scenario**: Depends on video quality (lighting, angles); Azure API costs for analysis; may require fine-tuning for specific environments; not fully real-time on low-end hardware; IoU association assumes gear overlaps with worker bboxes.
This scenario fits real-world applications in food processing, quality assurance, and industrial safety monitoring.
#### 2. Requirements: What You Need
To run this project, you'll need hardware, software, services, and files. Here's a breakdown:
- **Hardware/Environment**:
  - A computer with Python 3.8+ installed, preferably with a GPU for faster processing.
  - Internet access for Azure OpenAI API calls.
  - Sufficient RAM/CPU/GPU (e.g., 8GB+ RAM, NVIDIA GPU recommended for YOLO).
- **Software Dependencies** (Listed in `requirements.txt`):
  - opencv-python: For video processing and annotations.
  - ultralytics: For YOLOv8 object detection.
  - filterpy: For SORT tracking (via pip install filterpy).
  - azure-openai: For Azure OpenAI integration.
  - python-dotenv: For loading environment variables.
  - torch: For PyTorch (YOLO dependency).
  Install with: `pip install -r requirements.txt`.
- **Azure Services** (Required for Analysis):
  - **Azure OpenAI**: For vision-based ice coverage estimation using GPT-4o. Create a resource in Azure Portal, deploy GPT-4o, get endpoint and key.
  - Costs: Pay-per-use (e.g., ~$0.01 per image analysis); free tier for testing.
- **Files and Folders**:
  - `.env`: For Azure credentials (see template below).
  - `models/`: Folder for fine-tuned YOLO models (e.g., `crate_yolo.pt` for crates, `safety_yolo.pt` for persons and gear).
  - `input/surveillance_video.mp4`: The source video.
  - `output/annotated_output.mp4`: Generated automatically.
- **Skills/Knowledge**:
  - Basic Python and computer vision understanding.
  - Familiarity with Azure Portal for setting up OpenAI.
  - Dataset for fine-tuning YOLO (e.g., annotated images of crates, workers, and safety gear).
If you lack Azure access, alternatives like local models (e.g., segmentation with U-Net) could be adapted, but the project is optimized for Azure OpenAI vision.
#### 3. How to Achieve the Solution: Step-by-Step Guide
Here's how the project works under the hood and how you can set it up, run it, and customize it. The code in `main.py` automates this end-to-end.
- **Step 1: Project Setup**
  - Create the folder structure as shown.
  - Install dependencies: `pip install -r requirements.txt`.
  - Fill `.env` with your Azure details.
  - Fine-tune YOLOv8: Use Ultralytics to train on labeled datasets (crates for one model, persons/safety gear for the other).
  - Place your input video in `input/`.
- **Step 2: Running the Project**
  - Execute: `python main.py`.
  - What Happens Internally (Code Flow with Comments Reference):
    - Loads environment variables (.env).
    - Initializes YOLO models and Azure OpenAI client.
    - Processes video frames (skipping for efficiency).
    - Detects and tracks crates with YOLO and SORT.
    - Crops crate regions, encodes to base64, queries Azure OpenAI for coverage.
    - Applies thresholds and stability checks for alerts.
    - Detects and tracks workers with YOLO and SORT; associates gear using IoU; checks compliance.
    - Annotates frames and saves output video.
- **Step 3: Achieving Customization and Enhancements**
  - **Adjust Thresholds**: Tweak `coverage_threshold`, `min_consecutive_flags`, or `iou_threshold` in code.
  - **Improve Detection**: Retrain YOLO on more data; adjust frame_skip for speed/accuracy.
  - **Enhance Safety Module**: Improve IoU logic or add more gear types.
  - **Error Handling**: Add retries for Azure API; fallback to classical CV (e.g., color thresholding) if API fails.
  - **Scaling**: For live streams, replace VideoCapture with RTSP; batch API calls for efficiency.
  - **Testing**: Use sample videos; evaluate with metrics (mAP for detection, accuracy for coverage/compliance vs. ground truth).
  - **Deployment**: Run on edge devices (e.g., Jetson); integrate alerts with email/SMS via libraries like smtplib.
****
### Complete Ice Coverage Checking Project with Step-by-Step Comments
Below is the full "ice-coverage-checker" project. I've added detailed # comments to **main.py** explaining each step of the code (e.g., setup, detection, analysis, annotation). The code uses YOLO for detection, SORT for tracking, and Azure OpenAI for ice analysis. The bonus worker safety compliance module is fully implemented and enabled.
#### Folder Structure
```
ice-coverage-checker/
├── requirements.txt # Dependencies to install via pip
├── .env # Azure credentials (do not share)
├── models/ # Folder for YOLO models
│ ├── crate_yolo.pt # Fine-tuned YOLO for crates
│ └── safety_yolo.pt # Fine-tuned YOLO for safety (persons, coats, gloves, hats)
├── input/ # Folder for input video
│ └── surveillance_video.mp4 # Place your actual video here
├── output/ # Folder for generated file (created automatically)
│ └── annotated_output.mp4 # Output will be saved here
└── main.py # Main script with step-by-step comments
```
#### 1. requirements.txt
```
opencv-python==4.10.0.84
ultralytics==8.2.82
filterpy==1.4.5
azure-openai==1.0.0-beta.5
python-dotenv==1.0.1
torch==2.4.0
```
#### 2. .env (Template - Fill with your Azure details)
```
AZURE_OPENAI_ENDPOINT=your_openai_endpoint_here
AZURE_OPENAI_API_KEY=your_openai_key_here
```
#### 3. main.py (With # Comments Explaining Each Step)
```python
import cv2
import numpy as np
from ultralytics import YOLO  # For YOLOv8
from sort import Sort  # SORT tracker (requires filterpy)
import os
import base64
from openai import AzureOpenAI  # For Azure OpenAI vision model
from dotenv import load_dotenv

# Step 0: Load environment variables from .env file
# This loads Azure keys for secure access without hardcoding
load_dotenv()

# Step 1: Load pre-trained models
# Crate detector: YOLOv8 fine-tuned on crate detection (class: 'crate' = 0)
crate_detector = YOLO('models/crate_yolo.pt')  # Replace with your model path

# Step 1a: Set up Azure OpenAI client for vision-based ice coverage analysis
# Uses GPT-4o with vision capabilities for estimating ice coverage
aoai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

# Step 1b: Load safety detector: YOLOv8 fine-tuned for persons and gear
# Classes: 'person' = 0, 'coat' = 1, 'gloves' = 2, 'hat' = 3
safety_detector = YOLO('models/safety_yolo.pt')  # Enabled for bonus module

# Step 2: Initialize trackers
# SORT for multi-object tracking: one for crates, one for persons
crate_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
person_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

# Step 3: Video processing parameters
# Adjust these for your needs
video_path = 'input/surveillance_video.mp4'
output_path = 'output/annotated_output.mp4'
coverage_threshold = 80  # 80% ice coverage required (integer)
frame_skip = 5  # Process every 5th frame for efficiency
min_consecutive_flags = 3  # Require 3 consecutive FAILs to trigger alert
iou_threshold = 0.5  # IoU threshold for associating gear to persons

# Step 3a: Helper function to compute IoU between two bboxes
# Used for associating safety gear to workers
def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Step 4: Load video and prepare output writer
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Unable to open video file")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / frame_skip, (width, height))

# Temporary storage for tracking alerts (crate_id: consecutive_fails)
alert_tracker = {}

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames for efficiency
    
    # Step 5: Detect crates using YOLO
    # Run inference to get bounding boxes and confidences
    results = crate_detector(frame)
    detections = []  # Format for SORT: [x1, y1, x2, y2, conf]
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0:  # 'crate'
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
    
    # Step 6: Track crates
    # Update tracker with new detections; returns [x1, y1, x2, y2, id]
    crate_trackers = crate_tracker.update(np.array(detections) if detections else np.empty((0, 5)))
    
    for track in crate_trackers:
        x1, y1, x2, y2, crate_id = track.astype(int)
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Crate #{crate_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Step 7: Crop crate top region
        # Assume bounding box covers the top; adjust for perspective if needed
        crate_crop = frame[y1:y2, x1:x2]
        if crate_crop.size == 0:
            continue
        
        # Step 8: Prepare image for Azure OpenAI (encode to base64)
        _, buffer = cv2.imencode('.jpg', crate_crop)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Step 9: Query Azure OpenAI GPT-4o for ice coverage estimation
        # Prompt engineered for precise integer output
        try:
            response = aoai_client.chat.completions.create(
                model="gpt-4o",  # Your vision-capable deployment name
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in analyzing images of shrimp crates for ice coverage."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image of a shrimp crate. Estimate the percentage of the top layer covered with ice (white/crystalline areas) vs. exposed shrimps (reddish/pink). Provide only the percentage as an integer (0-100)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.0  # For deterministic output
            )
            
            # Parse response to get integer coverage
            coverage_str = response.choices[0].message.content.strip()
            coverage = int(coverage_str) if coverage_str.isdigit() else 0
            
        except Exception as e:
            print(f"Error in Azure OpenAI call: {e}")
            coverage = 0  # Fallback on error
        
        # Display coverage on frame
        color = (0, 255, 0) if coverage >= coverage_threshold else (0, 0, 255)
        cv2.putText(frame, f'{coverage}% Ice', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Step 10: Thresholding and alerting with stability check
        # Ensures alerts are not triggered by transient errors
        if coverage < coverage_threshold:
            alert_tracker[crate_id] = alert_tracker.get(crate_id, 0) + 1
            if alert_tracker[crate_id] >= min_consecutive_flags:
                cv2.putText(frame, 'ALERT: Insufficient Ice!', (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f'ALERT: Crate #{crate_id} has insufficient ice ({coverage}%)')  # Can integrate external alerts here
        else:
            alert_tracker[crate_id] = 0  # Reset counter
    
    # Bonus: Worker Safety Compliance (Fully implemented and enabled)
    # Step 11: Detect persons and safety gear using YOLO
    safety_results = safety_detector(frame)
    person_detections = []  # For SORT: [x1, y1, x2, y2, conf]
    gear_detections = {1: [], 2: [], 3: []}  # coat=1, gloves=2, hat=3: list of [x1, y1, x2, y2]
    for r in safety_results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            if cls == 0:  # 'person'
                person_detections.append([x1, y1, x2, y2, conf])
            elif cls in [1, 2, 3]:  # gear
                gear_detections[cls].append([x1, y1, x2, y2])
    
    # Step 12: Track persons
    person_trackers = person_tracker.update(np.array(person_detections) if person_detections else np.empty((0, 5)))
    
    for p_track in person_trackers:
        px1, py1, px2, py2, person_id = p_track.astype(int)
        # Draw person bounding box and ID
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame, f'Worker #{person_id}', (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Step 13: Check safety compliance for this worker
        # For each gear type, check if any gear bbox has IoU > threshold with person bbox
        items = {'coat': False, 'gloves': False, 'hat': False}
        gear_map = {1: 'coat', 2: 'gloves', 3: 'hat'}
        person_box = [px1, py1, px2, py2]
        for cls, gears in gear_detections.items():
            for gear_box in gears:
                if compute_iou(person_box, gear_box) > iou_threshold:
                    items[gear_map[cls]] = True
                    # Optionally draw gear box
                    gx1, gy1, gx2, gy2 = map(int, gear_box)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 1)
        
        # Determine compliance and annotate
        compliant = all(items.values())
        label = 'Compliant' if compliant else f'Missing: {", ".join([k for k, v in items.items() if not v])}'
        color = (0, 255, 0) if compliant else (0, 0, 255)
        cv2.putText(frame, f'{label}', (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if not compliant:
            print(f'ALERT: Worker #{person_id} is non-compliant: {label}')
    
    # Step 14: Write annotated frame to output video
    out.write(frame)

# Step 15: Release resources and clean up
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Annotated video saved to {output_path}")
```










