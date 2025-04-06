# AI Surveillance Pro+

**AI Surveillance Pro+** is a real-time object detection and tracking system powered by YOLOv8 and ByteTrack. It monitors webcam input, detects objects, tracks them, logs events, and alerts when objects enter a defined zone.

## 🔧 Features

- YOLOv8 object detection (all classes)
- ByteTrack object tracking with unique IDs
- Real-time webcam input
- Zone intrusion detection
- Trajectory visualization
- Voice alerts on zone entry
- CSV logging of events
- Saves annotated video to disk

## 📁 Project Structure
      ai_surveillance_pro/
├── main.py                  # Main script to run the surveillance system
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
├── yolov8n.pt               # Pre-trained YOLOv8 Nano model weights
├── zone_config.json         # JSON file for zone-specific configuration
├── logs/                    # Logs generated during execution
│   └── log.txt              # Example log file
├── outputs/                 # Captured outputs like snapshots, videos
│   └── snapshots/           # Saved snapshots of detected persons
└── test 2/                  # Project folder (if still nested, optional to show)

