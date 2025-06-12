# AHU Component Recognition Web App

This lightweight web application lets you take or upload a photo of an Air-Handling Unit (AHU) component and instantly classifies it using a Teachable Machine image model that runs entirely in the browser.

## Features

• Runs completely client-side — no server, no data leaves your device.

• Decision logic with clear confidence thresholds:
  * **✅ Single object detected** (≥ 75 % confidence)
  * **🔍 Multiple objects possibly detected** (≥ 40 % confidence for two classes)
  * **⚠️ Cannot identify** (otherwise)

## Getting Started

1. **Clone or download** this repository.

2. **Serve the directory** with a static file server so the browser can load the model files via HTTP. For example, using Python:

```bash
cd /path/to/project
python3 -m http.server 8000
```

3. Open your browser at `http://localhost:8000` and you should see the app.

4. Click **Choose File** or point your camera to capture a photo. The prediction result appears below the image preview.

## File Structure

```
├── index.html        # Main web page and UI
├── script.js         # Model loading + prediction logic
├── AHUmodel/         # Exported Teachable Machine model
│   ├── model.json
│   ├── metadata.json
│   └── weights.bin
└── README.md         # This file
```

## Customising

• **Styling** – Adjust CSS inside `index.html` or move it to a separate file.

• **Thresholds** – Edit `script.js` (`top1.prob >= 0.75`, etc.) to tweak the confidence logic.

• **Model** – Replace the contents of `AHUmodel` with your own exported model as long as you keep the same file names (`model.json`, `metadata.json`, `weights.bin`).

## License

MIT © 2025 Your Name 