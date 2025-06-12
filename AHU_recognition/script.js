// script.js

const MODEL_URL = 'AHUmodel/model.json'; // path to Teachable Machine model
const imageUpload = document.getElementById('imageUpload');
const previewImg = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');

let model;

/**
 * Load the TensorFlow.js model exported from Teachable Machine.
 */
async function loadModel() {
  try {
    loadingDiv.style.display = 'block';
    model = await tf.loadLayersModel(MODEL_URL);
    loadingDiv.style.display = 'none';
    console.log('Model loaded.');
  } catch (err) {
    loadingDiv.textContent = 'Failed to load model.';
    console.error(err);
  }
}

/**
 * Handle file selection, preview the image, and trigger prediction.
 */
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.onload = () => {
      predict(previewImg);
    };
  };
  reader.readAsDataURL(file);
}

/**
 * Preprocess the image and run prediction.
 * @param {HTMLImageElement} img
 */
async function predict(img) {
  if (!model) {
    alert('Model not loaded yet. Please wait.');
    return;
  }

  tf.engine().startScope();
  // Convert image to tensor
  const tensor = tf.browser
    .fromPixels(img)
    .resizeBilinear([224, 224])
    .toFloat()
    .div(255.0)
    .expandDims(); // shape [1, 224, 224, 3]

  // Predict probabilities
  const predictions = model.predict(tensor);
  const data = await predictions.data();
  tf.engine().endScope();

  // Map probabilities to labels (assuming labels in metadata.json order)
  // Fetch labels only once and cache
  const labels = await getLabels();

  const probs = Array.from(data);

  // Sort by probability descending
  const sorted = probs
    .map((prob, idx) => ({ label: labels[idx] || `Class ${idx}`, prob }))
    .sort((a, b) => b.prob - a.prob);

  // Decision logic
  const top1 = sorted[0];
  const top2 = sorted[1];

  let message = '';
  if (top1.prob >= 0.75) {
    message = `✅ Detected: ${top1.label} (confidence ${(top1.prob * 100).toFixed(1)}%)`;
  } else if (top1.prob >= 0.4 && top2.prob >= 0.4) {
    message = `🔍 Possible objects: ${top1.label} and ${top2.label}`;
  } else {
    message = '⚠️ Cannot identify, image unclear.';
  }

  resultDiv.textContent = message;
}

/**
 * Retrieve labels from metadata.json file exported by Teachable Machine.
 * The function fetches and caches the labels on first call.
 */
let cachedLabels;
async function getLabels() {
  if (cachedLabels) return cachedLabels;
  try {
    const response = await fetch('AHUmodel/metadata.json');
    const metadata = await response.json();
    cachedLabels = metadata.labels;
    return cachedLabels;
  } catch (err) {
    console.error('Failed to load labels:', err);
    return [];
  }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  loadModel();
  imageUpload.addEventListener('change', handleImageUpload);
}); 