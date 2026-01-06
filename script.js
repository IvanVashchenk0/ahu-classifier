// script.js

const MODEL_URL = 'AHUmodel/model.json'; // path to Teachable Machine model
const imageUpload = document.getElementById('imageUpload');
const folderUpload = document.getElementById('folderUpload');
const processFolderBtn = document.getElementById('processFolder');
const previewImg = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const batchResultsDiv = document.getElementById('batchResults');
const loadingDiv = document.getElementById('loading');

let model;
let selectedFiles = [];

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

/**
 * Handle folder selection
 */
function handleFolderSelection(event) {
  selectedFiles = Array.from(event.target.files).filter(file => 
    file.type.startsWith('image/')
  );
  
  if (selectedFiles.length > 0) {
    processFolderBtn.style.display = 'inline-block';
    processFolderBtn.textContent = `Process ${selectedFiles.length} images`;
  } else {
    processFolderBtn.style.display = 'none';
  }
}

/**
 * Process all images in the selected folder
 */
async function processFolder() {
  if (!model) {
    alert('Model not loaded yet. Please wait.');
    return;
  }

  if (selectedFiles.length === 0) {
    alert('No images selected.');
    return;
  }

  processFolderBtn.disabled = true;
  processFolderBtn.textContent = 'Processing...';
  batchResultsDiv.style.display = 'block';
  batchResultsDiv.innerHTML = '<p>Processing images, please wait...</p>';

  const results = {
    'Blower': [],
    'Filter': [],
    'Coil': [],
    'Motor': [],
    'Exterior': [],
    'Damper': [],
    'Ambiguous': []
  };

  const labels = await getLabels();

  // Process each image
  for (let i = 0; i < selectedFiles.length; i++) {
    const file = selectedFiles[i];
    const fileName = file.name;
    
    try {
      const img = await loadImageFromFile(file);
      const classification = await classifyImage(img, labels);
      
      if (classification.category === 'Ambiguous') {
        results['Ambiguous'].push(fileName);
      } else {
        results[classification.category].push(fileName);
      }
    } catch (error) {
      console.error(`Error processing ${fileName}:`, error);
      results['Ambiguous'].push(fileName);
    }

    // Update progress
    const progress = Math.round(((i + 1) / selectedFiles.length) * 100);
    batchResultsDiv.innerHTML = `<p>Processing... ${progress}% (${i + 1}/${selectedFiles.length})</p>`;
  }

  // Display results
  displayBatchResults(results, selectedFiles.length);
  processFolderBtn.disabled = false;
  processFolderBtn.textContent = `Process ${selectedFiles.length} images`;
}

/**
 * Load an image from a File object
 */
function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * Classify a single image and return the category
 */
async function classifyImage(img, labels) {
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

  const probs = Array.from(data);

  // Sort by probability descending
  const sorted = probs
    .map((prob, idx) => ({ label: labels[idx] || `Class ${idx}`, prob }))
    .sort((a, b) => b.prob - a.prob);

  // Decision logic (same as single image prediction)
  const top1 = sorted[0];
  const top2 = sorted[1];

  if (top1.prob >= 0.75) {
    return { category: top1.label, confidence: top1.prob };
  } else if (top1.prob >= 0.4 && top2.prob >= 0.4) {
    return { category: 'Ambiguous', confidence: top1.prob };
  } else {
    return { category: 'Ambiguous', confidence: top1.prob };
  }
}

/**
 * Display batch processing results
 */
function displayBatchResults(results, totalImages) {
  let html = '<h2 style="font-size: 1.3rem; margin-bottom: 1rem; text-align: center;">Batch Processing Results</h2>';
  html += `<p style="text-align: center; margin-bottom: 1rem; font-weight: bold;">Total images processed: ${totalImages}</p>`;

  // Display each category that has results
  const categories = ['Blower', 'Filter', 'Coil', 'Motor', 'Exterior', 'Damper', 'Ambiguous'];
  
  categories.forEach(category => {
    const files = results[category];
    if (files.length > 0) {
      const count = files.length;
      const label = category === 'Ambiguous' ? 'Ambiguous or Unidentified' : category.toLowerCase() + (count === 1 ? '' : 's');
      
      html += `<div class="category-section">`;
      html += `<div class="category-title">Found ${count} ${label}</div>`;
      html += `<ul class="file-list">`;
      files.forEach(fileName => {
        html += `<li>${fileName}</li>`;
      });
      html += `</ul></div>`;
    }
  });

  batchResultsDiv.innerHTML = html;
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  loadModel();
  imageUpload.addEventListener('change', handleImageUpload);
  folderUpload.addEventListener('change', handleFolderSelection);
  processFolderBtn.addEventListener('click', processFolder);
}); 