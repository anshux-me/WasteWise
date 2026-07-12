// API Configuration
// Fallback to the live Render backend if VITE_API_URL is not set
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://wastewise-2-vnij.onrender.com";

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const reuploadBtn = document.getElementById('reuploadBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const resultContent = document.getElementById('resultContent');
const emptyState = document.getElementById('emptyState');
const themeToggle = document.getElementById('themeToggle');

let selectedFile = null;

// Theme Toggle
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
});

// Load saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
    document.body.classList.remove('dark-mode');
} else {
    document.body.classList.add('dark-mode');
    if (!savedTheme) {
        localStorage.setItem('theme', 'dark');
    }
}

// Drag and Drop Events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// Browse Button Click
browseBtn.addEventListener('click', () => {
    fileInput.value = ''; // Clear previous file to allow re-selecting the same file
    fileInput.click();
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPG, PNG, JPEG)');
        return;
    }

    // Validate file size (5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert('File size must be less than 5MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        uploadArea.style.display = 'none';
        analyzeBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Remove Button
removeBtn.addEventListener('click', () => {
    resetUploadFlow();
});

// Reupload Button
reuploadBtn.addEventListener('click', () => {
    resetUploadFlow(true);
});

// Analyze Button
analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    analyzeImage();
});

// Analyze Image Function
async function analyzeImage() {
    // Show loading state
    analyzeBtn.style.display = 'none';
    loading.style.display = 'block';
    resultCard.style.display = 'none';
    emptyState.style.display = 'none';

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        // Hide loading
        loading.style.display = 'none';

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        loading.style.display = 'none';
        analyzeBtn.style.display = 'block';
        alert('Error analyzing image. Please try again.');
    }
}

// Display Results
function displayResults(data) {
    const prediction = data.prediction;
    const confidence = data.confidence;
    const confidencePercent = Math.round(confidence); // confidence is already 0-100 from backend

    let resultHTML = `
        <div class="result-item">
            <div class="result-classification">
                <div class="classification-icon">
                    ${prediction === 'Recyclable' ? '♻️' : '🗑️'}
                </div>
                <div class="classification-label ${prediction === 'Recyclable' ? 'recyclable-label' : 'non-recyclable-label'}">
                    ${prediction}
                </div>
                <div class="classification-description">
                    ${prediction === 'Recyclable' ? 'This item is recyclable.' : 'This item is not recyclable.'}
                </div>
            </div>
        </div>

        <div class="result-item">
            <div class="confidence-section">
                <div class="confidence-label">
                    <span>Confidence Score</span>
                    <span class="confidence-value">${confidencePercent}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${confidencePercent}%"></div>
                </div>
            </div>
        </div>

        <div class="result-item">
            <div class="reason-section">
                <h4>
                    ${prediction === 'Recyclable' ? '🌱' : '⚠️'} 
                    ${prediction === 'Recyclable' ? 'Why is it recyclable?' : 'Why is it not recyclable?'}
                </h4>
                <p>
                    ${getExplanation(prediction)}
                </p>
            </div>
        </div>

        <div class="result-item">
            <div class="action-section">
                <h4>🗑️ Suggested Action</h4>
                <p>
                    ${getSuggestedAction(prediction)}
                </p>
            </div>
        </div>
    `;

    resultContent.innerHTML = resultHTML;
    resultCard.style.display = 'block';
    emptyState.style.display = 'none';
}

// Get Explanation for Result
function getExplanation(prediction) {
    const explanations = {
        'Recyclable': 'This item is made of recyclable material such as plastic, paper, glass, or metal, which can be processed and reused to create new products.',
        'Non-Recyclable': 'This item is made of materials that cannot be processed through standard recycling facilities and should be disposed of as regular waste.'
    };
    return explanations[prediction] || 'Analysis complete.';
}

// Get Suggested Action
function getSuggestedAction(prediction) {
    const actions = {
        'Recyclable': 'Please dispose of this item in the recycling bin to help protect our environment and save natural resources.',
        'Non-Recyclable': 'Please dispose of this item in the regular waste bin. Consider reducing single-use items to minimize environmental impact.'
    };
    return actions[prediction] || 'Please dispose of this item appropriately.';
}

// Make upload area clickable
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

function resetUploadFlow(openPicker = false) {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.style.display = 'none';
    loading.style.display = 'none';
    resultCard.style.display = 'none';
    emptyState.style.display = 'block';

    if (openPicker) {
        fileInput.click();
    }
}
