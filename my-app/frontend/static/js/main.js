        // Animal classes array
const ANIMAL_CLASSES = [
    'gayal', 'giraffe', 'kangaru', 'lion', 'rhinoceros', 'sheep', 'tiger', 'zebra',
    'camel', 'camel bird', 'bear', 'panda', 'rabbit', 'cat', 'cow', 'crocodile',
    'deer', 'dog', 'elephant', 'goat', 'hargila bok', 'hippopotamus', 'horse',
    'kalo bok', 'lama', 'peacock', 'porcupine', 'ass', 'squirrel', 'monkey'
];

let predictionHistory = [];
let currentSlide = 0;

// Initialize slider
function initSlider() {
    const slides = document.querySelectorAll('.slide');
    setInterval(() => {
        slides[currentSlide].classList.remove('active');
        currentSlide = (currentSlide + 1) % slides.length;
        slides[currentSlide].classList.add('active');
    }, 4000);
}

// File handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('selectedImage').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

// Drag and drop handlers
const uploadZone = document.getElementById('uploadZone');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('selectedImage').classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    }
});


async function predictAnimal() {
    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        document.getElementById('uploadSection').classList.add('hidden');
        document.getElementById('loadingSection').classList.remove('hidden');

        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediction failed');
        
        const responseData = await response.json();
        const topPrediction = responseData.top_prediction;

        updatePredictionUI(topPrediction);
        addToHistory(topPrediction);
        
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error processing prediction. Please try again.');
        resetUI();
    }
}


function updatePredictionUI(data) {
    document.getElementById('predictedAnimal').textContent = data.animal;
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('scientificName').textContent = data.details.scientific || 'N/A';
    document.getElementById('habitat').textContent = data.details.habitat || 'N/A';
    document.getElementById('diet').textContent = data.details.diet || 'N/A';
    document.getElementById('status').textContent = data.details.status || 'N/A';
    document.getElementById('description').textContent = data.details.description || 'No description available';
}

function addToHistory(data) {
    const prediction = {
        animal: data.animal,
        confidence: (data.confidence * 100).toFixed(1),
        timestamp: new Date().toLocaleString(),
        image: document.getElementById('previewImage').src,
        details: data.details
    };
    
    predictionHistory.unshift(prediction);
    updateHistory();
}

async function loadHistoryItem(index) {
    const pred = predictionHistory[index];
    try {
        document.getElementById('loadingSection').classList.remove('hidden');
        
       
        const detailsResponse = await fetch(`/animal-details/${pred.animal}`);
        if (!detailsResponse.ok) throw new Error('Details not found');
        
        const details = await detailsResponse.json();
        
      
        document.getElementById('previewImage').src = pred.image;
        document.getElementById('selectedImage').classList.remove('hidden');
        updatePredictionUI({
            animal: pred.animal,
            confidence: pred.confidence / 100,
            model_confidences: {
                efficientnetv2: pred.confidence / 100,
                mobilenet: pred.confidence / 100,
                densenet: pred.confidence / 100
            },
            details: details
        });
        
        document.getElementById('uploadSection').classList.add('hidden');
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

    } catch (error) {
        console.error('Error loading history item:', error);
        alert('Error loading prediction history');
        resetUI();
    }
}

function updateHistory() {
    const historyPanel = document.getElementById('historyPanel');
    historyPanel.innerHTML = predictionHistory.length === 0 
        ? '<p class="text-gray-400 text-center">No predictions yet</p>'
        : predictionHistory.map((pred, index) => `
            <div class="history-item" onclick="loadHistoryItem(${index})">
                <div class="flex items-center space-x-3">
                    <img src="${pred.image}" alt="${pred.animal}" class="w-12 h-12 object-cover rounded">
                    <div>
                        <p class="font-semibold">${pred.animal}</p>
                        <p class="text-sm text-gray-400">${pred.confidence}% • ${pred.timestamp}</p>
                    </div>
                </div>
            </div>
        `).join('');
}

function newPrediction() {
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
    document.getElementById('selectedImage').classList.add('hidden');
    document.getElementById('fileInput').value = '';
}

function resetUI() {
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
    document.getElementById('selectedImage').classList.add('hidden');
}

// Theme management
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    const icons = document.querySelectorAll('#theme-icon, #mobile-theme-icon');
    icons.forEach(icon => {
        icon.className = document.body.classList.contains('light-theme') 
            ? 'fas fa-sun' 
            : 'fas fa-moon';
    });
}

// Search functionality
let searchTimeout;
document.getElementById('searchBar').addEventListener('input', function(e) {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(async () => {
        const query = e.target.value.trim().toLowerCase();
        if (query.length < 2) return;

        try {
            const response = await fetch(`/search-animals?query=${encodeURIComponent(query)}`);
            const data = await response.json();
            showSearchResults(data.results);
        } catch (error) {
            console.error('Search error:', error);
        }
    }, 300);
});

function showSearchResults(results) {
    
    console.log('Search results:', results);
    
}


function toggleMobileNav() {
    const mobileNav = document.querySelector('.mobile-nav');
    mobileNav.classList.toggle('active');
    document.body.classList.toggle('no-scroll');
}


document.addEventListener('click', (event) => {
    const mobileNav = document.querySelector('.mobile-nav');
    const menuBtn = document.querySelector('.mobile-menu-btn');
    
    if (!mobileNav.contains(event.target) && !menuBtn.contains(event.target)) {
        mobileNav.classList.remove('active');
        document.body.classList.remove('no-scroll');
    }
});


window.addEventListener('load', () => {
    initSlider();
    
});