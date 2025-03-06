// script.js

function capitalizeWord(word) {
    // Special case for "us" to become "US"
    if (word.toLowerCase() === 'us') {
        return 'US';
    }
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

function formatFileName(fileName) {
    const [baseName] = fileName.split('.');
    const words = baseName.split('_');
    const formattedName = words
        .map(word => capitalizeWord(word))
        .join(' ');

    return formattedName;
}

function changeImage() {
    const select = document.getElementById('imageSelect');
    const image = document.getElementById('selectedImage');

    if (select.value) {
        image.src = select.value;
        image.alt = formatFileName(select.value)
        image.style.display = 'block'; // Show image when selected
    } else {
        image.style.display = 'none'; // Hide image when no selection
    }
}

function changeUKButton(){

    const image = document.getElementById('selectedImage');
    image.src = 'plots/uk_ge_voting_intention_next.png';
    image.alt = 'UK General Election Voting Intention';
    image.style.display = 'block';
    
    isZoomed = false;
    zoomIcon.src = 'assets/zoom_icon.svg';
    zoomIcon.alt = 'Zoom In';
    zoomButton.style.display = 'flex';
}

function changeUSButton(){

    const image = document.getElementById('selectedImage');
    image.src = 'plots/us_presidential_net_approval_current_term.png';
    image.alt = 'US Presidential Voting Intention';
    image.style.display = 'block';
    
    isZoomed = false;
    zoomIcon.src = 'assets/zoom_icon.svg';
    zoomIcon.alt = 'Zoom In';
    zoomButton.style.display = 'flex';
}

// Variables to track zoomed state and alternate images
let isZoomed = false;
const zoomPairs = {
    'plots/us_presidential_net_approval_current_term.png': 'plots/us_presidential_net_approval_2017_present.png',
    'plots/us_presidential_net_approval_2017_present.png': 'plots/us_presidential_net_approval_current_term.png',
    'plots/uk_ge_voting_intention_next.png': 'plots/uk_ge_voting_intention_next_2010.png',
    'plots/uk_ge_voting_intention_next_2010.png': 'plots/uk_ge_voting_intention_next.png'
};

// Function to toggle zoom state
function toggleZoom() {
    const image = document.getElementById('selectedImage');
    const zoomIcon = document.getElementById('zoomIcon');

    // Get current image source
    const currentSrc = image.src.split('/').pop();
    const fullCurrentSrc = 'plots/' + currentSrc;

    // Check if we have an alternate image for this one
    if (zoomPairs[fullCurrentSrc]) {
        // Switch to alternate image
        image.src = zoomPairs[fullCurrentSrc];

        // Toggle zoom state and icon
        isZoomed = !isZoomed;
        zoomIcon.src = isZoomed ? 'assets/zoom_icon.svg' : 'assets/zoom_icon.svg';
        zoomIcon.alt = isZoomed ? 'Zoom Out' : 'Zoom In';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const zoomButton = document.getElementById('zoomButton');
    zoomButton.addEventListener('click', toggleZoom);
});