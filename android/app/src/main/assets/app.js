const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const tabBtns = document.querySelectorAll('.tab-btn');
const viewContainers = document.querySelectorAll('.view-container');
const scannerGrid = document.getElementById('scanner-grid');

// In emulator this would point to the backend, e.g. http://10.0.2.2:8000/api
// In production it would be a deployed endpoint like https://your-trade-app.onrender.com/api
const API_BASE = 'https://your-trade-app.onrender.com/api'; 
const CHAT_URL = `${API_BASE}/chat`;
const SCANNER_URL = `${API_BASE}/scanner/results`;

let scannerInterval = null;

// Tab Navigation Logic
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all tabs and views
        tabBtns.forEach(t => t.classList.remove('active'));
        viewContainers.forEach(v => v.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding view
        btn.classList.add('active');
        const targetView = document.getElementById(btn.getAttribute('data-target'));
        targetView.classList.add('active');

        // Manage Scanner Polling
        if (btn.getAttribute('data-target') === 'view-scanner') {
            fetchScannerData(); // Fetch immediately
            scannerInterval = setInterval(fetchScannerData, 5000); // Polling every 5s
        } else {
            if (scannerInterval) {
                clearInterval(scannerInterval);
                scannerInterval = null;
            }
        }
    });
});

// Chat Logic
function addMessage(text, sender, isLoading = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(sender === 'user' ? 'user-msg' : 'bot-msg');
    if (isLoading) msgDiv.classList.add('loading');
    
    // Simple markdown-like bold parsing for bot responses
    if (sender === 'bot') {
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
    
    msgDiv.innerHTML = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return msgDiv;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    addMessage(text, 'user');
    chatInput.value = '';
    
    const loadingMsg = addMessage('Analyzing technicals and parsing news sentiment...', 'bot', true);
    
    try {
        const response = await fetch(CHAT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        chatWindow.removeChild(loadingMsg);
        addMessage(data.response, 'bot');
    } catch (error) {
        console.error('Error:', error);
        chatWindow.removeChild(loadingMsg);
        addMessage('Sorry, I am unable to reach the backend server right now.', 'bot');
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Scanner Logic
async function fetchScannerData() {
    try {
        const response = await fetch(SCANNER_URL);
        if (!response.ok) throw new Error('Failed to fetch scanner data');
        
        const data = await response.json();
        renderScannerGrid(data.candidates);
    } catch (error) {
        console.error('Scanner Fetch Error:', error);
    }
}

function renderScannerGrid(candidates) {
    scannerGrid.innerHTML = ''; // Clear existing
    
    if (!candidates || candidates.length === 0) {
        scannerGrid.innerHTML = '<div style="text-align: center; color: #a1a1aa; width: 100%; margin-top: 2rem; grid-column: 1 / -1;">The Macro Filter is still analyzing stocks. Please check back in a few moments...</div>';
        return;
    }

    candidates.forEach(cand => {
        const card = document.createElement('div');
        card.classList.add('scanner-card');
        
        const confClass = cand.prob > 60 ? 'confidence-high' : 'confidence-med';
        
        card.innerHTML = `
            <div class="card-header">
                <span class="card-ticker">${cand.ticker}</span>
                <span class="card-price">₹${cand.price.toFixed(2)}</span>
            </div>
            <div class="card-body">
                <div class="card-stat">Prob: <strong class="${confClass}">${cand.prob}%</strong></div>
                <div class="card-stat">RSI: <strong>${cand.rsi.toFixed(1)}</strong></div>
            </div>
            <div class="card-body">
                <div class="card-stat">Target: <strong>₹${cand.target.toFixed(2)}</strong></div>
                <div class="card-stat">Stop: <strong>₹${cand.stop_loss.toFixed(2)}</strong></div>
            </div>
        `;
        scannerGrid.appendChild(card);
    });
}
