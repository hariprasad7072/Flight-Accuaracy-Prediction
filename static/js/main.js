// ═══════════════════════════════════════════════════════════
// AeroPredict AI — Main Application Script
// ═══════════════════════════════════════════════════════════

class AeroPredictApp {
    constructor() {
        this.predictionHistory = JSON.parse(localStorage.getItem('predictionHistory')) || [];
        this.currentPrediction = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadModelResults();
        this.setupRangeSliders();
        this.loadPredictionHistory();
    }

    // ─── Tab Navigation ───────────────────────────────────
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target));
        });

        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', 
            (e) => this.handlePrediction(e));

        // Get Weather Button
        const getWeatherBtn = document.getElementById('getWeatherBtn');
        if (getWeatherBtn) {
            getWeatherBtn.addEventListener('click', (e) => this.fetchWeatherData(e));
        }

        // Range sliders
        this.setupRangeSliders();

        // Clear history
        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearHistory());
        }

        // Export report
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportReport());
        }

        // Mode toggle (dark/light)
        const modeToggle = document.getElementById('modeToggle');
        if (modeToggle) {
            modeToggle.addEventListener('click', () => this.toggleMode());
        }
    }

    switchTab(btn) {
        // Remove active from all tabs
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));

        // Add active to clicked tab
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab') + 'Tab';
        document.getElementById(tabId).classList.remove('hidden');
    }

    setupRangeSliders() {
        document.querySelectorAll('.range-container input[type="range"]').forEach(slider => {
            const valSpan = document.getElementById(slider.name + '-val');
            if (valSpan) {
                // Set initial display
                this.updateRangeValueDisplay(slider);
                
                // Update on input
                slider.addEventListener('input', () => {
                    this.updateRangeValueDisplay(slider);
                });
            }
        });
    }
    
    updateRangeValueDisplay(slider) {
        const valSpan = document.getElementById(slider.name + '-val');
        if (!valSpan) return;
        
        const value = parseFloat(slider.value);
        const name = slider.name;
        
        if (name === 'distance') {
            valSpan.textContent = value + ' mi';
        } else if (name === 'temp') {
            valSpan.textContent = value + '°C';
        } else if (name === 'wind') {
            valSpan.textContent = value + ' km/h';
        } else if (name === 'vis') {
            valSpan.textContent = value + ' km';
        } else if (name === 'hum') {
            valSpan.textContent = value + '%';
        } else if (name === 'precip') {
            valSpan.textContent = value + ' mm';
        } else if (name === 'congestion' || name === 'load' || name === 'efficiency') {
            valSpan.textContent = (value * 100).toFixed(0) + '%';
        }
    }

    // ─── Prediction Handler ────────────────────────────────
    async fetchWeatherData(e) {
        e.preventDefault();
        
        const origin = document.querySelector('select[name="origin"]').value;
        const dest = document.querySelector('select[name="dest"]').value;
        const date = document.querySelector('input[name="date"]').value;
        const statusSpan = document.getElementById('weatherStatus');
        
        if (!origin || !dest) {
            if (statusSpan) statusSpan.textContent = '⚠️ Please select both airports';
            return;
        }
        
        try {
            if (statusSpan) statusSpan.textContent = '⏳ Fetching weather...';
            
            const response = await fetch('/api/fetch-all-weather', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    origin: origin,
                    dest: dest,
                    date: date
                })
            });
            
            const data = await response.json();
            console.log('Weather data:', data);
            
            if (data.success && data.origin) {
                const w = data.origin;
                
                // Update each field directly
                const tempInput = document.querySelector('input[name="temp"]');
                if (tempInput) {
                    tempInput.value = Math.round(w.temperature);
                }
                
                const windInput = document.querySelector('input[name="wind"]');
                if (windInput) {
                    windInput.value = Math.round(w.wind_speed);
                }
                
                const precipInput = document.querySelector('input[name="precip"]');
                if (precipInput) {
                    precipInput.value = w.precipitation.toFixed(1);
                }
                
                const visInput = document.querySelector('input[name="vis"]');
                if (visInput) {
                    visInput.value = w.visibility.toFixed(1);
                }
                
                const humInput = document.querySelector('input[name="hum"]');
                if (humInput) {
                    humInput.value = w.humidity;
                }
                
                const presInput = document.querySelector('input[name="pres"]');
                if (presInput) {
                    presInput.value = w.pressure.toFixed(0);
                }
                
                // Update display values
                document.getElementById('temp-val').textContent = Math.round(w.temperature) + '°C';
                document.getElementById('wind-val').textContent = Math.round(w.wind_speed) + ' km/h';
                document.getElementById('vis-val').textContent = w.visibility.toFixed(1) + ' km';
                document.getElementById('hum-val').textContent = w.humidity + '%';
                
                if (statusSpan) {
                    statusSpan.textContent = `✅ Weather: ${Math.round(w.temperature)}°C, ${Math.round(w.wind_speed)} km/h`;
                    statusSpan.style.color = '#4ade80';
                }
            } else {
                if (statusSpan) {
                    statusSpan.textContent = '❌ Failed to load weather';
                    statusSpan.style.color = '#f87171';
                }
            }
        } catch (error) {
            console.error('Weather error:', error);
            if (statusSpan) {
                statusSpan.textContent = '❌ Error: ' + error.message;
                statusSpan.style.color = '#f87171';
            }
        }
    }

    async handlePrediction(e) {
        e.preventDefault();

        const form = document.getElementById('predictionForm');
        const formData = new FormData(form);

        // Show loading
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('result').classList.add('hidden');
        document.getElementById('infoCard').classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.currentPrediction = data;
                this.displayPrediction(data);
                this.savePredictionToHistory(data);
            } else {
                this.showError(data.error || 'Prediction failed');
            }
        } catch (error) {
            this.showError('Error: ' + error.message);
            console.error('Prediction error:', error);
        } finally {
            document.getElementById('loading').classList.add('hidden');
        }
    }

    displayPrediction(data) {
        const resultDiv = document.getElementById('result');
        const prob = (data.probability * 100).toFixed(2);

        // Risk level color
        let riskClass = 'low';
        if (data.risk_level === 'CRITICAL') riskClass = 'critical';
        else if (data.risk_level === 'HIGH') riskClass = 'high';
        else if (data.risk_level === 'MODERATE') riskClass = 'moderate';

        // Update risk gauge
        document.getElementById('riskLevel').textContent = data.risk_level;
        document.getElementById('riskLevel').className = 'risk-tag ' + riskClass;
        document.getElementById('probText').textContent = prob + '%';
        document.getElementById('probFill').style.width = prob + '%';
        document.getElementById('probFill').style.backgroundColor = 
            data.prediction === 'DELAYED' ? 'var(--neon-pink)' : 'var(--neon-green)';

        // Update prediction text
        const predText = document.getElementById('predictionText');
        predText.textContent = data.prediction;
        predText.className = data.prediction === 'DELAYED' ? 'status-msg delayed' : 'status-msg on-time';

        // Display XAI explanations
        const xaiList = document.getElementById('xaiList');
        xaiList.innerHTML = '';

        data.explanations.forEach((exp, idx) => {
            const div = document.createElement('div');
            div.className = 'xai-item';
            div.innerHTML = `
                <div class="xai-header">
                    <span class="xai-rank">#${idx + 1}</span>
                    <span class="xai-feature">${exp.feature}</span>
                </div>
                <div class="xai-details">
                    <span class="xai-impact ${exp.impact === 'increased' ? 'xai-increase' : 'xai-decrease'}">
                        <i class="fas fa-arrow-${exp.impact === 'increased' ? 'up' : 'down'}"></i>
                        ${exp.impact.toUpperCase()}
                    </span>
                    <span class="xai-value">Value: ${exp.value}</span>
                    <span class="xai-importance">Impact: ${exp.importance}</span>
                </div>
            `;
            xaiList.appendChild(div);
        });

        resultDiv.classList.remove('hidden');
    }

    showError(message) {
        alert('⚠️ Error: ' + message);
        document.getElementById('loading').classList.add('hidden');
    }

    // ─── History Management ───────────────────────────────
    savePredictionToHistory(prediction) {
        const entry = {
            timestamp: new Date().toLocaleString(),
            prediction: prediction.prediction,
            probability: prediction.probability,
            riskLevel: prediction.risk_level,
            formData: new FormData(document.getElementById('predictionForm'))
        };

        this.predictionHistory.unshift(entry);
        if (this.predictionHistory.length > 20) {
            this.predictionHistory.pop();
        }

        localStorage.setItem('predictionHistory', JSON.stringify(this.predictionHistory));
        this.loadPredictionHistory();
    }

    loadPredictionHistory() {
        const historyDiv = document.getElementById('historyList');
        if (!historyDiv) return;

        historyDiv.innerHTML = '';

        if (this.predictionHistory.length === 0) {
            historyDiv.innerHTML = '<p style="color: var(--text-secondary);">No predictions yet</p>';
            return;
        }

        this.predictionHistory.slice(0, 10).forEach((entry, idx) => {
            const riskColor = {
                'CRITICAL': 'var(--neon-pink)',
                'HIGH': 'var(--neon-yellow)',
                'MODERATE': 'var(--accent)',
                'LOW': 'var(--neon-green)'
            }[entry.riskLevel];

            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = `
                <div class="history-time">${entry.timestamp}</div>
                <div class="history-result">
                    <span class="status-badge ${entry.prediction.toLowerCase()}">${entry.prediction}</span>
                    <span style="color: ${riskColor}; font-weight: 600;">${entry.riskLevel}</span>
                    <span style="color: var(--text-secondary);">${(entry.probability * 100).toFixed(1)}%</span>
                </div>
            `;
            historyDiv.appendChild(div);
        });
    }

    clearHistory() {
        if (confirm('Clear prediction history? This cannot be undone.')) {
            this.predictionHistory = [];
            localStorage.removeItem('predictionHistory');
            this.loadPredictionHistory();
        }
    }

    // ─── Export & Reporting ───────────────────────────────
    exportReport() {
        if (!this.currentPrediction) {
            alert('No prediction to export. Make a prediction first!');
            return;
        }

        const pred = this.currentPrediction;
        const date = new Date().toLocaleString();

        let csv = 'AeroPredict AI — Flight Delay Prediction Report\n';
        csv += `Generated: ${date}\n\n`;
        csv += 'PREDICTION RESULT\n';
        csv += `Prediction,${pred.prediction}\n`;
        csv += `Probability,${(pred.probability * 100).toFixed(2)}%\n`;
        csv += `Risk Level,${pred.risk_level}\n\n`;

        csv += 'TOP CONTRIBUTING FEATURES (SHAP Explanations)\n';
        csv += 'Feature,Impact Direction,Importance,Input Value\n';
        pred.explanations.forEach(exp => {
            csv += `${exp.feature},${exp.impact},${exp.importance},${exp.value}\n`;
        });

        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `prediction_${Date.now()}.csv`;
        a.click();
    }

    // ─── Model Results Loading ────────────────────────────
    async loadModelResults() {
        try {
            const response = await fetch('/api/model-results');
            const data = await response.json();

            const grid = document.getElementById('modelCardsGrid');
            if (!grid) return;

            data.results.forEach(model => {
                const card = document.createElement('div');
                card.className = 'model-card glass-card animate-in';
                card.innerHTML = `
                    <div class="model-header">
                        <h4>${model.Model}</h4>
                        <span class="model-badge ${model.Accuracy >= 0.95 ? 'excellent' : 'good'}">
                            ${(model.Accuracy * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div class="model-metrics">
                        <div class="metric">
                            <span class="metric-label">Precision</span>
                            <span class="metric-value">${(model.Precision * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Recall</span>
                            <span class="metric-value">${(model.Recall * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">F1-Score</span>
                            <span class="metric-value">${(model['F1-Score'] * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">AUC-ROC</span>
                            <span class="metric-value">${(model['AUC-ROC'] * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `;
                grid.appendChild(card);
            });

            // Load results table
            const table = document.getElementById('resultsTable');
            if (table && table.querySelector('thead')) {
                const tbody = table.querySelector('tbody') || document.createElement('tbody');
                data.results.forEach(model => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${model.Model}</td>
                        <td>${(model.Accuracy * 100).toFixed(2)}%</td>
                        <td>${(model.Precision * 100).toFixed(2)}%</td>
                        <td>${(model.Recall * 100).toFixed(2)}%</td>
                        <td>${(model['F1-Score'] * 100).toFixed(2)}%</td>
                        <td>${(model['AUC-ROC'] * 100).toFixed(2)}%</td>
                    `;
                });
                if (!table.querySelector('tbody')) {
                    table.appendChild(tbody);
                }
            }
        } catch (error) {
            console.error('Error loading model results:', error);
        }
    }

    // ─── Dark/Light Mode ──────────────────────────────────
    toggleMode() {
        const html = document.documentElement;
        const isDark = html.getAttribute('data-mode') === 'dark';
        const newMode = isDark ? 'light' : 'dark';
        
        html.setAttribute('data-mode', newMode);
        localStorage.setItem('appMode', newMode);
        
        // Update toggle button
        const btn = document.getElementById('modeToggle');
        if (btn) {
            btn.innerHTML = isDark ? 
                '<i class="fas fa-moon"></i>' :
                '<i class="fas fa-sun"></i>';
        }
    }
}

// Initialize app on load
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AeroPredictApp();
});
