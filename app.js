// Cyberbullying Detection Patterns
const cyberbullyingPatterns = {
    offensiveWords: [
        "stupid", "idiot", "loser", "freak", "weirdo", "ugly", "fat", "worthless", 
        "pathetic", "disgusting", "hate", "kill", "die", "death", "murder", 
        "hurt", "pain", "suffer", "destroy", "ruin", "trash", "garbage", "waste",
        "dumb", "retarded", "moron", "fool", "failure", "useless", "hopeless",
        "bitch", "bastard", "asshole", "fuck", "shit", "damn", "crap",
        "slut", "whore", "fag", "gay", "retard", "spastic", "psycho"
    ],
    threats: [
        "gonna kill", "will hurt", "going to get", "watch your back", "you're dead",
        "i'll find you", "gonna beat", "will destroy", "make you pay", "gonna hurt",
        "kill yourself", "end your life", "should die", "wish you were dead",
        "gonna beat you up", "i'll hurt you", "you better watch out", "gonna get you",
        "i'll make you", "you'll regret", "i'll show you", "gonna mess you up"
    ],
    harassment: [
        "nobody likes you", "everyone hates you", "no one cares", "you don't belong",
        "get lost", "go away", "shut up", "leave us alone", "you're annoying",
        "can't stand you", "you suck", "you're terrible", "worst person",
        "you're a joke", "embarrassing", "pathetic loser", "complete failure",
        "you're nothing", "waste of space", "shouldn't exist", "go kill yourself"
    ],
    exclusion: [
        "don't talk to", "ignore them", "not invited", "stay away", "get out",
        "don't want you", "not welcome", "leave us alone", "go somewhere else",
        "you don't fit in", "not one of us", "don't belong here", "go home",
        "we don't like", "stay out", "not wanted", "get away from us"
    ],
    discriminatory: [
        "because you're", "your kind", "people like you", "your race", "your religion",
        "typical", "all of you", "your people", "because of your", "you people",
        "your type", "where you come from", "go back to", "not our kind",
        "dirty", "savage", "uncivilized", "primitive", "inferior"
    ],
    intensifiers: [
        "really", "so", "very", "extremely", "totally", "completely", "absolutely",
        "fucking", "damn", "hell", "bloody", "freaking", "super", "mega"
    ],
    personalTargeting: [
        "you are", "you're", "your", "yourself", "you look", "you sound", "you act",
        "you smell", "you think", "you feel", "you make me", "you always"
    ],
    sexualHarassment: [
        "send pics", "show me your", "what are you wearing", "sexy", "hot body",
        "take off", "i want to", "come over", "meet me", "private chat",
        "nude", "naked", "strip", "touch yourself", "make me"
    ]
};

// Categories for classification
const categories = {
    THREATS: "Threats/Violence",
    HARASSMENT: "Harassment", 
    DISCRIMINATION: "Discrimination",
    EXCLUSION: "Exclusion",
    OFFENSIVE: "Offensive Language",
    SEXUAL: "Sexual Harassment"
};

// Application state
let analysisHistory = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
let statistics = JSON.parse(localStorage.getItem('statistics') || '{"total": 0, "positive": 0, "negative": 0}');

// DOM elements
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContent = document.getElementById('resultsContent');
const totalCount = document.getElementById('totalCount');
const positiveCount = document.getElementById('positiveCount');
const negativeCount = document.getElementById('negativeCount');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    updateStatistics();
    updateHistory();

    // Event listeners
    textInput.addEventListener('input', updateCharCount);
    textInput.addEventListener('input', toggleAnalyzeButton);
    clearBtn.addEventListener('click', clearText);
    analyzeBtn.addEventListener('click', analyzeText);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Initial state
    updateCharCount();
    toggleAnalyzeButton();
});

// Update character count
function updateCharCount() {
    const count = textInput.value.length;
    charCount.textContent = `${count} characters`;

    if (count > 1000) {
        charCount.style.color = 'var(--warning-color)';
    } else {
        charCount.style.color = 'var(--gray-500)';
    }
}

// Toggle analyze button state
function toggleAnalyzeButton() {
    const hasText = textInput.value.trim().length > 0;
    analyzeBtn.disabled = !hasText;
}

// Clear text input
function clearText() {
    textInput.value = '';
    updateCharCount();
    toggleAnalyzeButton();
    resultsSection.style.display = 'none';
    textInput.focus();
}

// Main analysis function
function analyzeText() {
    const text = textInput.value.trim();
    if (!text) return;

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Simulate processing delay for better UX
    setTimeout(() => {
        const result = classifyText(text);
        displayResults(result, text);
        saveAnalysis(text, result);
        updateStatistics();
        updateHistory();

        // Reset button
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Text';
    }, 800);
}

// Core classification algorithm
function classifyText(text) {
    const lowerText = text.toLowerCase();
    let score = 0;
    const detectedPatterns = [];
    const detectedCategories = new Set();
    const problematicWords = [];

    // Check for offensive words
    cyberbullyingPatterns.offensiveWords.forEach(word => {
        if (lowerText.includes(word.toLowerCase())) {
            score += 2;
            problematicWords.push(word);
            detectedCategories.add(categories.OFFENSIVE);
        }
    });

    // Check for threats (higher weight)
    cyberbullyingPatterns.threats.forEach(pattern => {
        if (lowerText.includes(pattern.toLowerCase())) {
            score += 5;
            detectedPatterns.push(pattern);
            detectedCategories.add(categories.THREATS);
        }
    });

    // Check for harassment patterns
    cyberbullyingPatterns.harassment.forEach(pattern => {
        if (lowerText.includes(pattern.toLowerCase())) {
            score += 4;
            detectedPatterns.push(pattern);
            detectedCategories.add(categories.HARASSMENT);
        }
    });

    // Check for exclusion language
    cyberbullyingPatterns.exclusion.forEach(pattern => {
        if (lowerText.includes(pattern.toLowerCase())) {
            score += 3;
            detectedPatterns.push(pattern);
            detectedCategories.add(categories.EXCLUSION);
        }
    });

    // Check for discriminatory language
    cyberbullyingPatterns.discriminatory.forEach(pattern => {
        if (lowerText.includes(pattern.toLowerCase())) {
            score += 4;
            detectedPatterns.push(pattern);
            detectedCategories.add(categories.DISCRIMINATION);
        }
    });

    // Check for sexual harassment
    cyberbullyingPatterns.sexualHarassment.forEach(pattern => {
        if (lowerText.includes(pattern.toLowerCase())) {
            score += 4;
            detectedPatterns.push(pattern);
            detectedCategories.add(categories.SEXUAL);
        }
    });

    // Check for personal targeting + offensive combinations
    const hasPersonalTargeting = cyberbullyingPatterns.personalTargeting.some(pattern => 
        lowerText.includes(pattern.toLowerCase())
    );

    const hasIntensifiers = cyberbullyingPatterns.intensifiers.some(pattern => 
        lowerText.includes(pattern.toLowerCase())
    );

    // Boost score for personal targeting
    if (hasPersonalTargeting && (problematicWords.length > 0 || detectedPatterns.length > 0)) {
        score += 2;
    }

    // Boost score for intensifiers
    if (hasIntensifiers && (problematicWords.length > 0 || detectedPatterns.length > 0)) {
        score += 1;
    }

    // Check for ALL CAPS (potential aggression)
    const capsWords = text.match(/[A-Z]{3,}/g);
    if (capsWords && capsWords.length > 2) {
        score += 1;
    }

    // Check for excessive punctuation (!!!, ???)
    const excessivePunctuation = text.match(/[!?]{3,}/g);
    if (excessivePunctuation && excessivePunctuation.length > 0) {
        score += 1;
    }

    // Classification threshold
    const threshold = 3;
    const isCyberbullying = score >= threshold;

    // Calculate confidence (0-100%)
    const maxPossibleScore = 20; // Reasonable maximum for confidence calculation
    const confidence = Math.min((score / maxPossibleScore) * 100, 100);

    return {
        isCyberbullying,
        score,
        confidence: Math.round(confidence),
        detectedPatterns: [...new Set([...detectedPatterns, ...problematicWords])],
        categories: Array.from(detectedCategories),
        severity: getSeverityLevel(score),
        hasPersonalTargeting,
        hasIntensifiers,
        hasCapsWords: capsWords && capsWords.length > 2,
        hasExcessivePunctuation: excessivePunctuation && excessivePunctuation.length > 0
    };
}

// Get severity level based on score
function getSeverityLevel(score) {
    if (score >= 10) return 'High';
    if (score >= 6) return 'Medium';
    if (score >= 3) return 'Low';
    return 'None';
}

// Display analysis results
function displayResults(result, originalText) {
    const { isCyberbullying, confidence, detectedPatterns, categories, severity } = result;

    let html = '';

    // Status indicator
    const statusClass = isCyberbullying ? 'result-status--danger' : 'result-status--safe';
    const statusIcon = isCyberbullying ? '⚠️' : '✅';
    const statusText = isCyberbullying ? 'CYBERBULLYING DETECTED' : 'NO CYBERBULLYING DETECTED';

    html += `
        <div class="result-status ${statusClass}">
            <span class="status-icon" style="font-size: 1.5rem;">${statusIcon}</span>
            <span class="status-text">${statusText}</span>
        </div>
    `;

    // Result details
    html += '<div class="result-details">';

    // Confidence score
    const confidenceLevel = confidence >= 70 ? 'high' : confidence >= 40 ? 'medium' : 'low';
    html += `
        <div class="detail-item">
            <h4>Confidence Score</h4>
            <p>${confidence}% confidence in classification</p>
            <div class="confidence-meter">
                <div class="confidence-bar">
                    <div class="confidence-fill confidence-fill--${confidenceLevel}" style="width: ${confidence}%"></div>
                </div>
            </div>
        </div>
    `;

    if (isCyberbullying) {
        // Severity level
        html += `
            <div class="detail-item">
                <h4>Severity Level</h4>
                <p><strong>${severity}</strong> - This content contains ${severity.toLowerCase()}-level cyberbullying indicators</p>
            </div>
        `;

        // Categories
        if (categories.length > 0) {
            html += `
                <div class="detail-item">
                    <h4>Categories Detected</h4>
                    <p>${categories.join(', ')}</p>
                </div>
            `;
        }

        // Detected patterns/words
        if (detectedPatterns.length > 0) {
            html += `
                <div class="detail-item">
                    <h4>Problematic Content Found</h4>
                    <p>The following potentially harmful content was detected:</p>
                    <div class="detected-words">
                        ${detectedPatterns.map(pattern => `<span class="word-tag">${pattern}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        // Recommendations
        html += `
            <div class="detail-item">
                <h4>Recommendations</h4>
                <p>This message contains language that could be harmful or offensive. Consider revising the content to be more respectful and constructive.</p>
            </div>
        `;
    } else {
        html += `
            <div class="detail-item">
                <h4>Analysis Summary</h4>
                <p>The analyzed text appears to be free of cyberbullying content. No harmful patterns or offensive language were detected.</p>
            </div>
        `;
    }

    html += '</div>';

    resultsContent.innerHTML = html;
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Save analysis to history
function saveAnalysis(text, result) {
    const analysis = {
        id: Date.now(),
        text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
        fullText: text,
        result: result.isCyberbullying,
        confidence: result.confidence,
        timestamp: new Date().toLocaleString(),
        categories: result.categories
    };

    analysisHistory.unshift(analysis);

    // Keep only last 20 analyses
    if (analysisHistory.length > 20) {
        analysisHistory = analysisHistory.slice(0, 20);
    }

    localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));

    // Update statistics
    statistics.total++;
    if (result.isCyberbullying) {
        statistics.positive++;
    } else {
        statistics.negative++;
    }

    localStorage.setItem('statistics', JSON.stringify(statistics));
}

// Update statistics display
function updateStatistics() {
    totalCount.textContent = statistics.total;
    positiveCount.textContent = statistics.positive;
    negativeCount.textContent = statistics.negative;
}

// Update history display
function updateHistory() {
    if (analysisHistory.length === 0) {
        historyList.innerHTML = '<p class="text-muted">No analyses performed yet.</p>';
        return;
    }

    const historyHtml = analysisHistory.map(analysis => {
        const resultClass = analysis.result ? 'history-result--danger' : 'history-result--safe';
        const resultText = analysis.result ? 'Cyberbullying' : 'Clean';

        return `
            <div class="history-item">
                <div class="history-text" title="${analysis.fullText}">${analysis.text}</div>
                <div class="history-result ${resultClass}">${resultText}</div>
                <div class="history-timestamp">${analysis.timestamp}</div>
            </div>
        `;
    }).join('');

    historyList.innerHTML = historyHtml;
}

// Clear analysis history
function clearHistory() {
    if (confirm('Are you sure you want to clear all analysis history?')) {
        analysisHistory = [];
        statistics = { total: 0, positive: 0, negative: 0 };

        localStorage.removeItem('analysisHistory');
        localStorage.removeItem('statistics');

        updateStatistics();
        updateHistory();
    }
}

// Export results functionality (bonus feature)
function exportResults() {
    if (analysisHistory.length === 0) {
        alert('No analysis history to export.');
        return;
    }

    const csvContent = "data:text/csv;charset=utf-8," + 
        "Timestamp,Text,Result,Confidence,Categories\n" +
        analysisHistory.map(analysis => 
            `"${analysis.timestamp}","${analysis.fullText.replace(/"/g, '""')}","${analysis.result ? 'Cyberbullying' : 'Clean'}","${analysis.confidence}%","${analysis.categories.join('; ')}"`
        ).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `cyberbullying_analysis_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to analyze
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        if (!analyzeBtn.disabled) {
            analyzeText();
        }
    }

    // Ctrl/Cmd + L to clear
    if ((event.ctrlKey || event.metaKey) && event.key === 'l') {
        event.preventDefault();
        clearText();
    }
});