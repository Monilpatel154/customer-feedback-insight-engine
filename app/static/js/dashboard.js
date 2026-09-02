/**
 * InsightPulse Dashboard Logic
 * Interactive Chart.js integration, live inference AJAX, and batch file processing
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSingleAnalyzer();
    initBatchUploader();
    fetchBenchmarkMetrics();
});

// Navigation & Tab Switching
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navItems.forEach(n => n.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

// Single Review Real-time Analyzer
function initSingleAnalyzer() {
    const textarea = document.getElementById('singleReviewInput');
    const charCount = document.getElementById('charCount');
    const btnAnalyze = document.getElementById('btnAnalyzeSingle');
    const emptyState = document.getElementById('singleEmptyState');
    const resultContent = document.getElementById('singleResultContent');
    const presetPills = document.querySelectorAll('.preset-pill');

    const presets = {
        delivery: "The parcel was delivered two days late and the outer box was completely crushed. The courier driver was very dismissive.",
        quality: "The build quality is exceptional, feels incredibly sturdy and premium. Battery easily lasts through a full two days of heavy use.",
        mixed: "Customer service was very polite and helpful when I called, but the companion mobile app crashes repeatedly during Bluetooth pairing.",
        support: "Worst customer care imaginable. Kept me on hold for forty minutes and the representative refused to process my warranty replacement."
    };

    textarea.addEventListener('input', () => {
        charCount.textContent = textarea.value.length;
    });

    presetPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const key = pill.getAttribute('data-preset');
            if (presets[key]) {
                textarea.value = presets[key];
                charCount.textContent = textarea.value.length;
                analyzeSingleReview(textarea.value);
            }
        });
    });

    btnAnalyze.addEventListener('click', () => {
        const text = textarea.value.trim();
        if (!text) {
            alert('Please enter a review text or choose a preset.');
            return;
        }
        analyzeSingleReview(text);
    });

    async function analyzeSingleReview(text) {
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Render Results
            emptyState.classList.add('hidden');
            resultContent.classList.remove('hidden');

            const badge = document.getElementById('sentimentBadge');
            badge.className = `sentiment-badge ${data.sentiment}`;
            badge.textContent = data.sentiment;

            const confPct = Math.round(data.confidence * 100);
            document.getElementById('confidenceValue').textContent = `${confPct}%`;
            document.getElementById('confidenceBar').style.width = `${confPct}%`;

            // Probabilities
            const probs = data.probabilities || {};
            document.getElementById('probPos').textContent = `${Math.round((probs.Positive || 0) * 100)}%`;
            document.getElementById('probNeu').textContent = `${Math.round((probs.Neutral || 0) * 100)}%`;
            document.getElementById('probNeg').textContent = `${Math.round((probs.Negative || 0) * 100)}%`;

            // Aspects
            const aspectsList = document.getElementById('aspectsList');
            aspectsList.innerHTML = '';
            const aspects = data.aspects || {};
            const aspectKeys = Object.keys(aspects);

            if (aspectKeys.length === 0) {
                aspectsList.innerHTML = '<div class="aspect-item"><span class="text-muted">No specific domain aspects detected (General Sentiment).</span></div>';
            } else {
                aspectKeys.forEach(asp => {
                    const info = aspects[asp];
                    const item = document.createElement('div');
                    item.className = 'aspect-item';
                    item.innerHTML = `
                        <div class="aspect-left">
                            <i class="fa-solid fa-tag"></i>
                            <div>
                                <span class="aspect-name">${asp}</span>
                                <div class="aspect-keywords">Keywords: ${info.keywords.join(', ')}</div>
                            </div>
                        </div>
                        <span class="aspect-badge ${info.sentiment}">${info.sentiment}</span>
                    `;
                    aspectsList.appendChild(item);
                });
            }

            // Dominant Topic
            const topicLabel = document.getElementById('topicLabel');
            if (data.topic) {
                topicLabel.textContent = `${data.topic.topic_label} (${Math.round(data.topic.confidence * 100)}% match)`;
            } else {
                topicLabel.textContent = "General Feedback";
            }

            // Cleaned Tokens
            document.getElementById('cleanedTokensDisplay').textContent = data.cleaned_text || "(empty)";

        } catch (err) {
            console.error(err);
            alert('Failed to connect to sentiment inference API.');
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Sentiment & Insight Analysis';
        }
    }
}

// Global chart references for destruction/updating
let sentimentChartInstance = null;
let aspectChartInstance = null;
let currentBatchData = null;

// Batch Uploader & BI Dashboard
function initBatchUploader() {
    const dropzone = document.getElementById('csvDropzone');
    const fileInput = document.getElementById('csvFileInput');
    const uploadStatus = document.getElementById('uploadStatus');
    const batchResults = document.getElementById('batchResultsSection');
    const btnDownloadSample = document.getElementById('btnDownloadSample');
    const btnExportCSV = document.getElementById('btnExportCSV');

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--color-primary)';
            dropzone.style.background = 'rgba(99, 102, 241, 0.05)';
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--border-glass)';
            dropzone.style.background = 'transparent';
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    btnDownloadSample.addEventListener('click', () => {
        window.location.href = '/api/download_sample';
    });

    btnExportCSV.addEventListener('click', () => {
        if (!currentBatchData) return;
        exportBatchToCSV(currentBatchData);
    });

    async function handleFile(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a valid .csv file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        dropzone.classList.add('hidden');
        uploadStatus.classList.remove('hidden');

        try {
            const res = await fetch('/api/batch_upload', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();

            if (result.error) {
                alert('Upload failed: ' + result.error);
                return;
            }

            renderBatchAnalytics(result);
            batchResults.classList.remove('hidden');
        } catch (err) {
            console.error(err);
            alert('Batch processing failed. Check server logs.');
        } finally {
            uploadStatus.classList.add('hidden');
            dropzone.classList.remove('hidden');
        }
    }
}

function renderBatchAnalytics(data) {
    currentBatchData = data.reviews;
    const analytics = data.analytics;

    // 1. KPI Cards
    document.getElementById('kpiTotal').textContent = analytics.total_samples.toLocaleString();
    const pPct = analytics.sentiment_percentages.Positive || 0;
    const nPct = analytics.sentiment_percentages.Negative || 0;
    document.getElementById('kpiPos').textContent = `${pPct}%`;
    document.getElementById('kpiNeg').textContent = `${nPct}%`;
    document.getElementById('kpiConf').textContent = `${Math.round(analytics.avg_confidence * 100)}%`;

    // 2. Sentiment Donut Chart
    const sCounts = analytics.sentiment_counts;
    const ctxSentiment = document.getElementById('sentimentChart').getContext('2d');
    if (sentimentChartInstance) sentimentChartInstance.destroy();

    sentimentChartInstance = new Chart(ctxSentiment, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [sCounts.Positive || 0, sCounts.Neutral || 0, sCounts.Negative || 0],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#9ca3af', font: { family: 'Inter' } } }
            },
            cutout: '70%'
        }
    });

    // 3. Aspect Breakdown Stacked Bar Chart
    const aspectLabels = Object.keys(analytics.aspect_counts);
    const posAspectData = aspectLabels.map(a => analytics.aspect_positive[a] || 0);
    const negAspectData = aspectLabels.map(a => analytics.aspect_negative[a] || 0);

    const ctxAspect = document.getElementById('aspectChart').getContext('2d');
    if (aspectChartInstance) aspectChartInstance.destroy();

    aspectChartInstance = new Chart(ctxAspect, {
        type: 'bar',
        data: {
            labels: aspectLabels,
            datasets: [
                {
                    label: 'Positive Mentions',
                    data: posAspectData,
                    backgroundColor: '#10b981',
                    borderRadius: 4
                },
                {
                    label: 'Negative Mentions',
                    data: negAspectData,
                    backgroundColor: '#ef4444',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
            },
            plugins: {
                legend: { position: 'bottom', labels: { color: '#9ca3af' } }
            }
        }
    });

    // 4. Actionable Business Recommendations List
    const recsList = document.getElementById('recommendationsList');
    recsList.innerHTML = '';
    const recs = analytics.recommendations || [];

    recs.forEach(r => {
        const item = document.createElement('div');
        item.className = `rec-item ${r.severity}`;
        item.innerHTML = `
            <div class="rec-header">
                <span class="rec-category">${r.category}</span>
                <span class="rec-severity ${r.severity}">${r.severity}</span>
            </div>
            <div class="rec-finding"><strong>Finding:</strong> ${r.finding}</div>
            <div class="rec-action"><strong>Recommended Action:</strong> ${r.action}</div>
            <div class="rec-impact"><i class="fa-solid fa-chart-line"></i> Expected Business Impact: ${r.impact}</div>
        `;
        recsList.appendChild(item);
    });

    // 5. Feedback Table
    const tbody = document.querySelector('#feedbackTable tbody');
    tbody.innerHTML = '';
    const displayReviews = currentBatchData.slice(0, 100); // show first 100

    displayReviews.forEach((rev, idx) => {
        const tr = document.createElement('tr');
        const aspectsText = (rev.detected_aspects || []).join(', ') || 'General';
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td style="max-width: 380px;">${rev.review_text}</td>
            <td><span class="sentiment-badge ${rev.predicted_sentiment}" style="font-size: 0.75rem; padding: 0.2rem 0.6rem;">${rev.predicted_sentiment}</span></td>
            <td>${Math.round((rev.confidence || 0) * 100)}%</td>
            <td><small>${aspectsText}</small></td>
            <td><small>${rev.dominant_topic || 'General'}</small></td>
        `;
        tbody.appendChild(tr);
    });
}

function exportBatchToCSV(data) {
    if (!data || data.length === 0) return;
    const headers = ["review_text", "predicted_sentiment", "confidence", "detected_aspects", "dominant_topic"];
    const rows = [headers.join(',')];

    data.forEach(item => {
        const text = `"${(item.review_text || '').replace(/"/g, '""')}"`;
        const aspects = `"${(item.detected_aspects || []).join('; ')}"`;
        const topic = `"${(item.dominant_topic || '').replace(/"/g, '""')}"`;
        rows.push([text, item.predicted_sentiment, item.confidence, aspects, topic].join(','));
    });

    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment_insight_report_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// Fetch Benchmark Metrics
async function fetchBenchmarkMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        if (!data || !data.comparison) return;

        const bestName = data.best_model_name;
        document.getElementById('sidebarModelStatus').innerHTML = `
            <i class="fa-solid fa-circle-check"></i> ${bestName} Active
        `;

        const tbody = document.querySelector('#benchmarkTable tbody');
        tbody.innerHTML = '';

        for (const [name, metrics] of Object.entries(data.comparison)) {
            const tr = document.createElement('tr');
            const isWinner = name === bestName;
            if (isWinner) tr.className = 'winner-row';

            tr.innerHTML = `
                <td>
                    ${name}
                    ${isWinner ? '<span class="winner-badge"><i class="fa-solid fa-crown"></i> Best Model</span>' : ''}
                </td>
                <td>${metrics.cv_f1_macro_mean.toFixed(3)} &plusmn; ${metrics.cv_f1_macro_std.toFixed(3)}</td>
                <td><strong>${(metrics.test_accuracy * 100).toFixed(2)}%</strong></td>
                <td><strong>${(metrics.test_f1_macro * 100).toFixed(2)}%</strong></td>
                <td>${(metrics.test_precision_macro * 100).toFixed(2)}%</td>
                <td>${(metrics.test_recall_macro * 100).toFixed(2)}%</td>
                <td>${metrics.inference_latency_ms} ms</td>
            `;
            tbody.appendChild(tr);
        }

        // LDA Topics in architecture tab
        const ldaContainer = document.getElementById('ldaTopicsDisplay');
        if (data.lda_topics && data.lda_topics.length > 0) {
            ldaContainer.innerHTML = '';
            data.lda_topics.forEach(t => {
                const div = document.createElement('div');
                div.className = 'lda-topic-card';
                div.innerHTML = `
                    <strong>${t.label}</strong>
                    <span>Top Weighted Terms: ${t.keywords.join(', ')}</span>
                `;
                ldaContainer.appendChild(div);
            });
        }

    } catch (err) {
        console.warn('Metrics not yet available. Train models via src/train.py first.');
    }
}
