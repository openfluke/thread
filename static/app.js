// T.H.R.E.A.D. - Frontend JavaScript

// ═══════════════════════════════════════════════════════════════════════════════
// Sidebar Toggle
// ═══════════════════════════════════════════════════════════════════════════════

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
}

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.querySelector('.sidebar-toggle');

    if (window.innerWidth <= 1024 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// Chart.js Defaults
// ═══════════════════════════════════════════════════════════════════════════════

if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#334155';
    Chart.defaults.font.family = "'Inter', sans-serif";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function formatPercent(num) {
    return num.toFixed(1) + '%';
}

function getColorForScore(score) {
    if (score >= 700) return '#10b981';
    if (score >= 500) return '#3b82f6';
    if (score >= 300) return '#f59e0b';
    return '#64748b';
}

// ═══════════════════════════════════════════════════════════════════════════════
// API Client
// ═══════════════════════════════════════════════════════════════════════════════

const api = {
    async getExperiments(params = {}) {
        const query = new URLSearchParams(params).toString();
        const res = await fetch(`/api/experiments${query ? '?' + query : ''}`);
        return res.json();
    },

    async getExperiment(id) {
        const res = await fetch(`/api/experiments/${id}`);
        return res.json();
    },

    async getStats() {
        const res = await fetch('/api/stats');
        return res.json();
    },

    async getLeaderboard() {
        const res = await fetch('/api/leaderboard');
        return res.json();
    },

    async getDomains() {
        const res = await fetch('/api/domains');
        return res.json();
    },

    async getLevels() {
        const res = await fetch('/api/levels');
        return res.json();
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Chart Builders
// ═══════════════════════════════════════════════════════════════════════════════

const charts = {
    // Line chart for accuracy over time
    accuracyTimeline(canvasId, windows, color = '#3b82f6') {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !windows || windows.length === 0) return null;

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: windows.map(w => (w.timeMs / 1000) + 's'),
                datasets: [{
                    label: 'Accuracy %',
                    data: windows.map(w => w.accuracy),
                    borderColor: color,
                    backgroundColor: color + '22',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, max: 100 },
                    x: { ticks: { maxTicksLimit: 12 } }
                }
            }
        });
    },

    // Bar chart for score comparison
    scoreComparison(canvasId, experiments) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !experiments || experiments.length === 0) return null;

        const domainColors = {
            classification: '#3b82f6',
            regression: '#10b981',
            generation: '#8b5cf6',
            sequence: '#f59e0b',
            reinforcement: '#ef4444',
            anomaly: '#ec4899',
            clustering: '#14b8a6',
            embedding: '#6366f1',
            optimization: '#f97316',
            metalearning: '#a855f7'
        };

        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: experiments.map(e => e.name),
                datasets: [{
                    label: 'Score',
                    data: experiments.map(e => e.results.score),
                    backgroundColor: experiments.map(e => domainColors[e.domain] || '#64748b'),
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true },
                    x: { ticks: { maxRotation: 45 } }
                }
            }
        });
    },

    // Radar chart for multi-metric comparison
    metricsRadar(canvasId, experiments) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !experiments || experiments.length === 0) return null;

        const colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444'];

        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Stability', 'Consistency', 'Throughput', 'Score'],
                datasets: experiments.slice(0, 5).map((e, i) => ({
                    label: e.name,
                    data: [
                        e.results.avgAccuracy,
                        e.results.stability,
                        e.results.consistency,
                        Math.min(100, Math.log10(e.results.throughputPerSec) * 25),
                        Math.min(100, e.results.score / 10)
                    ],
                    borderColor: colors[i],
                    backgroundColor: colors[i] + '33',
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: { beginAtZero: true, max: 100 }
                }
            }
        });
    },

    // Scatter plot for throughput vs accuracy
    throughputVsAccuracy(canvasId, experiments) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !experiments || experiments.length === 0) return null;

        const domainColors = {
            classification: '#3b82f688',
            regression: '#10b98188',
            generation: '#8b5cf688',
            sequence: '#f59e0b88',
            reinforcement: '#ef444488'
        };

        return new Chart(ctx, {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'Experiments',
                    data: experiments.map(e => ({
                        x: e.results.throughputPerSec,
                        y: e.results.avgAccuracy,
                        r: Math.sqrt(e.results.score) / 2,
                        name: e.name,
                        domain: e.domain
                    })),
                    backgroundColor: experiments.map(e => domainColors[e.domain] || '#64748b88')
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const d = ctx.raw;
                                return `${d.name}: ${d.y.toFixed(1)}% @ ${formatNumber(d.x)}/s`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Throughput (tokens/sec)' }
                    },
                    y: {
                        title: { display: true, text: 'Accuracy %' },
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    },

    // Doughnut chart for score breakdown
    scoreBreakdown(canvasId, experiment) {
        const ctx = document.getElementById(canvasId);
        if (!ctx || !experiment) return null;

        const r = experiment.results;

        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Throughput Factor', 'Stability Factor', 'Consistency Factor'],
                datasets: [{
                    data: [
                        Math.log10(r.throughputPerSec) * 10,
                        r.stability,
                        r.consistency
                    ],
                    backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Animation Helpers
// ═══════════════════════════════════════════════════════════════════════════════

function animateValue(element, start, end, duration = 1000) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out-cubic)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeOut;

        element.textContent = Math.round(current);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Page Initializations
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    // Animate stat values on home page
    document.querySelectorAll('.stat-value').forEach(el => {
        const value = parseInt(el.textContent);
        if (!isNaN(value) && value > 0) {
            animateValue(el, 0, value);
        }
    });

    // Add active state to nav links
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    console.log('🌊 T.H.R.E.A.D. initialized');
});

// Export for use in templates
window.THREAD = {
    api,
    charts,
    formatNumber,
    formatPercent,
    getColorForScore,
    animateValue
};
