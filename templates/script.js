// Global variable to store selected region
let selectedRegion = null;

function handleRegionClick(region) {
    // Update selected region
    selectedRegion = region;

    // Update UI to show selected region
    updateSelectedRegionUI(region);

    // Fetch and analyze news for the selected region
    fetchNewsAnalysis(region);

    // Fetch drought analysis data
    fetchDroughtAnalysis(region);
}

function updateSelectedRegionUI(region) {
    // Update the UI to show the selected region
    const regionInfo = document.getElementById('region-info');
    if (regionInfo) {
        regionInfo.textContent = `Selected Region: ${region}`;
    }
}

function getDroughtStatus(value) {
    if (value >= 0.5) {
        return { class: 'good', text: 'Good' };
    } else if (value >= 0.2) {
        return { class: 'moderate', text: 'Moderate' };
    } else {
        return { class: 'poor', text: 'Poor' };
    }
}

function updateDroughtAnalysisUI(data) {
    // Update NDVI value and status
    const ndviValue = document.getElementById('last-ndvi');
    const ndviStatus = document.getElementById('ndvi-status');
    if (ndviValue && ndviStatus) {
        ndviValue.textContent = data.ndvi.toFixed(3);
        const status = getDroughtStatus(data.ndvi);
        ndviStatus.className = `drought-status ${status.class}`;
        ndviStatus.textContent = status.text;
    }

    // Update LSTM score and status
    const lstmScore = document.getElementById('lstm-score');
    const lstmStatus = document.getElementById('lstm-status');
    if (lstmScore && lstmStatus) {
        lstmScore.textContent = data.lstm_score.toFixed(3);
        const status = getDroughtStatus(data.lstm_score);
        lstmStatus.className = `drought-status ${status.class}`;
        lstmStatus.textContent = status.text;
    }

    // Update drought chart
    const droughtChart = document.getElementById('drought-chart');
    if (droughtChart && data.drought_plot) {
        droughtChart.src = `data:image/png;base64,${data.drought_plot}`;
    }
}

function fetchDroughtAnalysis(region) {
    fetch('/analyze_drought', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ region: region })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                showError(data.error);
                return;
            }
            updateDroughtAnalysisUI(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to fetch drought analysis. Please try again.');
        });
}

function fetchNewsAnalysis(region) {
    // Show loading state
    showLoadingState();

    fetch('/analyze_news', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ region: region })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                showError(data.error);
                return;
            }

            // Update UI with news analysis results
            updateNewsAnalysisUI(data.analysis);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to fetch news analysis. Please try again.');
        });
}

function showLoadingState() {
    const sections = ['analysis-overview', 'food-crisis-analysis', 'climate-crisis-analysis', 'articles-list'];
    sections.forEach(sectionId => {
        const element = document.getElementById(sectionId);
        if (element) {
            element.innerHTML = '<div class="loading">Loading analysis results...</div>';
        }
    });
}

function showError(message) {
    const newsAnalysis = document.getElementById('news-analysis');
    if (newsAnalysis) {
        newsAnalysis.innerHTML = `<div class="error">${message}</div>`;
    }
}

function updateNewsAnalysisUI(analysis) {
    // Update overview section
    updateOverviewSection(analysis);

    // Update food crisis section
    updateFoodCrisisSection(analysis.food_crisis);

    // Update climate crisis section
    updateClimateCrisisSection(analysis.climate_crisis);

    // Update articles list
    updateArticlesList(analysis.articles);
}

function updateOverviewSection(analysis) {
    const overviewDiv = document.getElementById('analysis-overview');
    if (!overviewDiv) return;

    const stats = [
        { label: 'Total Articles', value: analysis.total_articles },
        { label: 'Food Crisis Articles', value: `${analysis.food_crisis.count} (${analysis.food_crisis.percent}%)` },
        { label: 'Climate Crisis Articles', value: `${analysis.climate_crisis.count} (${analysis.climate_crisis.percent}%)` }
    ];

    overviewDiv.innerHTML = `
        <div class="stats-grid">
            ${stats.map(stat => `
                <div class="stat-card">
                    <h4>${stat.label}</h4>
                    <div class="value">${stat.value}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function updateFoodCrisisSection(foodCrisis) {
    const foodDiv = document.getElementById('food-crisis-analysis');
    if (!foodDiv) return;

    const stats = [
        { label: 'Severity', value: foodCrisis.severity },
        { label: 'Tone', value: foodCrisis.tone },
        { label: 'Average Sentiment', value: foodCrisis.average_sentiment.toFixed(3) }
    ];

    foodDiv.innerHTML = `
        <div class="stats-grid">
            ${stats.map(stat => `
                <div class="stat-card">
                    <h4>${stat.label}</h4>
                    <div class="value">${stat.value}</div>
                </div>
            `).join('')}
        </div>
        <h4>Category Distribution</h4>
        <div class="stats-grid">
            ${Object.entries(foodCrisis.category_counts).map(([category, count]) => `
                <div class="stat-card">
                    <h4>${category.replace('_', ' ').toUpperCase()}</h4>
                    <div class="value">${count}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function updateClimateCrisisSection(climateCrisis) {
    const climateDiv = document.getElementById('climate-crisis-analysis');
    if (!climateDiv) return;

    const stats = [
        { label: 'Severity', value: climateCrisis.severity },
        { label: 'Tone', value: climateCrisis.tone },
        { label: 'Average Sentiment', value: climateCrisis.average_sentiment.toFixed(3) }
    ];

    climateDiv.innerHTML = `
        <div class="stats-grid">
            ${stats.map(stat => `
                <div class="stat-card">
                    <h4>${stat.label}</h4>
                    <div class="value">${stat.value}</div>
                </div>
            `).join('')}
        </div>
        <h4>Category Distribution</h4>
        <div class="stats-grid">
            ${Object.entries(climateCrisis.category_counts).map(([category, count]) => `
                <div class="stat-card">
                    <h4>${category.replace('_', ' ').toUpperCase()}</h4>
                    <div class="value">${count}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function updateArticlesList(articles) {
    const articlesDiv = document.getElementById('articles-list');
    if (!articlesDiv) return;

    if (!articles || articles.length === 0) {
        articlesDiv.innerHTML = '<p>No articles found for the selected region.</p>';
        return;
    }

    articlesDiv.innerHTML = articles.map(article => `
        <div class="article-card">
            <h4><a href="${article.url}" target="_blank">${article.title}</a></h4>
            <div class="meta">
                Published: ${new Date(article.publishedAt).toLocaleDateString()}
                | Sentiment: ${article.sentiment.toFixed(3)}
            </div>
            <div class="description">${article.description}</div>
            <div class="categories">
                ${article.categories.map(category => `
                    <span class="category-tag">${category.replace('_', ' ')}</span>
                `).join('')}
            </div>
        </div>
    `).join('');
}

// Add event listener for region selection
document.addEventListener('DOMContentLoaded', function () {
    // Example: If you have a map or list of regions, add click handlers here
    // For testing, you can manually trigger handleRegionClick with a region name
    // handleRegionClick('Nigeria');
}); 