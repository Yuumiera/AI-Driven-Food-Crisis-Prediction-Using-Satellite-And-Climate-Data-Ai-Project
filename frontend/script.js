document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('searchInput');
    const topSearchInput = document.getElementById('topSearchInput');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const topDropdownMenu = document.getElementById('topDropdownMenu');
    const selectedRegion = document.getElementById('selectedRegion');
    const mainContent = document.getElementById('mainContent');
    const topSearchBar = document.getElementById('topSearchBar');
    const resultsSection = document.getElementById('resultsSection');
    const statsContent = document.getElementById('statsContent');
    const heatmapImage = document.getElementById('heatmapImage');
    const classificationImage = document.getElementById('classificationImage');
    const ndviPlotImage = document.getElementById('ndviPlotImage');

    // Available regions list with friendly names
    const regions = [
        { id: 'sanliurfa', name: 'Şanlıurfa, Türkiye' },
        { id: 'punjab', name: 'Punjab, Hindistan' },
        { id: 'munich', name: 'Münih, Almanya' },
        { id: 'iowa', name: 'Iowa, ABD' },
        { id: 'kano', name: 'Kano, Nijerya' },
        { id: 'zacatecas', name: 'Zacatecas, Meksika' },
        { id: 'gauteng', name: 'Gauteng, Güney Afrika' },
        { id: 'addis_ababa', name: 'Addis Ababa, Etiyopya' },
        { id: 'yunnan', name: 'Yunnan, Çin' },
        { id: 'gujarat', name: 'Gujarat, Hindistan' },
        { id: 'cordoba', name: 'Córdoba, Arjantin' },
        { id: 'mato_grosso', name: 'Mato Grosso, Brezilya' },
        { id: 'nsw', name: 'New South Wales, Avustralya' }
    ];

    // Create dropdown items
    function createDropdownItems(filteredRegions, dropdownMenu, searchInput, isTopBar = false) {
        dropdownMenu.innerHTML = '';

        // Eğer filtreleme sonucu boşsa dropdown'ı gizle
        if (filteredRegions.length === 0) {
            dropdownMenu.classList.remove('show');
            return;
        }

        filteredRegions.forEach(region => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.className = 'dropdown-item';
            a.textContent = region.name;
            a.href = '#';
            a.addEventListener('click', async (e) => {
                e.preventDefault();
                searchInput.value = region.name;
                if (!isTopBar) {
                    await analyzeRegion(region.id);
                } else {
                    await analyzeRegion(region.id);
                }
                dropdownMenu.classList.remove('show');
            });
            li.appendChild(a);
            dropdownMenu.appendChild(li);
        });
    }

    // Show top search bar and results
    function showResults() {
        console.log('Showing results section...');
        const resultsSection = document.getElementById('resultsSection');
        const mainContent = document.getElementById('mainContent');
        const topSearchBar = document.getElementById('topSearchBar');

        if (!resultsSection || !mainContent || !topSearchBar) {
            console.error('Required elements not found:', {
                resultsSection: !!resultsSection,
                mainContent: !!mainContent,
                topSearchBar: !!topSearchBar
            });
            return;
        }

        console.log('Setting display properties...');
        resultsSection.style.display = 'block';
        mainContent.style.display = 'none';
        topSearchBar.style.display = 'block';

        console.log('Adding visible class to results section...');
        resultsSection.classList.add('visible');
        topSearchBar.classList.add('visible');

        console.log('Results section shown successfully');
    }

    // Update statistics display
    function updateStats(droughtStats, ndviStats, lstmScore) {
        // Determine color based on drought ratio
        let colorClass = '';
        const droughtRatio = droughtStats.drought_ratio;

        if (droughtRatio < 25) {
            colorClass = 'low-risk';  // Green
        } else if (droughtRatio < 55) {
            colorClass = 'medium-risk';  // Yellow/Orange
        } else {
            colorClass = 'high-risk';  // Red
        }

        const statsHTML = `
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Last NDVI</div>
                    <div class="stat-value">${ndviStats.last_ndvi.toFixed(3)}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Drought Risk</div>
                    <div class="stat-value ${colorClass}">
                        ${droughtStats.drought_ratio.toFixed(1)}%
                        <div class="risk-indicator ${colorClass}"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">LSTM Drought Score</div>
                    <div class="stat-value">${lstmScore !== null ? lstmScore.toFixed(2) : 'N/A'}</div>
                </div>
            </div>
        `;
        statsContent.innerHTML = statsHTML;
    }

    // Update future NDVI predictions table
    function updateFutureNdviTable(futureNdvi) {
        const tbody = document.querySelector('#futureNdviTable tbody');
        tbody.innerHTML = '';

        for (const [month, value] of Object.entries(futureNdvi)) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${month}</td>
                <td>${value.toFixed(3)}</td>
            `;
            tbody.appendChild(row);
        }
    }

    // Analyze region and update UI
    async function analyzeRegion(region) {
        try {
            console.log('Starting analysis for region:', region);

            // Get drought analysis results
            console.log('Fetching drought analysis...');
            const droughtResponse = await fetch(`/analyze_drought?region=${region}`);
            if (!droughtResponse.ok) {
                throw new Error(`Drought analysis failed: ${droughtResponse.statusText}`);
            }
            const droughtData = await droughtResponse.json();
            console.log('Drought analysis response:', droughtData);

            if (droughtData.error) {
                console.error('Drought analysis error:', droughtData.error);
                throw new Error(droughtData.error);
            }

            // Get NDVI prediction results
            console.log('Fetching NDVI prediction...');
            const ndviResponse = await fetch(`/predict_ndvi?region=${region}`);
            if (!ndviResponse.ok) {
                throw new Error(`NDVI prediction failed: ${ndviResponse.statusText}`);
            }
            const ndviData = await ndviResponse.json();
            console.log('NDVI prediction response:', ndviData);

            if (ndviData.error) {
                console.error('NDVI prediction error:', ndviData.error);
                throw new Error(ndviData.error);
            }

            // Get LSTM drought score
            console.log('Fetching LSTM drought score...');
            let lstmScore = null;
            let lstmTrendPlot = null;
            try {
                const lstmResponse = await fetch(`/predict_lstm_drought?region=${region}`);
                if (lstmResponse.ok) {
                    const lstmData = await lstmResponse.json();
                    lstmScore = lstmData.score;
                    lstmTrendPlot = lstmData.trend_plot;
                } else {
                    console.warn('LSTM drought score fetch failed');
                }
            } catch (e) {
                console.warn('LSTM drought score fetch error:', e);
            }

            // Update UI with results
            console.log('Updating UI...');
            selectedRegion.textContent = `Selected region: ${region.charAt(0).toUpperCase() + region.slice(1)}`;

            // Update drought analysis visualizations
            const heatmapImg = document.getElementById('droughtHeatmap');
            if (!heatmapImg) {
                throw new Error('Could not find heatmap image element');
            }

            console.log('Updating images...');
            heatmapImg.src = 'data:image/png;base64,' + droughtData.heatmap_image;

            // Update NDVI prediction visualizations
            const ndviPlotImg = document.getElementById('ndviPlot');
            if (!ndviPlotImg) {
                throw new Error('Could not find NDVI plot image element');
            }
            ndviPlotImg.src = 'data:image/png;base64,' + ndviData.plot_image;
            updateFutureNdviTable(ndviData.future_ndvi);

            // Update LSTM trend plot visualization
            const lstmTrendImg = document.getElementById('lstmTrendPlot');
            if (lstmTrendImg && lstmTrendPlot) {
                lstmTrendImg.src = 'data:image/png;base64,' + lstmTrendPlot;
            }

            // Update statistics
            updateStats(droughtData.stats, ndviData, lstmScore);

            console.log('Showing results...');
            showResults();
        } catch (error) {
            console.error('Error analyzing region:', error);
            alert(`Error analyzing region: ${error.message}`);
        }
    }

    // Search functionality for main search bar
    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        if (searchTerm.length === 0) {
            dropdownMenu.classList.remove('show');
            return;
        }

        const filteredRegions = regions.filter(region =>
            region.name.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, dropdownMenu, searchInput);
        dropdownMenu.classList.add('show');
    });

    // Search functionality for top search bar
    topSearchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        if (searchTerm.length === 0) {
            topDropdownMenu.classList.remove('show');
            return;
        }

        const filteredRegions = regions.filter(region =>
            region.name.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, topDropdownMenu, topSearchInput, true);
        topDropdownMenu.classList.add('show');
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', function (e) {
        if (!searchInput.contains(e.target) && !dropdownMenu.contains(e.target)) {
            dropdownMenu.classList.remove('show');
        }
        if (!topSearchInput.contains(e.target) && !topDropdownMenu.contains(e.target)) {
            topDropdownMenu.classList.remove('show');
        }
    });

    // Remove click event listeners for showing dropdowns
    searchInput.removeEventListener('click', function () { });
    topSearchInput.removeEventListener('click', function () { });
}); 