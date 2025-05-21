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

    // Available regions list
    const regions = [
        'sanliurfa',
        'avustralia',
        'punjab',
        'munich',
        'california',
        'iowa',
        'kano',
        'zacatecas',
        'gauteng'
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
            a.textContent = region.charAt(0).toUpperCase() + region.slice(1);
            a.href = '#';
            a.addEventListener('click', async (e) => {
                e.preventDefault();
                searchInput.value = region;
                if (!isTopBar) {
                    await analyzeRegion(region);
                } else {
                    await analyzeRegion(region);
                }
                dropdownMenu.classList.remove('show');
            });
            li.appendChild(a);
            dropdownMenu.appendChild(li);
        });
    }

    // Show top search bar and results
    function showResults() {
        mainContent.classList.add('hidden');
        topSearchBar.classList.add('visible');
        resultsSection.classList.add('visible');
        window.scrollTo(0, 0);
    }

    // Update statistics display
    function updateStats(droughtStats, ndviStats) {
        const statsHTML = `
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Last NDVI</div>
                    <div class="stat-value">${ndviStats.last_ndvi.toFixed(3)}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Drought Ratio</div>
                    <div class="stat-value">${droughtStats.drought_ratio.toFixed(1)}%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">NDVI R² Score</div>
                    <div class="stat-value">${ndviStats.metrics.r2.toFixed(3)}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Severity (Max Score)</div>
                    <div class="stat-value">${droughtStats.max_score.toFixed(3)}</div>
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
            // Get drought analysis results
            const droughtResponse = await fetch(`/analyze_drought?region=${region}`);
            const droughtData = await droughtResponse.json();

            if (droughtData.error) {
                console.error('Drought analysis error:', droughtData.error);
                return;
            }

            // Get NDVI prediction results
            const ndviResponse = await fetch(`/predict_ndvi?region=${region}`);
            const ndviData = await ndviResponse.json();

            if (ndviData.error) {
                console.error('NDVI prediction error:', ndviData.error);
                return;
            }

            // Update UI with results
            selectedRegion.textContent = `Selected region: ${region.charAt(0).toUpperCase() + region.slice(1)}`;

            // Update drought analysis visualizations
            document.getElementById('droughtHeatmap').src = 'data:image/png;base64,' + droughtData.heatmap_image;
            document.getElementById('droughtClassification').src = 'data:image/png;base64,' + droughtData.classification_image;

            // Update NDVI prediction visualizations
            document.getElementById('ndviPlot').src = 'data:image/png;base64,' + ndviData.plot_image;
            updateFutureNdviTable(ndviData.future_ndvi);

            // Update statistics
            updateStats(droughtData.stats, ndviData);

            showResults();
        } catch (error) {
            console.error('Error analyzing region:', error);
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
            region.toLowerCase().startsWith(searchTerm)
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
            region.toLowerCase().startsWith(searchTerm)
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