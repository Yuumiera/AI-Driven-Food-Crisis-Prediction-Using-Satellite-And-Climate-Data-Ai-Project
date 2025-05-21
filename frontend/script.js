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

    // Available regions list
    const regions = [
        'sanliurfa',
        'punjab',
        'yunnan',
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
                    // When selection is made from main search bar
                    await analyzeRegion(region);
                } else {
                    // When selection is made from top search bar
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
    function updateStats(stats) {
        const statsHTML = `
            <div class="col-md-4">
                <div class="stat-item">
                    <div class="stat-label">Mean Score</div>
                    <div class="stat-value">${stats.mean_score.toFixed(3)}</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-item">
                    <div class="stat-label">Drought Ratio</div>
                    <div class="stat-value">${stats.drought_ratio.toFixed(1)}%</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-item">
                    <div class="stat-label">Severity (Max Score)</div>
                    <div class="stat-value">${stats.max_score.toFixed(3)}</div>
                </div>
            </div>
        `;
        statsContent.innerHTML = statsHTML;
    }

    // Analyze region and update UI
    async function analyzeRegion(region) {
        try {
            const response = await fetch(`/analyze_drought?region=${region}`);
            const data = await response.json();

            if (data.error) {
                console.error('Analysis error:', data.error);
                return;
            }

            // Update UI with results
            selectedRegion.textContent = `Selected region: ${region.charAt(0).toUpperCase() + region.slice(1)}`;
            heatmapImage.src = `data:image/png;base64,${data.heatmap_image}`;
            classificationImage.src = `data:image/png;base64,${data.classification_image}`;
            updateStats(data.stats);
            showResults();
        } catch (error) {
            console.error('Error analyzing region:', error);
        }
    }

    // Initialize dropdowns
    createDropdownItems(regions, dropdownMenu, searchInput);
    createDropdownItems(regions, topDropdownMenu, topSearchInput, true);

    // Search functionality for main search bar
    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const filteredRegions = regions.filter(region =>
            region.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, dropdownMenu, searchInput);
    });

    // Search functionality for top search bar
    topSearchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const filteredRegions = regions.filter(region =>
            region.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, topDropdownMenu, topSearchInput, true);
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

    // Show dropdowns on input click
    searchInput.addEventListener('click', function () {
        dropdownMenu.classList.add('show');
    });

    topSearchInput.addEventListener('click', function () {
        topDropdownMenu.classList.add('show');
    });
}); 