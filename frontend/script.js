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
        //{ id: 'zacatecas', name: 'Zacatecas, Meksika' },
        { id: 'gauteng', name: 'Gauteng, Güney Afrika' },
        { id: 'addis_ababa', name: 'Addis Ababa, Etiyopya' },
        { id: 'yunnan', name: 'Yunnan, Çin' },
        { id: 'gujarat', name: 'Gujarat, Hindistan' },
        { id: 'cordoba', name: 'Córdoba, Arjantin' },
        //{ id: 'mato_grosso', name: 'Mato Grosso, Brezilya' },
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
        if (!statsContent) {
            console.warn('Stats content element not found');
            return;
        }

        // Determine color based on drought ratio
        let colorClass = '';
        const droughtRatio = droughtStats?.drought_ratio || 0;

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
                    <div class="stat-value">${ndviStats?.last_ndvi?.toFixed(3) || 'N/A'}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-item">
                    <div class="stat-label">Drought Percentage</div>
                    <div class="stat-value ${colorClass}">
                        ${droughtRatio.toFixed(1)}%
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
        if (!futureNdvi) {
            console.warn('No future NDVI data available');
            return;
        }

        const tbody = document.querySelector('#futureNdviTable tbody');
        if (!tbody) {
            console.warn('Future NDVI table body not found');
            return;
        }

        tbody.innerHTML = '';

        for (const [month, value] of Object.entries(futureNdvi)) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${month}</td>
                <td>${typeof value === 'number' ? value.toFixed(3) : 'N/A'}</td>
            `;
            tbody.appendChild(row);
        }
    }

    // LSTM verilerini periyodik olarak kontrol et
    async function pollLSTMData(region) {
        const maxAttempts = 60; // 5 dakika (5 saniye * 60)
        let attempts = 0;

        // Loading state göster
        const temperaturePlotImg = document.getElementById('temperaturePlot');
        const droughtPlotImg = document.getElementById('droughtPlot');
        if (temperaturePlotImg) {
            temperaturePlotImg.src = 'images/loading.gif';
        }
        if (droughtPlotImg) {
            droughtPlotImg.src = 'images/loading.gif';
        }

        const poll = async () => {
            try {
                const response = await fetch(`/predict_lstm_drought?region=${region}`);
                const data = await response.json();

                if (data.status === 'calculating') {
                    if (attempts < maxAttempts) {
                        attempts++;
                        setTimeout(poll, 5000); // 5 saniye bekle
                    } else {
                        throw new Error('LSTM calculation timeout - please try again');
                    }
                } else if (data.error) {
                    throw new Error(data.error);
                } else {
                    // LSTM verileri hazır, UI'ı güncelle
                    updateLSTMVisualizations(data);
                }
            } catch (error) {
                console.error('Error polling LSTM data:', error);
                // Hata durumunda UI'da göster
                if (temperaturePlotImg) {
                    temperaturePlotImg.src = 'images/error.png';
                    temperaturePlotImg.alt = 'Error loading temperature data: ' + error.message;
                }
                if (droughtPlotImg) {
                    droughtPlotImg.src = 'images/error.png';
                    droughtPlotImg.alt = 'Error loading drought data: ' + error.message;
                }
            }
        };

        poll();
    }

    // LSTM görselleştirmelerini güncelle
    function updateLSTMVisualizations(data) {
        if (data.temperature_plot) {
            const temperaturePlotImg = document.getElementById('temperaturePlot');
            if (temperaturePlotImg) {
                temperaturePlotImg.src = 'data:image/png;base64,' + data.temperature_plot;
            } else {
                console.warn('Temperature plot element not found');
            }
        }

        if (data.drought_plot) {
            const droughtPlotImg = document.getElementById('droughtPlot');
            if (droughtPlotImg) {
                droughtPlotImg.src = 'data:image/png;base64,' + data.drought_plot;
            } else {
                console.warn('Drought plot element not found');
            }
        }

        // LSTM skorunu label'a göre güncelle
        if (data.score !== undefined) {
            const statsContent = document.getElementById('statsContent');
            if (statsContent) {
                const statItems = statsContent.querySelectorAll('.stat-item');
                statItems.forEach(item => {
                    const label = item.querySelector('.stat-label');
                    if (label && label.textContent.includes('LSTM Drought Score')) {
                        const value = item.querySelector('.stat-value');
                        if (value) value.textContent = data.score.toFixed(2);
                    }
                });
            }
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

            // Update UI with initial results
            console.log('Updating UI...');
            const selectedRegionElement = document.getElementById('selectedRegion');
            if (selectedRegionElement) {
                selectedRegionElement.textContent = `Selected region: ${region.charAt(0).toUpperCase() + region.slice(1)}`;
            }

            // Update drought analysis visualizations
            const droughtScoreElement = document.getElementById('droughtScore');
            if (droughtScoreElement && droughtData.stats) {
                const droughtRatio = droughtData.stats.drought_ratio || 0;
                droughtScoreElement.textContent = `${droughtRatio.toFixed(1)}%`;

                // Add color class based on drought ratio
                droughtScoreElement.className = 'stat-value';
                if (droughtRatio < 25) {
                    droughtScoreElement.classList.add('low-risk');
                } else if (droughtRatio < 55) {
                    droughtScoreElement.classList.add('medium-risk');
                } else {
                    droughtScoreElement.classList.add('high-risk');
                }
            }

            // Update NDVI prediction visualizations
            const ndviPlotImg = document.getElementById('ndviPlot');
            console.log('NDVI data received:', ndviData);
            if (ndviPlotImg && ndviData.plot_image) {
                console.log('Updating NDVI plot image');
                ndviPlotImg.src = 'data:image/png;base64,' + ndviData.plot_image;
                if (ndviData.future_ndvi) {
                    console.log('Updating future NDVI table with data:', ndviData.future_ndvi);
                    updateFutureNdviTable(ndviData.future_ndvi);
                } else {
                    console.warn('No future NDVI data available');
                }
            } else {
                console.warn('NDVI plot image or element not found', {
                    hasPlotImg: !!ndviPlotImg,
                    hasPlotImage: !!ndviData?.plot_image
                });
                if (ndviPlotImg) {
                    ndviPlotImg.src = 'images/error.png';
                    ndviPlotImg.alt = 'Error loading NDVI plot';
                }
            }

            // Update statistics with initial data
            updateStats(
                droughtData.stats || { drought_ratio: 0 },
                { last_ndvi: ndviData.last_ndvi },
                null
            );

            // Start polling for LSTM data
            pollLSTMData(region);

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