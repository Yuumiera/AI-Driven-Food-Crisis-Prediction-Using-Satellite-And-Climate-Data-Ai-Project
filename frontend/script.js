document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('searchInput');
    const topSearchInput = document.getElementById('topSearchInput');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const topDropdownMenu = document.getElementById('topDropdownMenu');
    const selectedRegion = document.getElementById('selectedRegion');
    const mainContent = document.getElementById('mainContent');
    const topSearchBar = document.getElementById('topSearchBar');
    const resultsSection = document.getElementById('resultsSection');

    // Örnek bölge listesi - bu listeyi gerçek verilerle değiştirebilirsiniz
    const regions = [
        'Sub-Saharan Africa',
        'South Asia',
        'East Asia & Pacific',
        'Middle East & North Africa',
        'Latin America & Caribbean',
        'Europe & Central Asia',
        'North America'
    ];

    // Dropdown menüsünü oluştur
    function createDropdownItems(filteredRegions, dropdownMenu, searchInput, isTopBar = false) {
        dropdownMenu.innerHTML = '';
        filteredRegions.forEach(region => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.className = 'dropdown-item';
            a.textContent = region;
            a.href = '#';
            a.addEventListener('click', (e) => {
                e.preventDefault();
                searchInput.value = region;
                if (!isTopBar) {
                    // Ana arama çubuğundan seçim yapıldığında
                    selectedRegion.textContent = `Selected region: ${region}`;
                    showTopSearchBar(region);
                } else {
                    // Üst arama çubuğundan seçim yapıldığında
                    selectedRegion.textContent = `Selected region: ${region}`;
                }
                dropdownMenu.classList.remove('show');
            });
            li.appendChild(a);
            dropdownMenu.appendChild(li);
        });
    }

    // Üst arama çubuğunu göster
    function showTopSearchBar(selectedRegion) {
        // Ana içeriği gizle
        mainContent.classList.add('hidden');

        // Üst arama çubuğunu göster
        topSearchBar.classList.add('visible');
        topSearchInput.value = selectedRegion;

        // Sonuçlar bölümünü göster
        resultsSection.classList.add('visible');

        // Sayfayı en üste kaydır
        window.scrollTo(0, 0);
    }

    // Başlangıçta tüm bölgeleri göster
    createDropdownItems(regions, dropdownMenu, searchInput);
    createDropdownItems(regions, topDropdownMenu, topSearchInput, true);

    // Ana arama çubuğu için arama fonksiyonu
    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const filteredRegions = regions.filter(region =>
            region.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, dropdownMenu, searchInput);
    });

    // Üst arama çubuğu için arama fonksiyonu
    topSearchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();
        const filteredRegions = regions.filter(region =>
            region.toLowerCase().includes(searchTerm)
        );
        createDropdownItems(filteredRegions, topDropdownMenu, topSearchInput, true);
    });

    // Dropdown menülerini kapat
    document.addEventListener('click', function (e) {
        if (!searchInput.contains(e.target) && !dropdownMenu.contains(e.target)) {
            dropdownMenu.classList.remove('show');
        }
        if (!topSearchInput.contains(e.target) && !topDropdownMenu.contains(e.target)) {
            topDropdownMenu.classList.remove('show');
        }
    });

    // Input'lara tıklandığında dropdown'ları göster
    searchInput.addEventListener('click', function () {
        dropdownMenu.classList.add('show');
    });

    topSearchInput.addEventListener('click', function () {
        topDropdownMenu.classList.add('show');
    });
}); 