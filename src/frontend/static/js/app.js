/**
 * Main application JavaScript for OS Legal Explorer
 * This file handles the core functionality of the single-page application
 */

import { searchCases, getRecentCases, getCaseDetails, getCitationNetwork } from './utils/api.js';
import { createLoadingIndicator, createAlert, debounce, formatDate } from './utils/dom.js';

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeSearch();
    initializeCaseSelection();
    loadRecentCases();
});

/**
 * Initialize the search functionality
 */
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    if (!searchInput || !searchResults) return;

    // Create a debounced search function
    const debouncedSearch = debounce(async (query) => {
        if (query.length < 3) return;

        try {
            // Get filter values
            const jurisdiction = document.querySelector('select[name="jurisdiction"]')?.value || '';
            const court = document.querySelector('select[name="court"]')?.value || '';
            const startDate = document.querySelector('input[name="start_date"]')?.value || '';
            const endDate = document.querySelector('input[name="end_date"]')?.value || '';

            // Create filters object
            const filters = {
                jurisdiction,
                court,
                start_date: startDate,
                end_date: endDate
            };

            // Show loading indicator
            searchResults.innerHTML = '';
            searchResults.appendChild(createLoadingIndicator('lg', 'Searching...'));

            // Perform search
            const results = await searchCases(query, filters);

            // Display results
            displaySearchResults(results);
        } catch (error) {
            console.error('Search error:', error);
            searchResults.innerHTML = '';
            searchResults.appendChild(createAlert('Error searching cases: ' + error.message, 'error'));
        }
    }, 500);

    // Add event listener for search input
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        if (query.length >= 3) {
            debouncedSearch(query);
        } else if (query.length === 0) {
            // Clear results and show recent cases
            searchResults.innerHTML = '';
            loadRecentCases();
        }
    });

    // Add event listeners for filters
    document.querySelectorAll('select[name="jurisdiction"], select[name="court"], input[name="start_date"], input[name="end_date"]').forEach(filter => {
        filter.addEventListener('change', () => {
            const query = searchInput.value.trim();
            if (query.length >= 3) {
                debouncedSearch(query);
            }
        });
    });
}

/**
 * Display search results in the UI
 * @param {Object} results - Search results from the API
 */
function displaySearchResults(results) {
    const searchResults = document.getElementById('search-results');
    if (!searchResults) return;

    // Clear previous results
    searchResults.innerHTML = '';

    // Create results header
    const header = document.createElement('div');
    header.className = 'mb-4';
    header.innerHTML = `
        <h2 class="text-2xl font-bold">Search Results</h2>
        <p class="text-gray-600">${results.count} cases found</p>
    `;
    searchResults.appendChild(header);

    // If no results
    if (results.count === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'bg-white rounded-lg shadow p-6 text-center';
        noResults.innerHTML = `
            <p class="text-gray-600 mb-4">No cases found matching your search criteria.</p>
            <p class="text-gray-600">Try adjusting your search terms or filters.</p>
        `;
        searchResults.appendChild(noResults);
        return;
    }

    // Create results list
    const resultsList = document.createElement('div');
    resultsList.className = 'space-y-4';

    results.results.forEach(caseItem => {
        const caseCard = document.createElement('div');
        caseCard.className = 'bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow';
        caseCard.innerHTML = `
            <a href="#" data-case-id="${caseItem.id}" class="block">
                <h3 class="text-lg font-semibold text-blue-600 hover:text-blue-800">${caseItem.case_name}</h3>
                <div class="text-sm text-gray-600 mt-1">
                    <span>${caseItem.court_name || 'Unknown Court'}</span>
                    <span class="mx-2">•</span>
                    <span>${formatDate(caseItem.date_filed)}</span>
                </div>
                <p class="text-gray-700 mt-2 line-clamp-2">${caseItem.snippet || 'No excerpt available.'}</p>
            </a>
        `;
        resultsList.appendChild(caseCard);
    });

    searchResults.appendChild(resultsList);

    // Add pagination if needed
    if (results.count > results.results.length) {
        const pagination = document.createElement('div');
        pagination.className = 'mt-6 flex justify-center';
        pagination.innerHTML = `
            <button class="btn btn-outline btn-primary">Load More</button>
        `;
        searchResults.appendChild(pagination);
    }

    // Update Alpine.js state to show we have results
    const alpineData = Alpine.evaluate(searchResults, 'hasResults');
    if (alpineData) {
        Alpine.evaluate(searchResults, 'hasResults = true');
    }
}

/**
 * Initialize case selection functionality
 */
function initializeCaseSelection() {
    // Event delegation for case selection
    document.body.addEventListener('click', (e) => {
        const caseLink = e.target.closest('[data-case-id]');
        if (caseLink) {
            e.preventDefault();
            const caseId = caseLink.dataset.caseId;
            loadCaseDetails(caseId);
        }
    });
}

/**
 * Load case details from the API
 * @param {string} caseId - ID of the case to load
 */
async function loadCaseDetails(caseId) {
    const caseDetail = document.getElementById('case-detail');
    if (!caseDetail) return;

    try {
        // Show loading indicator
        caseDetail.innerHTML = '';
        caseDetail.appendChild(createLoadingIndicator('lg', 'Loading case details...'));

        // Fetch case details
        const caseData = await getCaseDetails(caseId);

        // Display case details
        displayCaseDetails(caseData);

        // Load citation network
        loadCitationNetwork(caseId);

        // Scroll to case detail on mobile
        if (window.innerWidth < 1024) {
            caseDetail.scrollIntoView({
                behavior: 'smooth'
            });
        }
    } catch (error) {
        console.error('Error loading case details:', error);
        caseDetail.innerHTML = '';
        caseDetail.appendChild(createAlert('Error loading case details: ' + error.message, 'error'));
    }
}

/**
 * Display case details in the UI
 * @param {Object} caseData - Case data from the API
 */
function displayCaseDetails(caseData) {
    const caseDetail = document.getElementById('case-detail');
    if (!caseDetail) return;

    // Clear previous content
    caseDetail.innerHTML = '';

    // Create case header
    const header = document.createElement('div');
    header.className = 'mb-6 pb-4 border-b';
    header.innerHTML = `
        <h2 class="text-2xl font-bold">${caseData.case_name}</h2>
        <div class="flex flex-wrap gap-2 mt-2">
            <span class="badge badge-primary">${caseData.court_name || 'Unknown Court'}</span>
            <span class="badge badge-secondary">${formatDate(caseData.date_filed)}</span>
            ${caseData.citation ? `<span class="badge badge-outline">${caseData.citation}</span>` : ''}
        </div>
    `;
    caseDetail.appendChild(header);

    // Create tabs for opinion text and citation map
    const tabs = document.createElement('div');
    tabs.className = 'tabs tabs-boxed mb-4';
    tabs.innerHTML = `
        <a class="tab tab-active" data-tab="opinion">Opinion Text</a>
        <a class="tab" data-tab="citation-map">Citation Network</a>
    `;
    caseDetail.appendChild(tabs);

    // Create tab content container
    const tabContent = document.createElement('div');
    tabContent.className = 'tab-content';

    // Opinion text content
    const opinionContent = document.createElement('div');
    opinionContent.id = 'opinion-content';
    opinionContent.className = 'prose max-w-none';
    opinionContent.innerHTML = `
        <div class="whitespace-pre-wrap font-serif text-gray-800 leading-relaxed">
            ${caseData.html || caseData.plain_text || 'No opinion text available.'}
        </div>
    `;
    tabContent.appendChild(opinionContent);

    // Citation map content
    const citationContent = document.createElement('div');
    citationContent.id = 'citation-map-content';
    citationContent.className = 'hidden';
    citationContent.innerHTML = `
        <div id="citation-map" class="h-[500px] border rounded-lg"></div>
        <div id="citation-map-loading" class="flex justify-center items-center py-8">
            <div class="loading loading-spinner loading-lg"></div>
        </div>
    `;
    tabContent.appendChild(citationContent);

    caseDetail.appendChild(tabContent);

    // Add tab switching functionality
    tabs.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            // Remove active class from all tabs
            tabs.querySelectorAll('.tab').forEach(t => t.classList.remove('tab-active'));

            // Add active class to clicked tab
            e.target.classList.add('tab-active');

            // Hide all tab content
            tabContent.querySelectorAll('> div').forEach(content => content.classList.add('hidden'));

            // Show selected tab content
            const tabId = e.target.dataset.tab;
            if (tabId === 'opinion') {
                opinionContent.classList.remove('hidden');
            } else if (tabId === 'citation-map') {
                citationContent.classList.remove('hidden');
                // Ensure citation map is loaded
                loadCitationNetwork(caseData.id);
            }
        });
    });
}

/**
 * Load citation network for a case
 * @param {string} caseId - ID of the case
 */
function loadCitationNetwork(caseId) {
    const citationMap = document.getElementById('citation-map');
    if (!citationMap) return;

    // Check if citation map is already loaded
    if (citationMap.dataset.loaded === caseId) return;

    // Set height for the visualization
    citationMap.style.height = '500px';

    // Load and render the citation network
    import('./citation_map.js')
        .then(module => {
            if (typeof module.loadAndRenderCitationNetwork === 'function') {
                module.loadAndRenderCitationNetwork(caseId, 'citation-map');
                citationMap.dataset.loaded = caseId;
            } else {
                console.error('Citation map module does not export loadAndRenderCitationNetwork function');
            }
        })
        .catch(error => {
            console.error('Error loading citation map module:', error);
            citationMap.innerHTML = '<div class="flex justify-center items-center h-full"><p>Failed to load citation network visualization.</p></div>';
        });
}

/**
 * Load recent cases from the API
 */
async function loadRecentCases() {
    const recentCases = document.getElementById('recent-cases');
    if (!recentCases) return;

    try {
        // Show loading indicator
        recentCases.innerHTML = '';
        recentCases.appendChild(createLoadingIndicator('lg', 'Loading recent cases...'));

        // Fetch recent cases
        const cases = await getRecentCases(10, 0);

        // Display recent cases
        displayRecentCases(cases);
    } catch (error) {
        console.error('Error loading recent cases:', error);
        recentCases.innerHTML = '';
        recentCases.appendChild(createAlert('Error loading recent cases: ' + error.message, 'error'));
    }
}

/**
 * Display recent cases in the UI
 * @param {Object} cases - Recent cases data from the API
 */
function displayRecentCases(cases) {
    const recentCases = document.getElementById('recent-cases');
    if (!recentCases) return;

    // Clear previous content
    recentCases.innerHTML = '';

    // If no cases
    if (!cases.results || cases.results.length === 0) {
        const noCases = document.createElement('div');
        noCases.className = 'bg-white rounded-lg shadow p-6 text-center';
        noCases.innerHTML = `
            <p class="text-gray-600">No recent cases available.</p>
        `;
        recentCases.appendChild(noCases);
        return;
    }

    // Create cases list
    const casesList = document.createElement('div');
    casesList.className = 'space-y-4';

    cases.results.forEach(caseItem => {
        const caseCard = document.createElement('div');
        caseCard.className = 'bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow';
        caseCard.innerHTML = `
            <a href="#" data-case-id="${caseItem.id}" class="block">
                <h3 class="text-lg font-semibold text-blue-600 hover:text-blue-800">${caseItem.case_name}</h3>
                <div class="text-sm text-gray-600 mt-1">
                    <span>${caseItem.court_name || 'Unknown Court'}</span>
                    <span class="mx-2">•</span>
                    <span>${formatDate(caseItem.date_filed)}</span>
                </div>
                <p class="text-gray-700 mt-2 line-clamp-2">${caseItem.snippet || 'No excerpt available.'}</p>
            </a>
        `;
        casesList.appendChild(caseCard);
    });

    recentCases.appendChild(casesList);

    // Add load more button
    if (cases.count > cases.results.length) {
        const loadMore = document.createElement('div');
        loadMore.className = 'mt-6 text-center';
        loadMore.innerHTML = `
            <button class="btn btn-outline btn-primary btn-sm"
                    hx-get="/api/recent-cases?offset=${cases.results.length}"
                    hx-target="#recent-cases"
                    hx-swap="outerHTML"
                    hx-indicator="#recent-cases-loading">
                Load More
            </button>
        `;
        recentCases.appendChild(loadMore);
    }
} 