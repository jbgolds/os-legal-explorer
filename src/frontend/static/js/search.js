/**
 * Search functionality for OS Legal Explorer
 * This module handles search input, API calls, and results display
 */

import { searchCases } from './utils/api.js';
import { debounce, createLoadingIndicator, createAlert } from './utils/dom.js';

/**
 * Initialize search functionality
 * @param {Object} options - Configuration options
 * @param {string} options.inputSelector - Selector for search input element
 * @param {string} options.resultsSelector - Selector for search results container
 * @param {string} options.loadingSelector - Selector for loading indicator
 * @param {number} options.debounceTime - Debounce time in milliseconds
 * @param {number} options.minChars - Minimum characters to trigger search
 */
export function initializeSearch(options = {}) {
    const {
        inputSelector = '#search-input',
        resultsSelector = '#search-results',
        loadingSelector = '#search-indicator',
        debounceTime = 500,
        minChars = 3
    } = options;

    const searchInput = document.querySelector(inputSelector);
    const searchResults = document.querySelector(resultsSelector);
    const loadingIndicator = document.querySelector(loadingSelector);

    if (!searchInput || !searchResults) {
        console.error('Search elements not found');
        return;
    }

    // Create debounced search function
    const debouncedSearch = debounce(async (query, filters = {}) => {
        if (query.length < minChars) return;

        try {
            // Show loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.remove('hidden');
            } else {
                // Create and append loading indicator if not found
                const loader = createLoadingIndicator('sm', 'Searching...');
                searchResults.innerHTML = '';
                searchResults.appendChild(loader);
            }

            // Perform search
            const results = await searchCases(query, filters);

            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.add('hidden');
            }

            // Render results
            renderSearchResults(results, searchResults);

            // Update Alpine.js state if available
            if (window.Alpine && searchResults.hasAttribute('x-data')) {
                Alpine.evaluate(searchResults, 'hasResults = true');
            }
        } catch (error) {
            console.error('Search error:', error);

            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.add('hidden');
            }

            // Show error message
            searchResults.innerHTML = '';
            searchResults.appendChild(createAlert('Error searching cases: ' + error.message, 'error'));
        }
    }, debounceTime);

    // Add event listener for search input
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();

        // Update Alpine.js state if available
        if (window.Alpine && searchInput.hasAttribute('x-model')) {
            Alpine.evaluate(searchInput, `searchQuery = "${query}"`);
        }

        if (query.length >= minChars) {
            // Get filter values
            const filters = getSearchFilters();
            debouncedSearch(query, filters);
        } else if (query.length === 0) {
            // Clear results
            searchResults.innerHTML = '';

            // Update Alpine.js state if available
            if (window.Alpine && searchResults.hasAttribute('x-data')) {
                Alpine.evaluate(searchResults, 'hasResults = false');
            }

            // Trigger event to load recent cases
            const event = new CustomEvent('search:cleared');
            document.dispatchEvent(event);
        }
    });

    // Add event listeners for filters
    document.querySelectorAll('select[name^="filter-"], input[name^="filter-"]').forEach(filter => {
        filter.addEventListener('change', () => {
            const query = searchInput.value.trim();
            if (query.length >= minChars) {
                const filters = getSearchFilters();
                debouncedSearch(query, filters);
            }
        });
    });

    // Function to get all search filters
    function getSearchFilters() {
        const filters = {};

        // Get jurisdiction filter
        const jurisdiction = document.querySelector('select[name="jurisdiction"]')?.value;
        if (jurisdiction) filters.jurisdiction = jurisdiction;

        // Get court filter
        const court = document.querySelector('select[name="court"]')?.value;
        if (court) filters.court = court;

        // Get date range filters
        const startDate = document.querySelector('input[name="start_date"]')?.value;
        if (startDate) filters.start_date = startDate;

        const endDate = document.querySelector('input[name="end_date"]')?.value;
        if (endDate) filters.end_date = endDate;

        return filters;
    }
}

/**
 * Render search results in the specified container
 * @param {Object} results - Search results from the API
 * @param {HTMLElement} container - Container element for results
 */
export function renderSearchResults(results, container) {
    // Clear previous results
    container.innerHTML = '';

    // Create results header
    const header = document.createElement('div');
    header.className = 'mb-4';
    header.innerHTML = `
        <h2 class="text-2xl font-bold">Search Results</h2>
        <p class="text-gray-600">${results.count} cases found</p>
    `;
    container.appendChild(header);

    // If no results
    if (results.count === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'bg-white rounded-lg shadow p-6 text-center';
        noResults.innerHTML = `
            <p class="text-gray-600 mb-4">No cases found matching your search criteria.</p>
            <p class="text-gray-600">Try adjusting your search terms or filters.</p>
        `;
        container.appendChild(noResults);
        return;
    }

    // Create results list
    const resultsList = document.createElement('div');
    resultsList.className = 'space-y-4';

    results.results.forEach(caseItem => {
        const caseCard = document.createElement('div');
        caseCard.className = 'bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow';

        // Format date if available
        const dateFormatted = caseItem.date_filed
            ? new Date(caseItem.date_filed).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
            : 'Unknown date';

        caseCard.innerHTML = `
            <a href="#" data-case-id="${caseItem.id}" class="block">
                <h3 class="text-lg font-semibold text-blue-600 hover:text-blue-800">${caseItem.case_name}</h3>
                <div class="text-sm text-gray-600 mt-1">
                    <span>${caseItem.court_name || 'Unknown Court'}</span>
                    <span class="mx-2">•</span>
                    <span>${dateFormatted}</span>
                </div>
                <p class="text-gray-700 mt-2 line-clamp-2">${caseItem.snippet || 'No excerpt available.'}</p>
            </a>
        `;
        resultsList.appendChild(caseCard);
    });

    container.appendChild(resultsList);

    // Add pagination if needed
    if (results.count > results.results.length) {
        const pagination = document.createElement('div');
        pagination.className = 'mt-6 flex justify-center';

        // Create load more button with HTMX attributes
        pagination.innerHTML = `
            <button class="btn btn-outline btn-primary"
                    hx-get="/api/search?q=${encodeURIComponent(results.query)}&offset=${results.results.length}"
                    hx-target="#search-results"
                    hx-swap="outerHTML"
                    hx-indicator="#search-indicator">
                Load More
            </button>
        `;
        container.appendChild(pagination);
    }
}

// Export the module functions
export default {
    initializeSearch,
    renderSearchResults
}; 