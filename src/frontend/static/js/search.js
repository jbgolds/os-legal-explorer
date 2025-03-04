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
 * @param {string} options.resultsSelector - Selector for search results dropdown
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

    // Create dropdown container if it doesn't exist
    let dropdownContainer = document.querySelector('#search-dropdown');
    if (!dropdownContainer) {
        dropdownContainer = document.createElement('div');
        dropdownContainer.id = 'search-dropdown';
        dropdownContainer.className = 'dropdown dropdown-open w-full';
        searchInput.parentNode.appendChild(dropdownContainer);

        // Move the input into the dropdown container
        const inputWrapper = document.createElement('div');
        inputWrapper.className = 'w-full';
        searchInput.parentNode.insertBefore(dropdownContainer, searchInput);
        inputWrapper.appendChild(searchInput);
        dropdownContainer.appendChild(inputWrapper);
    }

    // Create debounced search function
    const debouncedSearch = debounce(async (query, filters = {}) => {
        if (query.length < minChars) {
            hideDropdown();
            return;
        }

        try {
            // Show loading indicator
            showLoadingState();

            // Perform search
            const results = await searchCases(query, filters);

            // Hide loading indicator
            hideLoadingState();

            // Render results in dropdown
            renderSearchResults(results, searchResults);

        } catch (error) {
            console.error('Search error:', error);
            hideLoadingState();
            showErrorMessage(error.message);
        }
    }, debounceTime);

    // Add event listener for search input
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();

        if (query.length >= minChars) {
            const filters = getSearchFilters();
            debouncedSearch(query, filters);
        } else {
            hideDropdown();
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdownContainer.contains(e.target)) {
            hideDropdown();
        }
    });

    function showLoadingState() {
        if (loadingIndicator) {
            loadingIndicator.classList.remove('hidden');
        }
        searchInput.classList.add('loading');
    }

    function hideLoadingState() {
        if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
        }
        searchInput.classList.remove('loading');
    }

    function hideDropdown() {
        searchResults.innerHTML = '';
        dropdownContainer.classList.remove('dropdown-open');
    }

    function showErrorMessage(message) {
        searchResults.innerHTML = `
            <div class="dropdown-content menu p-2 shadow bg-base-100 rounded-box w-full">
                <div class="text-error p-4">${message}</div>
            </div>
        `;
        dropdownContainer.classList.add('dropdown-open');
    }

    // Function to get all search filters
    function getSearchFilters() {
        const filters = {};

        // Get jurisdiction filter
        const jurisdiction = document.querySelector('select[name="jurisdiction"]')?.value;
        if (jurisdiction && jurisdiction !== 'all') filters.jurisdiction = jurisdiction;

        // Get court filter
        const court = document.querySelector('select[name="court"]')?.value;
        if (court && court !== 'all') filters.court = court;

        // Get date range filters
        const startDate = document.querySelector('input[name="start_date"]')?.value;
        if (startDate) filters.start_date = startDate;

        const endDate = document.querySelector('input[name="end_date"]')?.value;
        if (endDate) filters.end_date = endDate;

        // Get year range filters
        const yearFrom = document.querySelector('input[name="year_from"]')?.value;
        if (yearFrom) filters.year_from = yearFrom;

        const yearTo = document.querySelector('input[name="year_to"]')?.value;
        if (yearTo) filters.year_to = yearTo;

        return filters;
    }
}

/**
 * Render search results in the dropdown
 * @param {Object} results - Search results from the API
 * @param {HTMLElement} container - Container element for results
 */
export function renderSearchResults(results, container) {
    const dropdownContainer = document.querySelector('#search-dropdown');

    // Create dropdown content
    const content = document.createElement('div');
    content.className = 'dropdown-content menu p-2 shadow bg-base-100 rounded-box w-full max-h-96 overflow-y-auto';

    // If no results
    if (!results.count || !results.results?.length) {
        content.innerHTML = `
            <div class="p-4 text-center text-gray-500">
                No cases found matching your search criteria.
            </div>
        `;
        container.innerHTML = '';
        container.appendChild(content);
        dropdownContainer.classList.add('dropdown-open');
        return;
    }

    // Create results list
    const resultsList = document.createElement('ul');
    resultsList.className = 'menu menu-compact';

    results.results.forEach(caseItem => {
        const li = document.createElement('li');

        // Format date
        const dateFormatted = caseItem.dateFiled
            ? new Date(caseItem.dateFiled).toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            })
            : '';

        // Create result item
        li.innerHTML = `
            <a href="#" data-cluster-id="${caseItem.cluster_id}" class="hover:bg-base-200">
                <div class="flex flex-col gap-1">
                    <div class="font-medium">${caseItem.caseName}</div>
                    <div class="text-sm text-gray-600">
                        ${caseItem.court}
                        ${dateFormatted ? `<span class="mx-1">•</span>${dateFormatted}` : ''}
                    </div>
                </div>
            </a>
        `;

        // Add click handler for case selection
        li.querySelector('a').addEventListener('click', async (e) => {
            e.preventDefault();
            const clusterId = e.currentTarget.dataset.clusterId;

            try {
                // Show loading state
                document.querySelector('#search-input').value = caseItem.caseName;
                hideDropdown();

                // Fetch case details from our backend
                const response = await fetch(`/api/case/${clusterId}`);
                if (!response.ok) throw new Error('Failed to fetch case details');
                const caseDetails = await response.json();

                // Dispatch event with case details
                const event = new CustomEvent('case:selected', {
                    detail: {
                        caseDetails,
                        clusterId
                    }
                });
                document.dispatchEvent(event);

            } catch (error) {
                console.error('Error fetching case details:', error);
                // Show error message
                showErrorMessage('Error fetching case details. Please try again.');
            }
        });

        resultsList.appendChild(li);
    });

    content.appendChild(resultsList);

    // Add "Show all results" link if there are more results
    if (results.count > results.results.length) {
        const showAll = document.createElement('div');
        showAll.className = 'p-2 text-center border-t';
        showAll.innerHTML = `
            <button class="btn btn-ghost btn-sm">
                Show all ${results.count} results
            </button>
        `;
        showAll.querySelector('button').addEventListener('click', () => {
            // TODO: Show full results page
            console.log('Show all results clicked');
        });
        content.appendChild(showAll);
    }

    // Update the container
    container.innerHTML = '';
    container.appendChild(content);
    dropdownContainer.classList.add('dropdown-open');
}

function hideDropdown() {
    const dropdown = document.querySelector('#search-dropdown');
    if (dropdown) {
        dropdown.classList.remove('dropdown-open');
        const results = dropdown.querySelector('#search-results');
        if (results) results.innerHTML = '';
    }
}

// Export the module functions
export default {
    initializeSearch,
    renderSearchResults
}; 