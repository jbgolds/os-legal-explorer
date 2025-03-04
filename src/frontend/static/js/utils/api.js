/**
 * API utility functions for OS Legal Explorer
 * This module provides functions for interacting with the backend API
 */

// Base API URL - adjust as needed for production
const API_BASE_URL = '';

/**
 * Generic function to make API requests
 * @param {string} endpoint - API endpoint to call
 * @param {Object} options - Fetch options
 * @returns {Promise} - Promise that resolves to the API response
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    // Default options
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
    };

    // Merge options
    const fetchOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, fetchOptions);

        // Check if the request was successful
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API request failed with status ${response.status}`);
        }

        // Parse JSON response
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

/**
 * Search for court cases
 * @param {string} query - Search query
 * @param {Object} filters - Optional filters (jurisdiction, date range, etc.)
 * @returns {Promise} - Promise that resolves to search results
 */
export async function searchCases(query, filters = {}) {
    const queryParams = new URLSearchParams({
        q: query,
        ...filters,
    });

    return apiRequest(`/api/search?${queryParams}`);
}

/**
 * Get recent court cases
 * @param {number} limit - Number of cases to retrieve
 * @param {number} offset - Offset for pagination
 * @returns {Promise} - Promise that resolves to recent cases
 */
export async function getRecentCases(limit = 10, offset = 0) {
    return apiRequest(`/api/recent-cases?limit=${limit}&offset=${offset}`);
}

/**
 * Get details for a specific court case
 * @param {string} caseId - ID of the case to retrieve
 * @returns {Promise} - Promise that resolves to case details
 */
export async function getCaseDetails(caseId) {
    return apiRequest(`/api/case/${caseId}`);
}

/**
 * Get citation network for a specific court case
 * @param {string} caseId - ID of the case
 * @param {number} depth - Depth of the citation network
 * @returns {Promise} - Promise that resolves to citation network data
 */
export async function getCitationNetwork(caseId, depth = 1) {
    return apiRequest(`/api/case/${caseId}/citations?depth=${depth}`);
}

/**
 * Submit feedback about a missing citation
 * @param {Object} feedback - Feedback data
 * @returns {Promise} - Promise that resolves to confirmation
 */
export async function submitMissingCitationFeedback(feedback) {
    return apiRequest('/api/feedback/citation/missing', {
        method: 'POST',
        body: JSON.stringify(feedback),
    });
}

/**
 * Submit general feedback
 * @param {Object} feedback - Feedback data
 * @returns {Promise} - Promise that resolves to confirmation
 */
export async function submitGeneralFeedback(feedback) {
    return apiRequest('/api/feedback/general', {
        method: 'POST',
        body: JSON.stringify(feedback),
    });
}

// Export the API functions
export default {
    searchCases,
    getRecentCases,
    getCaseDetails,
    getCitationNetwork,
    submitMissingCitationFeedback,
    submitGeneralFeedback,
}; 