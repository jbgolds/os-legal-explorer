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
        type: 'o',
        highlight: 'on',
        order_by: '-dateFiled',
        ...filters,
    });

    // Convert date filters to API format
    if (filters.start_date) {
        queryParams.set('filed_after', filters.start_date);
        queryParams.delete('start_date');
    }
    if (filters.end_date) {
        queryParams.set('filed_before', filters.end_date);
        queryParams.delete('end_date');
    }

    // Handle cursor-based pagination
    if (filters.cursor) {
        queryParams.set('cursor', filters.cursor);
    }

    return apiRequest(`/api/search?${queryParams}`);
}


/**
 * Get details for a specific case
 * @param {string} clusterId - ID of the case to fetch
 * @returns {Promise<Object>} Case details
 */
export function getOpinionDetails(clusterId) {
    return apiRequest(`/api/opinion/${clusterId}`);
}

/**
 * Get citation network for a case
 * @param {string} clusterId - ID of the case
 * @param {string} direction - Direction of citation network ('outgoing' or 'incoming')
 * @returns {Promise<Object>} Citation network data
 */
export function getCitationNetwork(clusterId, direction = 'outgoing') {
    return apiRequest(`/api/opinion/${clusterId}/citation-network?direction=${direction}`);
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
    getOpinionDetails,
    getCitationNetwork,
    submitMissingCitationFeedback,
    submitGeneralFeedback,
}; 