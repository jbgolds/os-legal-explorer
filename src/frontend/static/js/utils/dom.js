/**
 * DOM utility functions for OS Legal Explorer
 * This module provides helper functions for DOM manipulation
 */

/**
 * Show an element by removing the 'hidden' class
 * @param {HTMLElement|string} element - Element or selector
 */
export function showElement(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.remove('hidden');
    }
}

/**
 * Hide an element by adding the 'hidden' class
 * @param {HTMLElement|string} element - Element or selector
 */
export function hideElement(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.add('hidden');
    }
}

/**
 * Toggle the visibility of an element
 * @param {HTMLElement|string} element - Element or selector
 */
export function toggleElement(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.toggle('hidden');
    }
}

/**
 * Create a loading indicator
 * @param {string} size - Size of the loading indicator (xs, sm, md, lg)
 * @param {string} text - Optional text to display
 * @returns {HTMLElement} - Loading indicator element
 */
export function createLoadingIndicator(size = 'md', text = 'Loading...') {
    const container = document.createElement('div');
    container.className = 'text-center py-4';

    const spinner = document.createElement('span');
    spinner.className = `loading loading-spinner loading-${size}`;

    container.appendChild(spinner);

    if (text) {
        const textEl = document.createElement('p');
        textEl.className = 'mt-2 text-gray-600';
        textEl.textContent = text;
        container.appendChild(textEl);
    }

    return container;
}

/**
 * Create an alert message
 * @param {string} message - Alert message
 * @param {string} type - Alert type (info, success, warning, error)
 * @returns {HTMLElement} - Alert element
 */
export function createAlert(message, type = 'info') {
    const alertClass = type === 'error' ? 'alert-error' :
        type === 'success' ? 'alert-success' :
            type === 'warning' ? 'alert-warning' :
                'alert-info';

    const container = document.createElement('div');
    container.className = `alert ${alertClass} my-4`;

    // Add appropriate icon based on type
    let iconSvg;
    if (type === 'error') {
        iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
    } else if (type === 'success') {
        iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';
    } else if (type === 'warning') {
        iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>';
    } else {
        iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
    }

    const iconContainer = document.createElement('div');
    iconContainer.innerHTML = iconSvg;

    const messageEl = document.createElement('span');
    messageEl.textContent = message;

    container.appendChild(iconContainer.firstChild);
    container.appendChild(messageEl);

    return container;
}

/**
 * Format a date string
 * @param {string} dateString - Date string to format
 * @param {Object} options - Formatting options
 * @returns {string} - Formatted date string
 */
export function formatDate(dateString, options = { year: 'numeric', month: 'long', day: 'numeric' }) {
    if (!dateString) return '';
    return new Date(dateString).toLocaleDateString(undefined, options);
}

/**
 * Debounce a function call
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
export function debounce(func, wait = 300) {
    let timeout;
    return function (...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// Export the DOM utility functions
export default {
    showElement,
    hideElement,
    toggleElement,
    createLoadingIndicator,
    createAlert,
    formatDate,
    debounce,
}; 