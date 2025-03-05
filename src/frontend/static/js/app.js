/**
 * Main application JavaScript for OS Legal Explorer
 * This file handles the core functionality of the single-page application
 */

import { getOpinionDetails, getCitationNetwork } from './utils/api.js';
import { createLoadingIndicator, createAlert, formatDate } from './utils/dom.js';

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeCaseSelection();
});

/**
 * Initialize case selection functionality
 */
function initializeCaseSelection() {
    // Event delegation for case selection
    document.body.addEventListener('click', (e) => {
        const caseLink = e.target.closest('[data-case-id]');
        if (caseLink) {
            e.preventDefault();
            const clusterId = caseLink.dataset.clusterId;
            loadClusterDetails(clusterId);
        }
    });
}

/**
 * Load and display case details
 * @param {string} clusterId - The ID of the case to load
 */
async function loadClusterDetails(clusterId) {
    const caseDetail = document.getElementById('cluster-detail');
    if (!caseDetail) return;

    try {
        // Show loading state
        caseDetail.innerHTML = `
            <div class="card-body p-6">
                <div class="text-center py-8">
                    <span class="loading loading-spinner loading-lg"></span>
                    <p class="mt-4">Loading case details...</p>
                </div>
            </div>
        `;

        // Fetch both case details and citation network data in parallel using utility functions
        const [caseData, networkData] = await Promise.all([
            getOpinionDetails(clusterId),
            getCitationNetwork(clusterId).catch(err => {
                console.warn('Failed to load citation network:', err);
                return null;
            })
        ]);

        // Create case header
        const header = document.createElement('div');
        header.className = 'mb-6 pb-4 border-b';
        header.innerHTML = `
            <h2 class="text-2xl font-bold">${caseData.name}</h2>
            <div class="flex flex-wrap gap-2 mt-2">
                <span class="badge badge-primary">${caseData.court_name || 'Unknown Court'}</span>
                <span class="badge badge-secondary">${formatDate(caseData.date_filed)}</span>
                ${caseData.citation ? `<span class="badge badge-outline">${caseData.citation}</span>` : ''}
            </div>
        `;

        // Create tabs for opinion text and citation map
        const tabs = document.createElement('div');
        tabs.className = 'tabs tabs-boxed mb-4';
        tabs.innerHTML = `
            <a class="tab tab-active" data-tab="opinion">Opinion Text</a>
            <a class="tab" data-tab="citation-map">Citation Network</a>
        `;

        // Create tab content container
        const tabContent = document.createElement('div');
        tabContent.className = 'tab-content';

        // Opinion text content
        const opinionContent = document.createElement('div');
        opinionContent.id = 'opinion-content';
        opinionContent.className = 'prose max-w-none';

        if (caseData.opinion_text) {
            opinionContent.innerHTML = `
                <div class="mockup-browser border border-base-300">
                    <div class="mockup-browser-toolbar">
                        <div class="input border border-base-300">${caseData.name}</div>
                    </div>
                    <div class="px-4 py-8 border-t border-base-300 bg-base-200">
                        <div class="whitespace-pre-wrap font-serif leading-relaxed">
                            ${caseData.opinion_text}
                        </div>
                    </div>
                </div>
            `;
        } else {
            opinionContent.innerHTML = `
                <div class="alert alert-warning">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span>No opinion text available for this case.</span>
                </div>
            `;
        }

        tabContent.appendChild(opinionContent);

        // Citation map content
        const citationContent = document.createElement('div');
        citationContent.id = 'citation-map-content';
        citationContent.className = 'hidden';
        citationContent.innerHTML = `
            <div id="citation-map" class="h-[500px] border rounded-lg"></div>
        `;
        tabContent.appendChild(citationContent);

        // Clear and update case detail
        caseDetail.innerHTML = '';
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';
        cardBody.appendChild(header);
        cardBody.appendChild(tabs);
        cardBody.appendChild(tabContent);
        caseDetail.appendChild(cardBody);

        // If we have network data, initialize the citation map
        if (networkData) {
            // Load and initialize the citation network visualization
            await import('./citation_map.js').then(module => {
                if (typeof module.renderCitationNetwork === 'function') {
                    module.renderCitationNetwork(networkData, 'citation-map');
                    citationContent.dataset.loaded = 'true';
                }
            });
        }

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
                    // Call loadCitationNetwork when the citation map tab is activated
                    loadCitationNetwork(clusterId);
                }
            });
        });

    } catch (error) {
        console.error('Error loading case details:', error);
        caseDetail.innerHTML = `
            <div class="card-body p-6">
                <div class="alert alert-error">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Error loading case details. Please try again later.</span>
                </div>
            </div>
        `;
    }
}

// Add a function to process a cluster via the new API endpoint
async function processCluster(clusterId) {
    try {
        const response = await fetch(`/api/pipeline/process-cluster/${clusterId}`, { method: 'POST' });
        if (!response.ok) {
            throw new Error('Failed to trigger cluster processing');
        }
        const data = await response.json();
        createAlert('Cluster processing started.', 'success');
    } catch (e) {
        createAlert(e.message, 'error');
    }
}

// Modify loadCitationNetwork to show a 'Process Cluster' button when loading fails
function loadCitationNetwork(caseId) {
    const citationMap = document.getElementById('citation-map');
    if (!citationMap) return;

    // Check if citation map is already loaded
    if (citationMap.dataset.loaded === caseId) return;

    // Set height for the visualization
    citationMap.style.height = '500px';

    // Load and render the citation network
    import('./citation_map.js')
        .then(async module => {
            if (typeof module.renderCitationNetwork === 'function') {
                try {
                    // Use the API utility function to get network data
                    const networkData = await getCitationNetwork(caseId);
                    module.renderCitationNetwork(networkData, 'citation-map', { clusterId: caseId });
                    citationMap.dataset.loaded = caseId;
                } catch (error) {
                    console.error('Error fetching citation network data:', error);
                    citationMap.innerHTML = `
                        <div class="flex justify-center items-center h-full">
                            <div class="text-center">
                                <p class="text-xl font-bold">Failed to load citation network data.</p>
                                <p class="text-gray-500">${error.message || 'No data available.'}</p>
                                <button class="btn btn-sm btn-outline mt-4" onclick="processCluster(${caseId})">Process Cluster</button>
                            </div>
                        </div>
                    `;
                }
            } else {
                console.error('Citation map module does not export renderCitationNetwork function');
            }
        })
        .catch(error => {
            console.error('Error loading citation map module:', error);
            citationMap.innerHTML = `
                <div class="flex justify-center items-center h-full">
                    <div class="text-center">
                        <p class="text-xl font-bold">Failed to load citation network visualization.</p>
                        <p class="text-gray-500">${error.message || 'Visualization error.'}</p>
                        <button class="btn btn-sm btn-outline mt-4" onclick="processCluster(${caseId})">Process Cluster</button>
                    </div>
                </div>
            `;
        });
}

// Check if we're on a case detail page and load the case
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    const match = path.match(/^\/opinion\/(\d+)\/?$/);
    if (match) {
        const caseId = match[1];
        loadClusterDetails(caseId);
    }
}); 