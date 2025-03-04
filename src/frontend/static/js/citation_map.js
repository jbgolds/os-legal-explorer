/**
 * Citation Map Visualization
 * 
 * This module provides functions for rendering a citation network visualization
 * using D3.js. It creates an interactive force-directed graph showing how cases
 * cite each other, with color coding for different citation treatments.
 */

/**
 * Renders a citation network visualization using D3.js
 * 
 * @param {Object} data - The citation network data
 * @param {Array} data.nodes - Array of node objects representing cases
 * @param {Array} data.edges - Array of edge objects representing citations
 * @param {string} containerId - The ID of the container element for the visualization
 * @param {string} currentCaseId - The ID of the current case being viewed
 */
function renderCitationNetwork(data, containerId = 'citation-map', currentCaseId = null) {
    // Clear any existing content
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    // Create SVG element
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    const svg = d3.create('svg')
        .attr('width', width)
        .attr('height', height)
        .style('z-index', '1') // Add z-index to keep it behind other elements
        .style('position', 'relative'); // Ensure z-index works properly

    // Add a clip path to contain all elements
    svg.append('defs')
        .append('clipPath')
        .attr('id', 'citation-map-clip')
        .append('rect')
        .attr('width', width)
        .attr('height', height);

    // Create a group for zoom/pan and apply the clip path
    const g = svg.append('g')
        .attr('clip-path', 'url(#citation-map-clip)');

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Create the force simulation
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(40));

    // Create the links
    const link = g.append('g')
        .selectAll('line')
        .data(data.edges)
        .join('line')
        .attr('stroke-width', 2)
        .attr('stroke', d => getTreatmentColor(d.treatment));

    // Add nodes
    const node = g.append('g')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .selectAll('circle')
        .data(data.nodes)
        .join('circle')
        .attr('r', d => d.id === currentCaseId ? 8 : 5)
        .attr('fill', d => d.id === currentCaseId ? '#ff6b6b' : '#4299e1')
        .style('filter', 'none') // Remove any filter effects
        .style('opacity', 0.8) // Make slightly transparent
        .call(drag(simulation));

    // Add node labels
    const label = g.append('g')
        .selectAll('text')
        .data(data.nodes)
        .join('text')
        .attr('dy', -15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .text(d => d.name)
        .attr('pointer-events', 'none');

    // Add titles to nodes
    node.append('title')
        .text(d => d.name);

    // Add the SVG to the container
    container.appendChild(svg.node());

    // Update positions on each tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });

    // Drag behavior
    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
}

/**
 * Load and render the citation network for a case
 * @param {string} caseId - ID of the case
 * @param {string} containerId - ID of the container element
 */
export async function loadAndRenderCitationNetwork(caseId, containerId) {
    try {
        // Show loading state
        const container = document.getElementById(containerId);
        if (!container) return;

        // Fetch citation network data
        const response = await fetch(`/api/opinion/${caseId}/citation-network`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Render the network
        renderCitationNetwork(data, containerId);
    } catch (error) {
        console.error('Error loading citation network:', error);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="alert alert-error">
                    <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Error loading citation network. Please try again later.</span>
                </div>
            `;
        }
    }
}

/**
 * Returns the color for a citation treatment
 * 
 * @param {string} treatment - The citation treatment (positive, negative, cautionary, neutral)
 * @returns {string} - The color for the treatment
 */
function getTreatmentColor(treatment) {
    switch (treatment) {
        case 'positive':
            return '#22c55e'; // green-500
        case 'negative':
            return '#ef4444'; // red-500
        case 'cautionary':
        case 'caution':
            return '#eab308'; // yellow-500
        default:
            return '#6b7280'; // gray-500
    }
}

/**
 * Returns the CSS class for a citation treatment
 * 
 * @param {string} treatment - The citation treatment (positive, negative, cautionary, neutral)
 * @returns {string} - The CSS class for the treatment
 */
function getCitationTreatmentClass(treatment) {
    switch (treatment) {
        case 'positive':
            return 'text-green-600';
        case 'negative':
            return 'text-red-600';
        case 'cautionary':
        case 'caution':
            return 'text-yellow-600';
        default:
            return 'text-gray-600';
    }
}

/**
 * Formats a citation treatment for display
 * 
 * @param {string} treatment - The citation treatment (positive, negative, cautionary, neutral)
 * @returns {string} - The formatted treatment text
 */
function formatCitationTreatment(treatment) {
    if (!treatment) return '';

    switch (treatment) {
        case 'positive':
            return 'Positive Citation';
        case 'negative':
            return 'Negative Citation';
        case 'cautionary':
        case 'caution':
            return 'Cautionary Citation';
        default:
            return 'Neutral Citation';
    }
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        renderCitationNetwork,
        loadAndRenderCitationNetwork,
        getTreatmentColor,
        getCitationTreatmentClass,
        formatCitationTreatment
    };
} 