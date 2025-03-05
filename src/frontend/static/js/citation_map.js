/**
 * Citation Network Visualization
 * 
 * This module provides functions to visualize legal citation networks using D3.js.
 * It supports rendering nodes of different types (Opinion, Statute, etc.) and
 * citation relationships between them with visual cues for treatment and relevance.
 */

// Main function to render the citation network
function renderCitationNetwork(containerId, apiEndpoint, options = {}) {
    // Default options
    const defaults = {
        width: 800,
        height: 600,
        nodeRadius: 10,
        linkDistance: 200,        // Increased from 100 to 200 to spread nodes out more
        charge: -800,             // Increased from -300 to -800 for more repulsion
        treatmentColors: {
            'POSITIVE': '#4CAF50',   // Green
            'NEGATIVE': '#F44336',   // Red
            'NEUTRAL': '#9E9E9E',    // Gray
            'CAUTION': '#FF9800'     // Orange
        },
        typeColors: {
            'opinion_cluster': '#2196F3',  // Blue for opinions
            'statutes': '#9C27B0',         // Purple for statutes
            'constitution': '#E91E63',     // Pink for constitutional docs
            'default': '#607D8B'           // Default blue-gray
        },
        depthOpacity: {
            1: 1.0,   // First level fully opaque
            2: 0.7,   // Second level slightly transparent
            3: 0.4    // Third level more transparent
        },
        zoomExtent: [0.2, 5]      // Min and max zoom levels
    };

    // Merge provided options with defaults
    const config = { ...defaults, ...options };

    // Select the container element
    const container = d3.select(`#${containerId}`);

    // Get container dimensions
    const containerElement = document.getElementById(containerId);
    const containerRect = containerElement.parentElement.getBoundingClientRect();

    // Update width and height based on container size
    config.width = containerRect.width;
    config.height = Math.max(500, containerRect.height);

    // Show loading indicator
    container.html('<div class="network-loading">Loading citation network...</div>');

    // Fetch data from API
    fetch(apiEndpoint)
        .then(response => {
            if (!response.ok) {
                console.error(`API error: ${response.status} - ${response.statusText}`);
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Citation network data received: ${data.nodes.length} nodes, ${data.links.length} links`);

            // Clear container and create SVG
            container.html('');

            // If no data, show a message
            if (!data.nodes.length) {
                container.html('<div class="flex items-center justify-center h-full"><div class="text-center"><p class="text-xl font-bold">No Citation Data</p><p class="text-gray-500">No citation network data available for this document.</p></div></div>');
                return;
            }

            // Create SVG that fills the container
            const svg = container.append('svg')
                .attr('width', config.width)
                .attr('height', config.height)
                .attr('class', 'citation-network-svg');

            // Create definitions for markers (arrows) - at SVG level so they can be used by both network and legend
            const defs = svg.append('defs');

            // Add arrow markers for different treatments
            Object.entries(config.treatmentColors).forEach(([treatment, color]) => {
                defs.append('marker')
                    .attr('id', `arrow-${treatment.toLowerCase()}`)
                    .attr('viewBox', '0 -5 10 10')
                    .attr('refX', 15)
                    .attr('refY', 0)
                    .attr('markerWidth', 6)
                    .attr('markerHeight', 6)
                    .attr('orient', 'auto')
                    .append('path')
                    .attr('d', 'M0,-5L10,0L0,5')
                    .attr('fill', color);
            });

            // Create default arrow marker
            defs.append('marker')
                .attr('id', 'arrow-default')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 15)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', '#666');

            // Create a group for all network elements that will be transformed during zoom
            const g = svg.append('g')
                .attr('class', 'network-container');

            // Set up the simulation
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links)
                    .id(d => d.id)
                    .distance(config.linkDistance))
                .force('charge', d3.forceManyBody().strength(config.charge))
                .force('center', d3.forceCenter(config.width / 2, config.height / 2));

            // Create links
            const link = g.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(data.links)
                .enter().append('line')
                .attr('stroke-width', d => {
                    // Scale up the relevance value for better visibility
                    if (d.relevance) {
                        // Scale from 1-5 to 1-8 for better visibility
                        return Math.min(1 + (d.relevance * 1.5), 8);
                    }
                    return 1; // Default width
                })
                .attr('stroke', d => {
                    if (d.treatment && config.treatmentColors[d.treatment]) {
                        return config.treatmentColors[d.treatment];
                    }
                    return '#666'; // Default gray
                })
                .attr('opacity', d => {
                    if (d.metadata && d.metadata.depth && config.depthOpacity[d.metadata.depth]) {
                        return config.depthOpacity[d.metadata.depth];
                    }
                    return 0.6; // Default opacity
                })
                .attr('marker-end', d => {
                    if (d.treatment && config.treatmentColors[d.treatment]) {
                        return `url(#arrow-${d.treatment.toLowerCase()})`;
                    }
                    return 'url(#arrow-default)';
                });

            // Create nodes
            const node = g.append('g')
                .attr('class', 'nodes')
                .selectAll('circle')
                .data(data.nodes)
                .enter().append('circle')
                .attr('r', config.nodeRadius)
                .attr('fill', d => {
                    // Determine color based on document type
                    const docType = d.type.toLowerCase();
                    for (const [type, color] of Object.entries(config.typeColors)) {
                        if (docType.includes(type)) {
                            return color;
                        }
                    }
                    return config.typeColors.default;
                })
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Add node labels
            const nodeLabels = g.append('g')
                .attr('class', 'node-labels')
                .selectAll('text')
                .data(data.nodes)
                .enter().append('text')
                .attr('dx', 12)
                .attr('dy', 4)
                .text(d => {
                    // Truncate long labels
                    const label = d.citation_string || '';
                    return label.length > 25 ? label.substring(0, 22) + '...' : label;
                });

            // Add tooltips
            node.append('title')
                .text(d => {
                    let tooltip = `${d.citation_string || 'Unknown'}\n`;
                    tooltip += `Type: ${d.type || 'Unknown'}\n`;

                    if (d.court) {
                        tooltip += `Court: ${d.court}\n`;
                    }

                    if (d.year) {
                        tooltip += `Year: ${d.year}\n`;
                    }

                    if (d.metadata) {
                        if (d.metadata.citation) {
                            tooltip += `Citation: ${d.metadata.citation}\n`;
                        }
                        if (d.metadata.docket_number) {
                            tooltip += `Docket: ${d.metadata.docket_number}\n`;
                        }
                    }

                    return tooltip;
                });

            // Link tooltips
            link.append('title')
                .text(d => {
                    let tooltip = `${d.source.citation_string || d.source} → ${d.target.citation_string || d.target}\n`;

                    if (d.treatment) {
                        tooltip += `Treatment: ${d.treatment}\n`;
                    }

                    if (d.relevance) {
                        tooltip += `Relevance: ${d.relevance}\n`;
                    }

                    if (d.metadata) {
                        if (d.metadata.citation_text) {
                            tooltip += `Citation: ${d.metadata.citation_text}\n`;
                        }

                        if (d.metadata.reasoning) {
                            tooltip += `Reasoning: ${d.metadata.reasoning}\n`;
                        }
                    }

                    return tooltip;
                });

            // Define zoom behavior
            const zoom = d3.zoom()
                .scaleExtent(config.zoomExtent)
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });

            // Apply zoom behavior to SVG
            svg.call(zoom);

            // Update node and link positions on simulation tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                nodeLabels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });

            // Functions for drag behavior
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            // Add legend - OUTSIDE the zoom group so it stays static
            const legend = svg.append('g')
                .attr('class', 'legend')
                .attr('transform', 'translate(10, 20)');

            // Document type legend
            const typeLegend = legend.append('g').attr('class', 'type-legend');
            let yPos = 0;

            Object.entries(config.typeColors).forEach(([type, color], i) => {
                const legendItem = typeLegend.append('g')
                    .attr('transform', `translate(0, ${yPos})`);

                legendItem.append('circle')
                    .attr('r', 6)
                    .attr('fill', color);

                legendItem.append('text')
                    .attr('x', 12)
                    .attr('y', 4)
                    .text(type.charAt(0).toUpperCase() + type.slice(1));

                yPos += 20;
            });

            // Treatment legend
            const treatmentLegend = legend.append('g')
                .attr('class', 'treatment-legend')
                .attr('transform', `translate(150, 0)`);

            yPos = 0;
            Object.entries(config.treatmentColors).forEach(([treatment, color], i) => {
                const legendItem = treatmentLegend.append('g')
                    .attr('transform', `translate(0, ${yPos})`);

                legendItem.append('line')
                    .attr('x1', 0)
                    .attr('y1', 0)
                    .attr('x2', 20)
                    .attr('y2', 0)
                    .attr('stroke', color)
                    .attr('stroke-width', 2)
                    .attr('marker-end', `url(#arrow-${treatment.toLowerCase()})`);

                legendItem.append('text')
                    .attr('x', 25)
                    .attr('y', 4)
                    .text(treatment);

                yPos += 20;
            });

            // Store the simulation and zoom in a global namespace for use with zoom controls
            window.citationNetworkState = window.citationNetworkState || {};
            window.citationNetworkState[containerId] = {
                svg: svg,
                simulation: simulation,
                zoom: zoom
            };
        })
        .catch(error => {
            console.error('Error fetching or rendering citation network:', error);
            container.html(`
                <div class="flex items-center justify-center h-full">
                    <div class="text-center">
                        <p class="text-xl font-bold text-error">Error Loading Citation Network</p>
                        <p class="text-gray-500">${error.message || 'Failed to load citation network data'}</p>
                        <button class="btn btn-sm btn-outline mt-4" onclick="location.reload()">Retry</button>
                    </div>
                </div>
            `);
        });
}

// Export the function
window.renderCitationNetwork = renderCitationNetwork; 