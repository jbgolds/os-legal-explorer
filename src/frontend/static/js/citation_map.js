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
        linkDistance: 250,        // Increased from 100 to 200 to spread nodes out more
        charge: -1600,            // Reduced from -2000 to -1000 for better balance with clustering
        enableClustering: true,   // Enable clustering by default
        clusteringStrength: 0.6,  // Increased from 0.4 to 0.6 for stronger type grouping
        treatmentColors: {
            'POSITIVE': '#4CAF50',   // Green
            'NEGATIVE': '#F44336',   // Red
            'NEUTRAL': '#9E9E9E',    // Gray
            'CAUTION': '#FF9800'     // Orange
        },
        typeColors: {
            'judicial_opinion': '#2196F3',           // Blue
            'statutes_codes_regulations': '#9C27B0', // Purple
            'constitution': '#E91E63',               // Pink
            'administrative_agency_ruling': '#FF5722', // Deep Orange
            'congressional_report': '#FFC107',       // Amber
            'external_submission': '#8BC34A',        // Light Green
            'electronic_resource': '#00BCD4',        // Cyan
            'law_review': '#3F51B5',                 // Indigo
            'legal_dictionary': '#009688',           // Teal
            'other': '#607D8B'                       // Blue Gray for any unrecognized type
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

            // Helper function to consistently determine node type
            function getNodeType(node) {
                const docType = node.type.toLowerCase();

                // Map API document types to our color configuration types
                const typeMapping = {
                    'opinion_cluster': 'judicial_opinion',
                    'statutes': 'statutes_codes_regulations',
                    'constitutional_documents': 'constitution',
                    'admin_rulings': 'administrative_agency_ruling',
                    'congressional_reports': 'congressional_report',
                    'submissions': 'external_submission',
                    'law_reviews': 'law_review',
                    'other_legal_documents': 'other'
                };

                // First try mapping the API type to our color configuration
                if (typeMapping[docType] && config.typeColors[typeMapping[docType]]) {
                    return typeMapping[docType];
                }

                // Then try exact match
                if (config.typeColors[docType]) {
                    return docType;
                }

                // If no exact match, try partial match for backward compatibility
                for (const type of Object.keys(config.typeColors)) {
                    if (type !== 'other' && docType.includes(type)) {
                        return type;
                    }
                }

                // Default to 'other' if no match found
                return 'other';
            }

            // Set initial positions in a circle to help the simulation converge faster
            const centerX = config.width / 2;
            const centerY = config.height / 2;
            const radius = Math.min(config.width, config.height) * 0.35;

            // Group nodes by type
            const nodesByType = {};
            data.nodes.forEach(node => {
                const type = getNodeType(node);
                if (!nodesByType[type]) {
                    nodesByType[type] = [];
                }
                nodesByType[type].push(node);
            });

            // Position each type group at a different angle
            const typeCount = Object.keys(nodesByType).length;
            let typeIndex = 0;

            Object.entries(nodesByType).forEach(([type, nodes]) => {
                const typeAngle = (typeIndex / typeCount) * 2 * Math.PI;
                const typeX = centerX + radius * Math.cos(typeAngle);
                const typeY = centerY + radius * Math.sin(typeAngle);

                // Position nodes in a small cluster around their type's position
                nodes.forEach((node, i) => {
                    const nodeRadius = 50; // Small radius for the node cluster
                    const nodeAngle = (i / nodes.length) * 2 * Math.PI;
                    node.x = typeX + nodeRadius * Math.cos(nodeAngle);
                    node.y = typeY + nodeRadius * Math.sin(nodeAngle);
                });

                typeIndex++;
            });

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
                    .attr('refX', 25)
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
                .attr('refX', 25)
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
                .force('center', d3.forceCenter(config.width / 2, config.height / 2))
                .force('radial', d3.forceRadial(
                    Math.min(config.width, config.height) * 0.35, // Same radius as in forceCluster
                    config.width / 2,
                    config.height / 2
                ).strength(0.05)); // Weak radial force to maintain overall structure

            // Add clustering force only if enabled
            if (config.enableClustering) {
                simulation.force('cluster', forceCluster()
                    .strength(config.clusteringStrength)
                    .nodes(data.nodes)
                    .getType(d => getNodeType(d)));
            }

            // Custom force to cluster nodes of the same type
            function forceCluster() {
                let nodes;
                let strength = 0.1;
                let getType;
                let centers = {};
                let typeAngles = {}; // Store the angle for each type
                let radius; // Radius for the radial distribution

                function force(alpha) {
                    // Set radius based on the simulation area
                    radius = Math.min(config.width, config.height) * 0.35; // Use 35% of the smaller dimension

                    // First, identify all unique types
                    const types = new Set();
                    nodes.forEach(d => {
                        types.add(getType(d));
                    });

                    // Assign an angle to each type if not already assigned
                    if (Object.keys(typeAngles).length === 0) {
                        const typeArray = Array.from(types);
                        const angleStep = (2 * Math.PI) / typeArray.length;

                        typeArray.forEach((type, i) => {
                            typeAngles[type] = i * angleStep;
                        });
                    }

                    // Calculate the center of mass for each type
                    centers = {};
                    nodes.forEach(d => {
                        const type = getType(d);
                        if (!centers[type]) {
                            centers[type] = { x: 0, y: 0, count: 0 };
                        }
                        centers[type].x += d.x;
                        centers[type].y += d.y;
                        centers[type].count += 1;
                    });

                    // Calculate the average position for each type
                    Object.keys(centers).forEach(type => {
                        if (centers[type].count > 0) {
                            centers[type].x /= centers[type].count;
                            centers[type].y /= centers[type].count;
                        }
                    });

                    // Get the center of the simulation
                    const centerX = config.width / 2;
                    const centerY = config.height / 2;

                    // Move nodes toward their type's position on the circle
                    nodes.forEach(d => {
                        const type = getType(d);
                        if (centers[type]) {
                            // Calculate the target position on the circle for this type
                            const angle = typeAngles[type];
                            const targetX = centerX + radius * Math.cos(angle);
                            const targetY = centerY + radius * Math.sin(angle);

                            // Move toward the type's target position on the circle
                            d.vx += (targetX - d.x) * strength * alpha;
                            d.vy += (targetY - d.y) * strength * alpha;

                            // Also move toward the center of mass of the same type nodes
                            // This creates the clustering effect within each type
                            if (centers[type].count > 1) {
                                const centerForce = 0.3; // Reduced from 1.0 to allow more spread
                                d.vx += (centers[type].x - d.x) * centerForce * strength * alpha;
                                d.vy += (centers[type].y - d.y) * centerForce * strength * alpha;
                            }
                        }
                    });
                }

                force.initialize = function (_) {
                    nodes = _;
                };

                force.strength = function (_) {
                    return arguments.length ? (strength = _, force) : strength;
                };

                force.getType = function (_) {
                    return arguments.length ? (getType = _, force) : getType;
                };

                force.nodes = function (_) {
                    return arguments.length ? (nodes = _, force) : nodes;
                };

                return force;
            }

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
                .selectAll('g') // Changed from 'text' to 'g' to group text and background
                .data(data.nodes)
                .enter().append('g'); // Create a group for each node

            // Add a background rectangle for each text node
            node.append('rect')
                .attr('rx', 5) // Rounded corners
                .attr('ry', 5)
                .attr('fill', 'white')
                .attr('fill-opacity', 0.7) // Semi-transparent
                .attr('stroke', d => {
                    const docType = getNodeType(d);
                    return config.typeColors[docType] || config.typeColors.other;
                })
                .attr('stroke-width', 2);

            // Add the text on top of the background
            const nodeText = node.append('text')
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'middle')
                .text(d => {
                    // Truncate long labels
                    const label = d.citation_string || '';
                    return label.length > 25 ? label.substring(0, 22) + '...' : label;
                })
                .attr('fill', d => {
                    const docType = getNodeType(d);
                    return config.typeColors[docType] || config.typeColors.other;
                })
                .style('font-weight', 'bold')
                .style('font-size', '12px');

            // Calculate and set the background rectangle size based on text content
            node.each(function () {
                const text = d3.select(this).select('text');
                const textBBox = text.node().getBBox();
                const padding = 5;

                d3.select(this).select('rect')
                    .attr('x', -textBBox.width / 2 - padding)
                    .attr('y', -textBBox.height / 2 - padding)
                    .attr('width', textBBox.width + (padding * 2))
                    .attr('height', textBBox.height + (padding * 2));
            });

            // Add interaction behaviors to the group
            node.style('cursor', 'pointer')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('click', function (event, d) {
                    event.stopPropagation(); // Prevent click from propagating to SVG
                    showNodeDetails(d, event);
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

            // Add click event to links
            link.on('click', function (event, d) {
                event.stopPropagation(); // Prevent click from propagating to SVG
                showLinkDetails(d, event);
            });

            // Add click handler to SVG to close any open details panel
            svg.on('click', function () {
                // Close any open details panel
                hideDetailsPanel();
            });

            // Function to show node details
            function showNodeDetails(node, event) {
                // Remove any existing details panel
                hideDetailsPanel();

                // Create details panel
                const detailsPanel = container.append('div')
                    .attr('class', 'network-details-panel')
                    .style('position', 'absolute')
                    .style('top', '10px')
                    .style('right', '10px')
                    .style('background', 'white')
                    .style('border', '1px solid #ddd')
                    .style('border-radius', '4px')
                    .style('padding', '12px')
                    .style('box-shadow', '0 2px 10px rgba(0,0,0,0.1)')
                    .style('z-index', '1000')
                    .style('max-width', '350px')
                    .style('max-height', '80%')
                    .style('overflow-y', 'auto');

                // Add close button
                detailsPanel.append('div')
                    .style('text-align', 'right')
                    .style('margin-bottom', '8px')
                    .append('button')
                    .attr('class', 'details-close-btn')
                    .style('background', 'none')
                    .style('border', 'none')
                    .style('cursor', 'pointer')
                    .style('font-size', '16px')
                    .text('×')
                    .on('click', function () {
                        hideDetailsPanel();
                    });

                // Add title
                detailsPanel.append('h3')
                    .style('margin', '0 0 10px 0')
                    .style('font-size', '16px')
                    .style('font-weight', 'bold')
                    .text('Document Details');

                // Create content container
                const content = detailsPanel.append('div');

                // Add node information
                content.append('p')
                    .style('font-weight', 'bold')
                    .style('margin', '10px 0 5px 0')
                    .text(node.citation_string || 'Unknown Document');

                // Add type with colored indicator
                const typeRow = content.append('div')
                    .style('display', 'flex')
                    .style('align-items', 'center')
                    .style('margin-bottom', '5px');

                const docType = getNodeType(node);
                let typeColor = config.typeColors[docType] || config.typeColors.other;

                typeRow.append('span')
                    .style('display', 'inline-block')
                    .style('width', '12px')
                    .style('height', '12px')
                    .style('border-radius', '50%')
                    .style('background-color', typeColor)
                    .style('margin-right', '8px');

                typeRow.append('span')
                    .text(`Type: ${node.type || 'Unknown'}`);

                // Add other basic information
                if (node.court) {
                    content.append('p')
                        .style('margin', '5px 0')
                        .text(`Court: ${node.court}`);
                }

                if (node.year) {
                    content.append('p')
                        .style('margin', '5px 0')
                        .text(`Year: ${node.year}`);
                }

                // Add metadata section if available
                if (node.metadata) {
                    content.append('h4')
                        .style('margin', '15px 0 5px 0')
                        .style('font-size', '14px')
                        .style('font-weight', 'bold')
                        .text('Additional Information');

                    const metadataList = content.append('dl')
                        .style('margin', '0')
                        .style('padding', '0');

                    // Add all available metadata
                    Object.entries(node.metadata).forEach(([key, value]) => {
                        if (value && typeof value !== 'object') {
                            const formattedKey = key.replace(/_/g, ' ')
                                .split(' ')
                                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                .join(' ');

                            metadataList.append('dt')
                                .style('font-weight', 'bold')
                                .style('margin-top', '8px')
                                .text(formattedKey);

                            metadataList.append('dd')
                                .style('margin-left', '0')
                                .style('margin-bottom', '5px')
                                .text(value);
                        }
                    });
                }

                // Add link to view full document if ID is available
                if (node.id) {
                    content.append('div')
                        .style('margin-top', '15px')
                        .append('a')
                        .attr('href', `/opinion/${node.id}`)
                        .attr('target', '_blank')
                        .style('color', '#2196F3')
                        .style('text-decoration', 'none')
                        .text('View Full Document →');
                }
            }

            // Function to show link details
            function showLinkDetails(link, event) {
                // Remove any existing details panel
                hideDetailsPanel();

                // Create details panel
                const detailsPanel = container.append('div')
                    .attr('class', 'network-details-panel')
                    .style('position', 'absolute')
                    .style('top', '10px')
                    .style('right', '10px')
                    .style('background', 'white')
                    .style('border', '1px solid #ddd')
                    .style('border-radius', '4px')
                    .style('padding', '12px')
                    .style('box-shadow', '0 2px 10px rgba(0,0,0,0.1)')
                    .style('z-index', '1000')
                    .style('max-width', '350px')
                    .style('max-height', '80%')
                    .style('overflow-y', 'auto');

                // Add close button
                detailsPanel.append('div')
                    .style('text-align', 'right')
                    .style('margin-bottom', '8px')
                    .append('button')
                    .attr('class', 'details-close-btn')
                    .style('background', 'none')
                    .style('border', 'none')
                    .style('cursor', 'pointer')
                    .style('font-size', '16px')
                    .text('×')
                    .on('click', function () {
                        hideDetailsPanel();
                    });

                // Add title
                detailsPanel.append('h3')
                    .style('margin', '0 0 10px 0')
                    .style('font-size', '16px')
                    .style('font-weight', 'bold')
                    .text('Citation Relationship');

                // Create content container
                const content = detailsPanel.append('div');

                // Add citation relationship
                const relationshipRow = content.append('div')
                    .style('display', 'flex')
                    .style('align-items', 'center')
                    .style('margin-bottom', '10px');

                // Source document
                relationshipRow.append('span')
                    .style('font-weight', 'bold')
                    .text(link.source.citation_string || link.source);

                // Arrow with treatment color
                const arrowColor = link.treatment && config.treatmentColors[link.treatment]
                    ? config.treatmentColors[link.treatment]
                    : '#666';

                relationshipRow.append('span')
                    .style('margin', '0 8px')
                    .style('color', arrowColor)
                    .style('font-weight', 'bold')
                    .text('→');

                // Target document
                relationshipRow.append('span')
                    .style('font-weight', 'bold')
                    .text(link.target.citation_string || link.target);

                // Add treatment if available
                if (link.treatment) {
                    const treatmentRow = content.append('div')
                        .style('margin', '10px 0')
                        .style('display', 'flex')
                        .style('align-items', 'center');

                    treatmentRow.append('span')
                        .style('display', 'inline-block')
                        .style('width', '12px')
                        .style('height', '3px')
                        .style('background-color', arrowColor)
                        .style('margin-right', '8px');

                    treatmentRow.append('span')
                        .text(`Treatment: ${link.treatment}`);
                }

                // Add relevance if available
                if (link.relevance) {
                    content.append('p')
                        .style('margin', '5px 0')
                        .text(`Relevance: ${link.relevance}`);
                }

                // Add metadata section if available
                if (link.metadata) {
                    content.append('h4')
                        .style('margin', '15px 0 5px 0')
                        .style('font-size', '14px')
                        .style('font-weight', 'bold')
                        .text('Citation Details');

                    // Citation text
                    if (link.metadata.citation_text) {
                        content.append('div')
                            .style('margin', '10px 0')
                            .style('padding', '8px')
                            .style('background', '#f5f5f5')
                            .style('border-left', '3px solid ' + arrowColor)
                            .style('font-style', 'italic')
                            .text(link.metadata.citation_text);
                    }

                    // Reasoning
                    if (link.metadata.reasoning) {
                        content.append('p')
                            .style('margin', '10px 0 5px 0')
                            .style('font-weight', 'bold')
                            .text('Reasoning:');

                        content.append('p')
                            .style('margin', '0')
                            .text(link.metadata.reasoning);
                    }

                    // Add any other metadata
                    Object.entries(link.metadata).forEach(([key, value]) => {
                        if (value && typeof value !== 'object' &&
                            key !== 'citation_text' && key !== 'reasoning') {

                            const formattedKey = key.replace(/_/g, ' ')
                                .split(' ')
                                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                .join(' ');

                            content.append('p')
                                .style('margin', '5px 0')
                                .text(`${formattedKey}: ${value}`);
                        }
                    });
                }

                // Add links to view the source and target documents
                const linksSection = content.append('div')
                    .style('margin-top', '15px')
                    .style('display', 'flex')
                    .style('justify-content', 'space-between');

                if (link.source.id) {
                    linksSection.append('a')
                        .attr('href', `/opinion/${link.source.id}`)
                        .attr('target', '_blank')
                        .style('color', '#2196F3')
                        .style('text-decoration', 'none')
                        .text('View Citing Document');
                }

                if (link.target.id) {
                    linksSection.append('a')
                        .attr('href', `/opinion/${link.target.id}`)
                        .attr('target', '_blank')
                        .style('color', '#2196F3')
                        .style('text-decoration', 'none')
                        .text('View Cited Document');
                }
            }

            // Function to hide details panel
            function hideDetailsPanel() {
                container.selectAll('.network-details-panel').remove();
            }

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
                    .attr('transform', d => `translate(${d.x}, ${d.y})`);
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

            // Common styling variables
            const itemHeight = 24; // Increased height between items
            const iconSize = 10; // Size for all icons (circles and arrows)
            const textOffset = 20; // Consistent text offset

            // Add a title for the document types legend
            legend.append('text')
                .attr('x', 0)
                .attr('y', 0)
                .attr('font-weight', 'bold')
                .text('Document Types:');

            // Always show all document types in the legend
            const typeEntries = Object.entries(config.typeColors);

            // Document type legend
            let yPos = 15; // Start below the title
            typeEntries.forEach(([type, color], i) => {
                const legendItem = legend.append('g')
                    .attr('transform', `translate(0, ${yPos})`);

                // Draw colored circle
                legendItem.append('circle')
                    .attr('r', iconSize / 2)
                    .attr('cx', iconSize / 2)
                    .attr('cy', 0)
                    .attr('fill', color);

                // Format the type name for better readability
                let displayName = type.split('_').map(word =>
                    word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ');

                legendItem.append('text')
                    .attr('x', textOffset)
                    .attr('y', 4)
                    .text(displayName);

                yPos += itemHeight;
            });

            // Add a title for the treatment legend
            legend.append('text')
                .attr('x', 0)
                .attr('y', yPos + 10)
                .attr('font-weight', 'bold')
                .text('Citation Treatments:');

            yPos += 25; // Extra space after title

            // Treatment legend
            Object.entries(config.treatmentColors).forEach(([treatment, color], i) => {
                const legendItem = legend.append('g')
                    .attr('transform', `translate(0, ${yPos})`);

                // Create a colored arrow
                const arrowGroup = legendItem.append('g')
                    .attr('transform', `translate(${iconSize / 2}, 0)`);

                // Arrow line
                arrowGroup.append('line')
                    .attr('x1', -iconSize / 2)
                    .attr('y1', 0)
                    .attr('x2', iconSize / 2)
                    .attr('y2', 0)
                    .attr('stroke', color)
                    .attr('stroke-width', 2);

                // Arrow head
                arrowGroup.append('polygon')
                    .attr('points', `${iconSize / 2},0 0,-${iconSize / 4} 0,${iconSize / 4}`)
                    .attr('fill', color);

                legendItem.append('text')
                    .attr('x', textOffset)
                    .attr('y', 4)
                    .text(treatment);

                yPos += itemHeight;
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