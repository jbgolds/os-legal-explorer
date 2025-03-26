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
        collisionRadius: 50,
        linkDistance: 250,        // Increased from 100 to 200 to spread nodes out more
        charge: -1600,            // Reduced from -2000 to -1000 for better balance with clustering
        enableClustering: true,   // Enable clustering by default
        clusteringStrength: 0.6,  // Increased from 0.4 to 0.6 for stronger type grouping
        textSimilarityStrength: 1.0, // SUPER strong text similarity force
        textSimilarityThreshold: 0.2, // Lower threshold to catch more similar names
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

    // Extract direction from API endpoint
    const direction = apiEndpoint.includes('direction=incoming') ? 'incoming' : 'outgoing';
    
    // Create a global storage for the network state
    if (!window.citationNetworkState) {
        window.citationNetworkState = {};
    }
    
    // Store the direction in the network state
    if (!window.citationNetworkState[containerId]) {
        window.citationNetworkState[containerId] = {};
    }
    window.citationNetworkState[containerId].direction = direction;

    // Select the container element
    const container = d3.select(`#${containerId}`);
    
    // Also get a direct DOM reference to the container
    const containerDOMElement = document.getElementById(containerId);

    // Get container dimensions
    const containerElement = document.getElementById(containerId);
    const containerRect = containerElement.parentElement.getBoundingClientRect();

    // Update width and height based on container size
    config.width = containerRect.width;
    config.height = Math.max(500, containerRect.height);

    // Clean up any previous network state for this container
    if (window.citationNetworkState[containerId].simulation) {
        // Stop any running simulation
        window.citationNetworkState[containerId].simulation.stop();
    }
    if (window.citationNetworkState[containerId].svg) {
        // Remove any event listeners from old SVG
        window.citationNetworkState[containerId].svg.on('.zoom', null);
    }

    // Always clear the container first before adding the loading indicator
    // Use vanilla JS to completely clear the container
    if (containerDOMElement) {
        containerDOMElement.innerHTML = '';
    }
    
    // Show loading indicator using D3
    container.append('div')
        .attr('class', 'network-loading')
        .text('Loading citation network...');

    // Fetch data from API
    fetch(apiEndpoint)
        .then(response => {
            if (response.status === 404) {
                const match = window.location.pathname.match(/^\/opinion\/([^\/]+)/);
                const clusterId = (match && match[1]) ? match[1] : 'unknown';
                console.warn(`API returned 404, falling back to dummy data for cluster ${clusterId}`);
                return { nodes: [{ id: clusterId, citation_string: `Cluster ${clusterId}`, type: 'opinion_cluster' }], links: [] };
            }
            if (!response.ok) {
                console.error(`API error: ${response.status} - ${response.statusText}`);
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Citation network data received: ${data.nodes.length} nodes, ${data.links.length} links`);

            // Check for incoming citation treatments
            if (direction === 'incoming') {
                const treatments = (data.links && data.links.length > 0) 
                    ? data.links.map(link => link.treatment || "").filter(t => t)
                    : [];
                    
                // Update treatment banner (will hide it if no treatments)
                updateTreatmentBanner(treatments);
            } else {
                // Always hide the treatment banner when viewing outgoing citations
                const bannerEl = document.getElementById('incoming-treatment-banner');
                if (bannerEl) {
                    bannerEl.classList.add('hidden');
                }
            }

            // Clear the container again before rendering the network
            // Use vanilla JS to completely clear the container
            if (containerDOMElement) {
                containerDOMElement.innerHTML = '';
            }
            
            // Process and render the full network with all data
            renderNetwork(containerId, data, config);
        })
        .catch(error => {
            console.error('Error fetching citation network data:', error);
            container.html(`<div class="flex items-center justify-center h-full"><div class="text-center"><p class="text-xl font-bold text-error">Error loading citation network</p><p class="text-gray-500">${error.message}</p></div></div>`);
        });
}

// Helper function to update the treatment banner
function updateTreatmentBanner(treatments) {
    const bannerEl = document.getElementById('incoming-treatment-banner');
    const textEl = document.getElementById('incoming-treatment-text');

    if (bannerEl && textEl) {
        // Hide the banner if there are no treatments
        if (!treatments || treatments.length === 0) {
            bannerEl.classList.add('hidden');
            return;
        }
        
        // Show banner if there are incoming citations with treatments
        bannerEl.classList.remove('hidden');

        // Check for treatments in priority order: POSITIVE > NEGATIVE > CAUTION > NEUTRAL
        
        if (treatments.includes("NEGATIVE")) {
            bannerEl.querySelector('.alert').className = "alert alert-error";
            textEl.textContent = "This case has NEGATIVE incoming citations";
        }
        else if (treatments.includes("CAUTION")) {
            bannerEl.querySelector('.alert').className = "alert alert-warning";
            textEl.textContent = "This case has CAUTION incoming citations";
        }
        else if (treatments.includes("POSITIVE")) {
            bannerEl.querySelector('.alert').className = "alert alert-success";
            textEl.textContent = "This case has POSITIVE incoming citations";
        }
        else {
            // Default for NEUTRAL or unspecified treatments
            bannerEl.querySelector('.alert').className = "alert alert-info";
            textEl.textContent = "This case has incoming citations";
        }
    }
}

// Function to render the network with the given data and config
function renderNetwork(containerId, data, config) {
    const container = d3.select(`#${containerId}`);
    const match = window.location.pathname.match(/^\/opinion\/([^\/]+)/);
    const clusterId = match ? match[1] : null;

    // Get the direction from the network state
    const direction = window.citationNetworkState[containerId].direction || 'outgoing';

    if (data.nodes.length === 1 && data.links.length === 0) {
        console.error('Only one node found with no citation relationships.');
        // Check if we should hide the process button (for incoming citations)
        let processButton = '';
        if (!config.hideProcessButton) {
            processButton = config.clusterId ?
                `<button class="btn btn-sm btn-outline mt-4" onclick="processCluster(${config.clusterId})">Process Citations</button>` :
                `<button class="btn btn-sm btn-outline mt-4" onclick="processCluster()">Process Citations</button>`;
        }

        // Use custom message if provided, otherwise use default
        const errorMessage = config.customEmptyMessage || "Only a single node was found in the graph, with no relations.";
        const subMessage = config.hideProcessButton ? "" : "Please process the cluster to create a citation network.";

        container.html(`<div class="flex items-center justify-center h-full"><div class="text-center"><p class="text-xl font-bold text-error">${errorMessage}</p>${subMessage ? `<p class="text-gray-500">${subMessage}</p>` : ''}${processButton}</div></div>`);
        return;
    }

    // If no data, show a message
    if (!data.nodes.length) {
        const noDataMessage = config.customEmptyMessage || "No Citation Data";
        const noDataSubMessage = config.customEmptySubMessage || "No citation network data available for this document.";
        container.html(`<div class="flex items-center justify-center h-full"><div class="text-center"><p class="text-xl font-bold">${noDataMessage}</p><p class="text-gray-500">${noDataSubMessage}</p></div></div>`);
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

    // Create SVG element
    const svg = container.append('svg')
        .attr('width', config.width)
        .attr('height', config.height)
        .attr('class', 'citation-network-svg')
        .on('click', function (event) {
            if (event.target === this) {
                hideDetailsPanel();
            }
        });

    // Set default for similarity lines toggle (controlled externally via HTML button)
    if (typeof window.similarityEnabled === 'undefined') {
        window.similarityEnabled = false;
    }

    // Create definitions for markers (arrows) - at SVG level
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

    // Position nodes in a circle to start
    data.nodes.forEach(node => {
        const angle = Math.random() * 2 * Math.PI;
        const radius = Math.min(config.width, config.height) * 0.3;
        node.x = config.width / 2 + radius * Math.cos(angle);
        node.y = config.height / 2 + radius * Math.sin(angle);
    });

    // Convert source/target references in links to objects if they're not already
    data.links.forEach(link => {
        // Make sure source and target are referring to the actual node objects
        if (typeof link.source === 'string' || typeof link.source === 'number') {
            const sourceNode = data.nodes.find(n => n.id === link.source);
            if (sourceNode) {
                link.source = sourceNode;
            }
        }
        if (typeof link.target === 'string' || typeof link.target === 'number') {
            const targetNode = data.nodes.find(n => n.id === link.target);
            if (targetNode) {
                link.target = targetNode;
            }
        }
    });

    // Set up the simulation
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links)
            .id(d => d.id)
            .distance(config.linkDistance))
        .force('charge', d3.forceManyBody().strength(config.charge))
        .force('center', d3.forceCenter(config.width / 2, config.height / 2))
        // Add moderate radial force to help distribute nodes
        .force('radial', d3.forceRadial(
            Math.min(config.width, config.height) * 0.2,
            config.width / 2,
            config.height / 2
        ).strength(0.3))
        .force('collision', d3.forceCollide().radius(d => config.nodeRadius + config.collisionRadius).iterations(3));

    // Add clustering force only if enabled
    if (config.enableClustering) {
        simulation.force('cluster', forceCluster()
            .strength(config.clusteringStrength)
            .nodes(data.nodes)
            .getType(d => getNodeType(d)));
    }

    // Always add text similarity force
    const textSimilarityForce = forceTextSimilarity()
        .strength(config.textSimilarityStrength)
        .similarityThreshold(config.textSimilarityThreshold)
        .nodes(data.nodes);

    simulation.force('textSimilarity', textSimilarityForce);

    // Reduce other forces to make text similarity more dominant
    simulation.force('charge').strength(config.charge * 0.15);
    simulation.force('link').distance(config.linkDistance * 1.5);

    if (config.enableClustering) {
        simulation.force('cluster').strength(config.clusteringStrength * 0.5);
    }

    // Make simulation strong at the beginning to help convergence
    simulation.alpha(0.8).restart();

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
            radius = Math.min(config.width, config.height) * 0.5; // Use 50% of the smaller dimension for better distribution

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

    // Custom force to attract nodes with similar citation strings
    function forceTextSimilarity() {
        let nodes;
        let strength = 0.1;
        let similarityThreshold = 0.4; // Minimum similarity score to consider (0-1)
        let similarityCache = {}; // Cache for similarity calculations
        let maxPairsToCheck = 5000; // Limit computation for large networks
        let similarityGroups = {}; // Track groups of similar nodes

        // Calculate Levenshtein distance between two strings
        function levenshteinDistance(a, b) {
            if (a.length === 0) return b.length;
            if (b.length === 0) return a.length;

            const matrix = Array(a.length + 1).fill().map(() => Array(b.length + 1).fill(0));

            for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
            for (let j = 0; j <= b.length; j++) matrix[0][j] = j;

            for (let i = 1; i <= a.length; i++) {
                for (let j = 1; j <= b.length; j++) {
                    const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + cost
                    );
                }
            }

            return matrix[a.length][b.length];
        }

        // Calculate similarity score between two strings (0-1)
        function calculateSimilarity(a, b) {
            // Use cache if available
            const cacheKey = a < b ? `${a}|${b}` : `${b}|${a}`;
            if (similarityCache[cacheKey] !== undefined) {
                return similarityCache[cacheKey];
            }

            if (!a || !b) return 0;

            // Normalize strings for comparison
            const str1 = a.toLowerCase().trim();
            const str2 = b.toLowerCase().trim();

            // For very different length strings, return low similarity
            if (Math.abs(str1.length - str2.length) > str1.length * 0.7) {
                similarityCache[cacheKey] = 0;
                return 0;
            }

            // Quick check for exact matches or very similar strings
            if (str1 === str2) {
                similarityCache[cacheKey] = 1;
                return 1;
            }

            // Check if one string contains the other
            if (str1.includes(str2) || str2.includes(str1)) {
                const containmentScore = 0.8; // High similarity for containment
                similarityCache[cacheKey] = containmentScore;
                return containmentScore;
            }

            const distance = levenshteinDistance(str1, str2);
            const maxLength = Math.max(str1.length, str2.length);

            if (maxLength === 0) {
                similarityCache[cacheKey] = 1;
                return 1; // Both empty strings
            }

            // Convert distance to similarity score (0-1)
            const similarity = 1 - (distance / maxLength);
            similarityCache[cacheKey] = similarity;
            return similarity;
        }

        function force(alpha) {
            // Apply force only if we have enough nodes
            if (!nodes || nodes.length < 2) return;

            // Reset similarity groups
            similarityGroups = {};

            // Scale the force by alpha, but use a MUCH higher multiplier to make it SUPER strong
            const scaledStrength = strength * alpha * 10.0; // Increased from 5.0 to 10.0

            // For large networks, limit the number of pairs we check
            const totalPairs = (nodes.length * (nodes.length - 1)) / 2;
            const checkAllPairs = totalPairs <= maxPairsToCheck;

            // For each pair of nodes, calculate similarity and apply force
            for (let i = 0; i < nodes.length; i++) {
                const nodeA = nodes[i];
                // Try to get citation string from multiple possible properties
                const citationA = nodeA.citation_string || nodeA.name || nodeA.title || nodeA.id || '';

                // Skip nodes without citation strings
                if (!citationA) continue;

                // For large networks, only check a subset of pairs
                const maxJ = checkAllPairs ? nodes.length : Math.min(nodes.length, i + 100);

                for (let j = i + 1; j < maxJ; j++) {
                    const nodeB = nodes[j];
                    // Try to get citation string from multiple possible properties
                    const citationB = nodeB.citation_string || nodeB.name || nodeB.title || nodeB.id || '';

                    // Skip nodes without citation strings
                    if (!citationB) continue;

                    // Calculate similarity between citation strings
                    const similarity = calculateSimilarity(citationA, citationB);

                    // Apply force only if similarity is above threshold
                    if (similarity > similarityThreshold) {
                        // Calculate force direction
                        const dx = nodeB.x - nodeA.x;
                        const dy = nodeB.y - nodeA.y;
                        const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                        // Force strength proportional to similarity, with a stronger effect
                        // Square the similarity to make high similarities even stronger
                        const force = (similarity * similarity * scaledStrength) / distance;

                        // Apply attractive force with increased strength
                        nodeA.vx += dx * force * 2; // Double the force
                        nodeA.vy += dy * force * 2;
                        nodeB.vx -= dx * force * 2;
                        nodeB.vy -= dy * force * 2;

                        // Track similarity groups for visualization
                        if (similarity > 0.5) {
                            // Create a unique group ID for this pair
                            const groupId = `group_${i}_${j}`;

                            // Add both nodes to this group
                            if (!similarityGroups[groupId]) {
                                similarityGroups[groupId] = {
                                    nodes: [nodeA, nodeB],
                                    similarity: similarity
                                };
                            }

                            // Mark nodes as being in a similarity group
                            nodeA.inSimilarityGroup = true;
                            nodeB.inSimilarityGroup = true;
                        }
                    }
                }
            }

            // Update node appearance based on similarity groups
            nodes.forEach(node => {
                if (node.inSimilarityGroup) {
                    node.hasSimilar = true;
                }
            });
        }

        force.similarityGroups = function () {
            return similarityGroups;
        };

        force.initialize = function (_) {
            nodes = _;
        };

        force.strength = function (_) {
            return arguments.length ? (strength = _, force) : strength;
        };

        force.similarityThreshold = function (_) {
            return arguments.length ? (similarityThreshold = _, force) : similarityThreshold;
        };

        force.nodes = function (_) {
            if (arguments.length) {
                nodes = _;
                similarityCache = {}; // Reset cache when nodes change
                return force;
            }
            return nodes;
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
        .enter().append('g');

    // Add a background rectangle for each text node
    node.append('rect')
        .attr('rx', 5) // Rounded corners
        .attr('ry', 5)
        .attr('fill', d => d.id === clusterId ? '#FFD700' : 'white')
        .attr('fill-opacity', 0.7) // Semi-transparent
        .attr('stroke', d => {
            const docType = getNodeType(d);
            return config.typeColors[docType] || config.typeColors.other;
        })
        .attr('stroke-width', 2); // Standard border width for all nodes

    // Add the text on top of the background
    const nodeText = node.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .each(function (d) {
            // Update the label splitting logic to use 30 characters per line
            const label = d.citation_string || '';
            let line1 = '', line2 = '';
            if (label.length <= 30) {
                line1 = label;
            } else if (label.length <= 60) {
                const mid = Math.ceil(label.length / 2);
                line1 = label.substring(0, mid);
                line2 = label.substring(mid);
            } else {
                line1 = label.substring(0, 30);
                line2 = label.substring(30, 60) + '...';
            }
            d3.select(this).text("");
            if (line2) {
                d3.select(this).append('tspan')
                    .attr('x', 0)
                    .attr('dy', '-0.6em')
                    .text(line1);
                d3.select(this).append('tspan')
                    .attr('x', 0)
                    .attr('dy', '1.2em')
                    .text(line2);
            } else {
                d3.select(this).append('tspan')
                    .attr('x', 0)
                    .attr('dy', '0.35em')
                    .text(line1);
            }
        })
        .attr('fill', d => {
            const docType = getNodeType(d);
            return config.typeColors[docType] || config.typeColors.other;
        })
        .style('font-weight', 'bold') // Bold text for all nodes
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
            showDetails(d, event);
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

            if (d.reasoning) {
                tooltip += `Reasoning: ${d.reasoning}\n`;
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
    /*
    link.on('click', function (event, d) {
        event.stopPropagation(); // Prevent click from propagating to SVG
        showDetails(d, event);
    });
    */

    // Add click handler to SVG to close any open details panel
    svg.on('click', function () {
        hideDetailsPanel();
    });

    // Function to show details
    function showDetails(item, event) {
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

        // Create a header with flexbox layout
        const header = detailsPanel.append('div')
            .style('display', 'flex')
            .style('justify-content', 'space-between')
            .style('align-items', 'center')
            .style('margin-bottom', '10px');

        // Add the title to the header
        header.append('h3')
            .style('margin', '0')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('Document Details');

        // Add close button to the header
        header.append('button')
            .attr('class', 'details-close-btn')
            .style('background', 'none')
            .style('border', 'none')
            .style('cursor', 'pointer')
            .style('font-size', '20px')
            .style('line-height', '1')
            .style('padding', '0 0 0 10px')
            .style('margin', '0')
            .style('color', '#666')
            .text('×')
            .on('click', function () {
                hideDetailsPanel();
            });

        // Create content container
        const content = detailsPanel.append('div');

        // Find all links connected to this node
        const connectedLinks = {
            incoming: data.links.filter(link => link.target.id === item.id),
            outgoing: data.links.filter(link => link.source.id === item.id)
        };

        // Render the main document details
        renderDocumentDetails(item, content);

        // If there are connected links, show them in a tabbed interface
        if (connectedLinks.incoming.length > 0 || connectedLinks.outgoing.length > 0) {
            // Add a section for relationships
            const relationshipsSection = content.append('div')
                .style('margin-top', '20px');

            // Create tabs for incoming and outgoing citations
            const tabContainer = relationshipsSection.append('div')
                .style('border-bottom', '1px solid #ddd');

            // Create tab buttons
            const tabs = [];

            if (connectedLinks.incoming.length > 0) {
                tabs.push({
                    label: `Cited By (${connectedLinks.incoming.length})`,
                    links: connectedLinks.incoming,
                    direction: 'incoming'
                });
            }

            if (connectedLinks.outgoing.length > 0) {
                tabs.push({
                    label: `Cites (${connectedLinks.outgoing.length})`,
                    links: connectedLinks.outgoing,
                    direction: 'outgoing'
                });
            }

            // Create tab buttons and content
            tabs.forEach((tab, index) => {
                // Create tab button
                tab.button = tabContainer.append('button')
                    .attr('class', `tab-btn tab-${index}`)
                    .style('background', 'none')
                    .style('border', 'none')
                    .style('padding', '8px 12px')
                    .style('cursor', 'pointer')
                    .style('font-weight', index === 0 ? 'bold' : 'normal')
                    .style('border-bottom', index === 0 ? '3px solid #2196F3' : '3px solid transparent')
                    .text(tab.label);

                // Create content container
                tab.content = relationshipsSection.append('div')
                    .attr('class', `tab-content content-${index}`)
                    .style('display', index === 0 ? 'block' : 'none');

                // Render the relationships
                renderRelationships(tab.links, tab.content, tab.direction);
            });

            // Add tab switching functionality
            tabs.forEach((tab, tabIndex) => {
                tab.button.on('click', function () {
                    // Update all tab buttons and content
                    tabs.forEach((t, i) => {
                        // Update button styles
                        t.button.style('font-weight', i === tabIndex ? 'bold' : 'normal')
                            .style('border-bottom', i === tabIndex ? '3px solid #2196F3' : '3px solid transparent');

                        // Show/hide content
                        t.content.style('display', i === tabIndex ? 'block' : 'none');
                    });
                });
            });
        }

        // Function to render document details
        function renderDocumentDetails(doc, container) {
            // Document title
            container.append('p')
                .style('font-weight', 'bold')
                .style('margin', '15px 0 10px 0')
                .style('word-wrap', 'break-word')
                .style('overflow-wrap', 'break-word')
                .style('hyphens', 'auto')
                .style('max-width', '100%')
                .text(doc.citation_string || 'Unknown Document');

            // Document type with colored indicator
            const docType = (doc.type || 'other').toLowerCase();

            // Use the same getNodeType function that's used for the visualization
            const mappedType = getNodeType(doc);
            let typeColor = config.typeColors[mappedType] || config.typeColors.other;

            // Map enum type to human-readable label
            function getTypeLabel(type) {
                if (!type) return 'Unknown';

                // First, map API document types to our internal enum types (same as in getNodeType)
                const apiToEnumMapping = {
                    'opinion_cluster': 'judicial_opinion',
                    'statutes': 'statutes_codes_regulations',
                    'constitutional_documents': 'constitution',
                    'admin_rulings': 'administrative_agency_ruling',
                    'congressional_reports': 'congressional_report',
                    'submissions': 'external_submission',
                    'law_reviews': 'law_review',
                    'other_legal_documents': 'other'
                };

                // Convert to lowercase for case-insensitive matching
                const lowerType = type.toLowerCase();

                // Try to map API type to enum type first
                const enumType = apiToEnumMapping[lowerType] || lowerType;

                // Then map enum type to human-readable label
                const enumToLabelMapping = {
                    'judicial_opinion': 'Judicial Opinion',
                    'statutes_codes_regulations': 'Statute/Code/Regulation',
                    'constitution': 'Constitution',
                    'administrative_agency_ruling': 'Administrative/Agency Ruling',
                    'congressional_report': 'Congressional Report',
                    'external_submission': 'External Submission',
                    'electronic_resource': 'Electronic Resource',
                    'law_review': 'Law Review',
                    'legal_dictionary': 'Legal Dictionary',
                    'other': 'Other'
                };

                // Return the human-readable label, or format the type name if no mapping exists
                if (enumToLabelMapping[enumType]) {
                    return enumToLabelMapping[enumType];
                } else {
                    // Format the type name for better readability (same as in legend)
                    return enumType.split('_').map(word =>
                        word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ');
                }
            }

            const typeRow = container.append('div')
                .style('display', 'flex')
                .style('align-items', 'center')
                .style('margin-bottom', '5px');

            typeRow.append('span')
                .style('display', 'inline-block')
                .style('width', '12px')
                .style('height', '12px')
                .style('border-radius', '50%')
                .style('background-color', typeColor)
                .style('margin-right', '8px');

            typeRow.append('span')
                .text(`Type: ${getTypeLabel(doc.type)}`);

            // Basic document info
            if (doc.court) {
                container.append('p')
                    .style('margin', '5px 0')
                    .text(`Court: ${doc.court}`);
            }

            if (doc.year) {
                container.append('p')
                    .style('margin', '5px 0')
                    .text(`Year: ${doc.year}`);
            }

            // Metadata section
            if (doc.metadata) {
                const metadataList = container.append('dl')
                    .style('margin', '0')
                    .style('padding', '0');

                Object.entries(doc.metadata).forEach(([key, value]) => {
                    // Skip the primary_table field
                    if (value && typeof value !== 'object' && key !== 'primary_table') {
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

            // Link to full document
            if (doc.id) {
                const linkContainer = container.append('div')
                    .style('margin-top', '15px');

                // Check if this is a document with a primary_id (which would be a cluster_id)
                // or if the id itself is a numeric cluster ID
                const hasValidClusterId = (doc.primary_id && /^\d+$/.test(doc.primary_id)) ||
                    (!doc.primary_id && /^\d+$/.test(doc.id));

                if (hasValidClusterId) {
                    // Use primary_id if available, otherwise use id
                    const clusterId = doc.primary_id || doc.id;
                    linkContainer.append('a')
                        .attr('href', `/opinion/${clusterId}`)
                        .attr('target', '_blank')
                        .style('color', '#2196F3')
                        .style('text-decoration', 'none')
                        .text('View Full Document →');
                } else {
                    // No valid cluster ID available
                    linkContainer.append('p')
                        .style('font-style', 'italic')
                        .style('font-size', '13px')
                        .style('color', '#666')
                        .text('Cited Document Not Available :(');
                }
            }
        }

        // Function to render relationships
        function renderRelationships(links, container, direction) {
            if (links.length === 0) {
                container.append('p')
                    .style('font-style', 'italic')
                    .style('color', '#666')
                    .text('No relationships found.');
                return;
            }

            // Create a list of relationships
            const relationshipList = container.append('div')
                .style('margin-top', '10px');

            // Sort links by relevance (if available) or treatment
            links.sort((a, b) => {
                // First by relevance (higher first)
                if (a.relevance && b.relevance) {
                    return b.relevance - a.relevance;
                }
                // Then by treatment (positive first)
                if (a.treatment && b.treatment) {
                    const treatmentOrder = {
                        'POSITIVE': 0,
                        'NEUTRAL': 1,
                        'CAUTION': 2,
                        'NEGATIVE': 3
                    };
                    return (treatmentOrder[a.treatment] || 99) - (treatmentOrder[b.treatment] || 99);
                }
                return 0;
            });

            // Render each relationship
            links.forEach(link => {
                // Get the other document (not the current one)
                const otherDoc = direction === 'incoming' ? link.source : link.target;

                // Create relationship item
                const relationshipItem = relationshipList.append('div')
                    .style('margin-bottom', '15px')
                    .style('padding', '10px')
                    .style('background', '#f8f9fa')
                    .style('border-radius', '4px')
                    .style('border-left', '3px solid ' + (link.treatment && config.treatmentColors[link.treatment] ?
                        config.treatmentColors[link.treatment] : '#666'));

                // Relationship header with document name and arrow
                const relationshipHeader = relationshipItem.append('div')
                    .style('display', 'flex')
                    .style('align-items', 'center')
                    .style('margin-bottom', '5px');

                // Just show the other document name, regardless of direction
                relationshipHeader.append('span')
                    .style('font-weight', 'bold')
                    .text(otherDoc.citation_string || 'Unknown Document');

                // Add relationship details
                const detailsContainer = relationshipItem.append('div')
                    .style('margin-top', '5px')
                    .style('font-size', '13px');

                // Treatment
                if (link.treatment) {
                    detailsContainer.append('div')
                        .text(`Treatment: ${link.treatment}`);
                }

                // Relevance
                if (link.relevance) {
                    detailsContainer.append('div')
                        .text(`Relevance: ${link.relevance}`);
                }

                // Citation text
                if (link.metadata && link.metadata.citation_text) {
                    detailsContainer.append('div')
                        .style('margin-top', '8px')
                        .style('padding', '8px')
                        .style('background', '#f5f5f5')
                        .style('font-style', 'italic')
                        .text(`"${link.metadata.citation_text}"`);
                }

                // Reasoning
                if (link.reasoning) {
                    detailsContainer.append('div')
                        .style('margin-top', '8px')
                        .text(`Reasoning: ${link.reasoning}`);
                }

            });
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

    // Create a group for similarity lines
    const similarityLines = g.append('g')
        .attr('class', 'similarity-lines');

    // Update positions on each tick
    simulation.on('tick', () => {
        // Update link positions
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        // Update node positions
        node
            .attr('transform', d => `translate(${d.x}, ${d.y})`);

        // Update similarity lines
        similarityLines.selectAll('line').remove();

        if (window.similarityEnabled && simulation.force('textSimilarity')) {
            const similarityForce = simulation.force('textSimilarity');
            const groups = Object.values(similarityForce.similarityGroups());

            groups.forEach(group => {
                if (group.nodes.length >= 2 && group.similarity > 0.7) {
                    const cx = d3.mean(group.nodes, d => d.x);
                    const cy = d3.mean(group.nodes, d => d.y);
                    group.nodes.forEach(node => {
                        similarityLines.append('line')
                            .attr('x1', node.x)
                            .attr('y1', node.y)
                            .attr('x2', cx)
                            .attr('y2', cy)
                            .attr('stroke', '#FF00FF') // Magenta
                            .attr('stroke-width', 2)
                            .attr('stroke-dasharray', '3,3')
                            .attr('opacity', 0.6);
                    });
                }
            });
        }
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

    // Store the simulation, zoom, and direction in a global namespace for use with controls
    window.citationNetworkState = window.citationNetworkState || {};
    window.citationNetworkState[containerId] = {
        svg: svg,
        simulation: simulation,
        zoom: zoom,
        direction: direction
    };
}

// Modify processCluster function to alert immediately when processButton is clicked
function processCluster(id) {
    if (!id) {
        const match = window.location.pathname.match(/^\/opinion\/([^\/]+)/);
        id = match ? match[1] : 'default cluster';
    }
    alert("Processing initiated. Please refresh in 1-2 minutes.");
    console.log("Processing cluster", id);
    // API call to start processing the cluster
    fetch(`/api/pipeline/process-cluster/${id}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ clusterId: id })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Cluster starting to be processed:", data);
            // Removed alert from here
        })
        .catch(error => {
            console.error("Error processing cluster:", error);
            alert("There was an error processing the cluster. Please try again.");
        });
}
window.processCluster = processCluster;

// Export the function
window.renderCitationNetwork = renderCitationNetwork; 