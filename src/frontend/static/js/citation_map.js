/**
 * Citation Network Visualization
 * 
 * This module provides functions to visualize legal citation networks using D3.js.
 * It supports rendering nodes of different types (Opinion, Statute, etc.) and
 * citation relationships between them with visual cues for treatment and relevance.
 */

// Main function to render the citation network
function renderCitationNetwork(containerId, apiEndpoint, options = {}) {
    // Debug logs to help identify container issues
    console.log("renderCitationNetwork called with containerId:", containerId);
    console.log("API endpoint:", apiEndpoint);
    
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
    console.log("Network direction:", direction);
    
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
    console.log("Container element:", container.node());
    
    // Also get a direct DOM reference to the container
    const containerDOMElement = document.getElementById(containerId);
    if (!containerDOMElement) {
        console.error(`Container element with ID "${containerId}" not found in DOM`);
    }

    // Get container dimensions
    const containerElement = document.getElementById(containerId);
    const containerRect = containerElement && containerElement.parentElement ? 
        containerElement.parentElement.getBoundingClientRect() : 
        { width: defaults.width, height: defaults.height };

    // Log dimensions before applying
    console.log("Original container dimensions:", containerRect.width, containerRect.height);

    // CRITICAL FIX: Ensure reasonable height for the network and default to much smaller height
    config.width = containerRect.width || defaults.width;
    config.height = Math.min(Math.max(500, containerRect.height || defaults.height), 800); // Cap height at 800px
    console.log("Adjusted network dimensions:", config.width, config.height);

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
    console.log("Updating treatment banner. Banner element:", bannerEl, "Text element:", textEl, "Treatments:", treatments);

    if (bannerEl && textEl) {
        // Hide the banner if there are no treatments
        if (!treatments || treatments.length === 0) {
            bannerEl.classList.add('hidden');
            return;
        }
        
        // Show banner if there are incoming citations with treatments
        bannerEl.classList.remove('hidden');

        // Make sure banner doesn't overlap or hide the network
        bannerEl.style.position = "relative";
        bannerEl.style.zIndex = "1";  // Lower z-index to ensure it doesn't cover network
        
        // IMPORTANT: Set a reasonable height for the banner
        bannerEl.style.maxHeight = "50px";
        bannerEl.style.overflow = "hidden";

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
    console.log("Rendering network in container:", containerId);
    const container = d3.select(`#${containerId}`);
    const match = window.location.pathname.match(/^\/opinion\/([^\/]+)/);
    const clusterId = match ? match[1] : null;

    // Get the direction from the network state
    const direction = window.citationNetworkState[containerId].direction || 'outgoing';
    console.log("Network direction for rendering:", direction);

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
        .style('overflow', 'hidden')  // Changed from 'visible' to 'hidden' to clip content outside the SVG 
        .style('display', 'block')    
        .style('margin-top', '10px')  
        .style('position', 'relative')
        .style('top', '0')           
        .on('click', function (event) {
            if (event.target === this) {
                hideDetailsPanel();
            }
        });

    // Remove the extra SVG parent visibility check
    console.log("Created SVG with dimensions:", config.width, config.height);

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
        .force('center', d3.forceCenter(config.width / 2, config.height / 3)) // Position at 1/3 height for better visibility
        // Add moderate radial force to help distribute nodes
        .force('radial', d3.forceRadial(
            Math.min(config.width, config.height) * 0.2,
            config.width / 2,
            config.height / 3  // Center at 1/3 height instead of 1/2
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

    // Add click event to SVG to close any open details panel
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
    
    // IMPORTANT FIX: Set initial transform to center the graph properly
    // and use a reasonable initial scale
    const initialScale = direction === 'incoming' ? 0.6 : 0.5; // Slightly larger scale for incoming
    svg.call(zoom.transform, d3.zoomIdentity
        .translate(config.width/2, config.height/3) // Position at 1/3 height
        .scale(initialScale)
        .translate(-config.width/2, -config.height/3));
    
    console.log("Applied initial zoom transform with scale:", initialScale);

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
        direction: direction,
        data: data // Store reference to the data for the table
    };
    
    // Remove dynamic reset button - already exists in HTML
    
    console.log("Network rendering complete. State:", window.citationNetworkState[containerId]);
    
    // Render the citation table if it exists
    if (document.getElementById('citation-table-body')) {
        renderCitationTable(data, config);
    }

    // After SVG rendering is complete, add a direct fix to force the containerDOMElement to show the network
    // Add this code at the end of renderNetwork function, just before the return
    console.log("Ensuring network visibility for container:", containerId);
    const networkContainer = document.getElementById(containerId);
    if (networkContainer) {
        networkContainer.style.position = 'relative';
        networkContainer.style.minHeight = '500px';
        networkContainer.style.visibility = 'visible';
        networkContainer.style.display = 'block';
        networkContainer.style.zIndex = '5';  // Make sure it's above other elements
        console.log("Applied container visibility fixes");
    }
}

// Function to reset the network view
function resetNetworkView(containerId) {
    console.log("Resetting view for container:", containerId);
    if (!containerId) {
        // If no container ID provided, try to use the first one available
        containerId = Object.keys(window.citationNetworkState)[0];
    }
    
    if (containerId && window.citationNetworkState[containerId]) {
        const networkState = window.citationNetworkState[containerId];
        if (networkState.svg && networkState.zoom) {
            const svg = networkState.svg;
            const width = parseInt(svg.attr('width'));
            const height = parseInt(svg.attr('height'));
            
            // Get the current direction to apply appropriate scale
            const direction = networkState.direction || 'outgoing';
            const scale = direction === 'incoming' ? 0.6 : 0.5;
            
            // Reset zoom with animation - position higher in the viewport
            svg.transition()
                .duration(750)
                .call(networkState.zoom.transform, d3.zoomIdentity
                    .translate(width/2, height/3)
                    .scale(scale)
                    .translate(-width/2, -height/3));
                    
            console.log("View reset applied for direction:", direction);
        }
    }
}

// Make resetNetworkView available globally
window.resetNetworkView = resetNetworkView;

// Helper function to consistently determine node type
function getNodeType(node, config) {
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

// Helper function to get human-readable type label
function getTypeLabel(type) {
    if (!type) return 'Unknown';

    // Map enum type to human-readable label
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
    if (enumToLabelMapping[type]) {
        return enumToLabelMapping[type];
    } else {
        // Format the type name for better readability (same as in legend)
        return type.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
}

// Calculate similarity score between two strings (0-1)
function calculateSimilarity(a, b) {
    if (!a || !b) return 0;

    // Normalize strings for comparison
    const str1 = a.toLowerCase().trim();
    const str2 = b.toLowerCase().trim();

    // For very different length strings, return low similarity
    if (Math.abs(str1.length - str2.length) > str1.length * 0.7) {
        return 0;
    }

    // Quick check for exact matches or very similar strings
    if (str1 === str2) {
        return 1;
    }

    // Check if one string contains the other
    if (str1.includes(str2) || str2.includes(str1)) {
        return 0.8; // High similarity for containment
    }

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

    const distance = levenshteinDistance(str1, str2);
    const maxLength = Math.max(str1.length, str2.length);

    if (maxLength === 0) {
        return 1; // Both empty strings
    }

    // Convert distance to similarity score (0-1)
    return 1 - (distance / maxLength);
}

// Function to prepare table data, combining nodes with their link information
function prepareTableData(data, direction, config) {
    const match = window.location.pathname.match(/^\/opinion\/([^\/]+)/);
    const clusterId = match ? match[1] : null;
    
    // Create a map of nodes by id for quick lookup
    const nodesMap = new Map();
    data.nodes.forEach(node => {
        nodesMap.set(node.id, { ...node });
    });
    
    // Process links based on direction
    const tableData = [];
    
    // Skip the central node (the current opinion)
    data.nodes.forEach(node => {
        if (node.id !== clusterId) {
            // Create a fresh copy of node data for this direction
            const nodeData = { ...node };
            
            // Find the link involving this node based on the current direction
            let link = null;
            
            if (direction === 'outgoing') {
                // For outgoing, look for links where clusterId is the source and node is the target
                link = data.links.find(l => 
                    (l.source.id === clusterId && l.target.id === node.id) ||
                    (typeof l.source === 'string' && l.source === clusterId && l.target.id === node.id));
                
                if (link) {
                    nodeData.treatment = link.treatment || 'NEUTRAL';
                    nodeData.relevance = link.relevance; // Use relevance exactly as provided by API
                    nodeData.citation_text = link.metadata?.citation_text || '';
                    nodeData.section = link.section || link.metadata?.opinion_section || 'MAJORITY';
                    
                    // Check all possible locations for reasoning data
                    nodeData.reasoning = link.reasoning || link.metadata?.reasoning || '';
                    
                    // Log the link data to debug
                    console.log('Link data for outgoing node:', node.id, link);
                }
            } else {
                // For incoming, look for links where node is the source and clusterId is the target
                link = data.links.find(l => 
                    (l.target.id === clusterId && l.source.id === node.id) ||
                    (typeof l.target === 'string' && l.target === clusterId && l.source.id === node.id));
                
                if (link) {
                    nodeData.treatment = link.treatment || 'NEUTRAL';
                    nodeData.relevance = link.relevance; // Use relevance exactly as provided by API
                    nodeData.citation_text = link.metadata?.citation_text || '';
                    nodeData.section = link.section || link.metadata?.opinion_section || 'MAJORITY';
                    
                    // Check all possible locations for reasoning data
                    nodeData.reasoning = link.reasoning || link.metadata?.reasoning || '';
                    
                    // Log the link data to debug
                    console.log('Link data for incoming node:', node.id, link);
                }
            }
            
            // Only add to table data if we found a link
            if (link) {
                // Add standardized type
                nodeData.standardType = getNodeType(node, config);
                
                // Add to table data
                tableData.push(nodeData);
            }
        }
    });
    
    return tableData;
}

// Function to render citation table
function renderCitationTable(data, config) {
    if (!data || !data.nodes || data.nodes.length === 0) {
        console.log('No citation data to display in table');
        return;
    }

    // Get the direction from config, defaulting to outgoing
    const direction = config.direction || 'outgoing';
    console.log(`Rendering citation table with direction: ${direction}`);

    // Update table header to show the current direction
    const tableHeader = document.querySelector('.citation-table-header');
    if (tableHeader) {
        const directionText = direction.charAt(0).toUpperCase() + direction.slice(1);
        tableHeader.textContent = `Citation Table (${directionText})`;
    }

    // Prepare data for table
    const tableData = prepareTableData(data, direction, config);
    
    // Populate filter options
    populateFilterOptions(tableData);
    
    // Render initial table
    renderTable(tableData, config);
    
    // Set up event listeners for table interactions
    setupTableEventListeners(data, config);
}

// Function to populate filter options
function populateFilterOptions(tableData) {
    const typeFilterOptions = document.getElementById('type-filter-options');
    if (!typeFilterOptions) return;
    
    // Clear existing options
    typeFilterOptions.innerHTML = '';
    
    // Add "All" option
    const allOption = document.createElement('li');
    const allLink = document.createElement('a');
    allLink.textContent = 'All Types';
    allLink.setAttribute('data-type', 'all');
    allLink.classList.add('active');
    allOption.appendChild(allLink);
    typeFilterOptions.appendChild(allOption);
    
    // Get unique types
    const types = [...new Set(tableData.map(item => item.standardType))];
    
    // Add option for each type
    types.forEach(type => {
        const option = document.createElement('li');
        const link = document.createElement('a');
        link.textContent = getTypeLabel(type);
        link.setAttribute('data-type', type);
        option.appendChild(link);
        typeFilterOptions.appendChild(option);
    });
}

// Function to render the citation table
function renderTable(tableData, config, sortField = 'citation_string', sortOrder = 'asc', filter = '', filterType = 'all', groupSimilar = false) {
    const tableBody = document.getElementById('citation-table-body');
    if (!tableBody) return;
    
    // Clear the table body
    tableBody.innerHTML = '';
    
    // Group by type first
    let groupedData = groupDataByType(tableData);
    
    // Sort the groups
    Object.keys(groupedData).forEach(type => {
        groupedData[type] = sortData(groupedData[type], sortField, sortOrder);
    });
    
    // If grouping by similarity is enabled, group similar items within each type
    if (groupSimilar) {
        Object.keys(groupedData).forEach(type => {
            groupedData[type] = groupSimilarItems(groupedData[type]);
        });
    }
    
    // Filter data if needed
    if (filter || filterType !== 'all') {
        Object.keys(groupedData).forEach(type => {
            if (filterType !== 'all' && type !== filterType) {
                delete groupedData[type];
            } else if (filter) {
                groupedData[type] = groupedData[type].filter(item => 
                    (item.citation_string && item.citation_string.toLowerCase().includes(filter.toLowerCase())) ||
                    (item.court && item.court.toLowerCase().includes(filter.toLowerCase())) ||
                    (item.year && String(item.year).includes(filter)) ||
                    (item.section && item.section.toLowerCase().includes(filter.toLowerCase()))
                );
            }
        });
    }
    
    // Define type order to ensure judicial opinions come first
    const typeOrder = [
        'judicial_opinion', // Judicial opinions first
        'statutes_codes_regulations',
        'constitution',
        'administrative_agency_ruling',
        'congressional_report',
        'law_review',
        'external_submission',
        'electronic_resource',
        'legal_dictionary',
        'other'
    ];
    
    // Get available types and sort them according to our preferred order
    const availableTypes = Object.keys(groupedData).filter(type => groupedData[type].length > 0);
    availableTypes.sort((a, b) => {
        const indexA = typeOrder.indexOf(a);
        const indexB = typeOrder.indexOf(b);
        
        // If both types are in our order list, sort by their position
        if (indexA !== -1 && indexB !== -1) {
            return indexA - indexB;
        }
        
        // If only one is in the list, prioritize it
        if (indexA !== -1) return -1;
        if (indexB !== -1) return 1;
        
        // If neither is in the list, maintain alphabetical order
        return a.localeCompare(b);
    });
    
    // Render the grouped data in the defined order
    let rowIndex = 0;
    availableTypes.forEach(type => {
        // Add type header
        const typeHeaderRow = document.createElement('tr');
        typeHeaderRow.className = 'group-header';
        typeHeaderRow.style.backgroundColor = '#f0f0f0';
        
        const typeHeaderCell = document.createElement('td');
        typeHeaderCell.colSpan = 5; // 5 columns (added Section column)
        typeHeaderCell.style.fontWeight = 'bold';
        typeHeaderCell.textContent = getTypeLabel(type);
        
        typeHeaderRow.appendChild(typeHeaderCell);
        tableBody.appendChild(typeHeaderRow);
        
        // Add items for this type
        groupedData[type].forEach((item, index) => {
            // Handle similarity groups
            if (item.similarItems) {
                // This is a group of similar items, render the first one
                const primaryItem = item.items[0];
                const row = createTableRow(primaryItem, rowIndex++);
                row.classList.add('similar-group-primary');
                
                // Add expander icon in the first cell
                const citationCell = row.querySelector('td:first-child');
                const expanderIcon = document.createElement('span');
                expanderIcon.innerHTML = '▶';
                expanderIcon.className = 'expander-icon mr-2 cursor-pointer';
                expanderIcon.setAttribute('data-expanded', 'false');
                citationCell.prepend(expanderIcon);
                
                tableBody.appendChild(row);
                
                // Create container for similar items (initially hidden)
                const similarContainer = document.createElement('tr');
                similarContainer.className = 'similar-items-container hidden';
                const similarCell = document.createElement('td');
                similarCell.colSpan = 5; // 5 columns (added Section column)
                similarCell.style.padding = '0';
                
                // Create inner table for similar items
                const similarTable = document.createElement('table');
                similarTable.className = 'table w-full';
                const similarBody = document.createElement('tbody');
                
                // Add the primary item as the first row in the dropdown with a white background
                const primaryRow = createTableRow(primaryItem, null, false);
                primaryRow.style.backgroundColor = 'white';
                primaryRow.classList.add('similar-group-primary-duplicate');

                // Apply consistent styling to citation cells in the primary row
                const primaryCitationCell = primaryRow.querySelector('td:first-child');
                if (primaryCitationCell) {
                    primaryCitationCell.style.maxWidth = '200px';
                    primaryCitationCell.style.wordWrap = 'break-word';
                    primaryCitationCell.style.whiteSpace = 'normal';
                }

                similarBody.appendChild(primaryRow);
                
                // Add click event to highlight node for primary row
                primaryRow.addEventListener('click', function(event) {
                    event.stopPropagation(); // Prevent triggering parent row's expander
                    const nodeId = this.getAttribute('data-citation-id');
                    highlightNetworkNode(nodeId);
                });
                
                // Add a divider after the primary item
                const dividerRow = document.createElement('tr');
                const dividerCell = document.createElement('td');
                dividerCell.colSpan = 5; // 5 columns (added Section column)
                dividerCell.style.padding = '0';
                dividerCell.style.borderBottom = '1px dashed #ccc';
                dividerRow.appendChild(dividerCell);
                similarBody.appendChild(dividerRow);
                
                // Add the rest of the similar items
                item.items.slice(1).forEach(similarItem => {
                    const similarRow = createTableRow(similarItem, null, false);
                    similarRow.style.backgroundColor = '#f8f8f8';
                    
                    // Apply consistent styling to citation cells in the similar items dropdown
                    const citationCell = similarRow.querySelector('td:first-child');
                    if (citationCell) {
                        citationCell.style.maxWidth = '200px';
                        citationCell.style.wordWrap = 'break-word';
                        citationCell.style.whiteSpace = 'normal';
                    }
                    
                    // Add click event to highlight node for similar row
                    similarRow.addEventListener('click', function(event) {
                        event.stopPropagation(); // Prevent triggering parent row's expander
                        const nodeId = this.getAttribute('data-citation-id');
                        highlightNetworkNode(nodeId);
                    });
                    
                    similarBody.appendChild(similarRow);
                });
                
                similarTable.appendChild(similarBody);
                similarCell.appendChild(similarTable);
                similarContainer.appendChild(similarCell);
                tableBody.appendChild(similarContainer);
                
                // Store references for toggle functionality
                row.expanderIcon = expanderIcon;
                row.similarContainer = similarContainer;
                
                // Add click event to expander icon
                expanderIcon.addEventListener('click', function(event) {
                    event.stopPropagation(); // Prevent triggering the row click event
                    toggleExpander(row);
                });
                
                // Add click event to the entire row to toggle expander
                row.addEventListener('click', function(event) {
                    // Toggle the expander when clicking anywhere on the row
                    toggleExpander(row);
                });
            } else {
                // Regular item
                const row = createTableRow(item, rowIndex++);
                
                // Add click event to highlight node for regular rows
                row.addEventListener('click', function(event) {
                    const nodeId = this.getAttribute('data-citation-id');
                    highlightNetworkNode(nodeId);
                });
                
                tableBody.appendChild(row);
            }
        });
    });
    
    // Helper function to toggle expander state
    function toggleExpander(row) {
        const expanderIcon = row.expanderIcon;
        const similarContainer = row.similarContainer;
        
        const isExpanded = expanderIcon.getAttribute('data-expanded') === 'true';
        expanderIcon.textContent = isExpanded ? '▶' : '▼';
        expanderIcon.setAttribute('data-expanded', !isExpanded);
        similarContainer.classList.toggle('hidden');
    }
    
    // Update sort indicators
    updateSortIndicators(sortField, sortOrder);
}

// Function to create a table row for a citation
function createTableRow(item, rowIndex, includeBgColor = true) {
    const row = document.createElement('tr');
    if (rowIndex !== null && rowIndex % 2 === 1 && includeBgColor) {
        row.classList.add('bg-base-200');
    }
    
    // Citation
    const citationCell = document.createElement('td');
    citationCell.textContent = item.citation_string || 'Unknown';
    citationCell.setAttribute('data-citation-id', item.id);
    // Add styling to limit width and enable wrapping
    citationCell.style.maxWidth = '200px';
    citationCell.style.wordWrap = 'break-word';
    citationCell.style.whiteSpace = 'normal';
    row.appendChild(citationCell);
    
    // Relevance
    const relevanceCell = document.createElement('td');
    relevanceCell.style.width = '60px'; // Narrower width
    if (typeof item.relevance !== 'undefined' && item.relevance !== null) {
        // Use the raw relevance value from the API without transformation
        relevanceCell.textContent = item.relevance;
    } else {
        relevanceCell.textContent = '—';
    }
    row.appendChild(relevanceCell);
    
    // Treatment
    const treatmentCell = document.createElement('td');
    treatmentCell.style.width = '90px'; // Narrower width
    if (item.treatment) {
        // Create a span with visible background color
        const treatmentSpan = document.createElement('span');
        
        // Normalize the treatment value to handle any casing issues
        const treatmentValue = item.treatment.toUpperCase();
        treatmentSpan.textContent = treatmentValue;
        treatmentSpan.className = 'badge';
        
        // Add appropriate class based on treatment with strong colors
        switch (treatmentValue) {
            case 'POSITIVE':
                treatmentSpan.classList.add('badge-success');
                // Add inline styles to ensure it's completely filled green with no outline
                treatmentSpan.style.backgroundColor = '#4CAF50';
                treatmentSpan.style.color = 'white';
                treatmentSpan.style.border = 'none';
                break;
            case 'NEGATIVE':
                treatmentSpan.classList.add('badge-error');
                // Add inline styles to ensure it's completely filled red with no outline
                treatmentSpan.style.backgroundColor = '#F44336';
                treatmentSpan.style.color = 'white';
                treatmentSpan.style.border = 'none';
                break;
            case 'CAUTION':
                treatmentSpan.classList.add('badge-warning');
                // Add inline styles to ensure it's completely filled orange with no outline
                treatmentSpan.style.backgroundColor = '#FF9800';
                treatmentSpan.style.color = 'black';
                treatmentSpan.style.border = 'none';
                break;
            default:
                treatmentSpan.classList.add('badge-neutral');
        }
        
        treatmentCell.appendChild(treatmentSpan);
    } else {
        treatmentCell.textContent = '—';
    }
    row.appendChild(treatmentCell);
    
    // Section
    const sectionCell = document.createElement('td');
    sectionCell.style.width = '90px'; // Narrower width
    if (item.section) {
        const sectionSpan = document.createElement('span');
        sectionSpan.textContent = item.section;
        sectionSpan.className = 'badge';
        
        // Add appropriate class based on section
        switch (item.section.toUpperCase()) {
            case 'MAJORITY':
                sectionSpan.classList.add('badge-section-majority');
                break;
            case 'CONCURRING':
                sectionSpan.classList.add('badge-section-concurring');
                break;
            case 'DISSENTING':
                sectionSpan.classList.add('badge-section-dissenting');
                break;
            default:
                sectionSpan.classList.add('badge-section-other');
        }
        
        sectionCell.appendChild(sectionSpan);
    } else {
        sectionCell.textContent = '—';
    }
    row.appendChild(sectionCell);
    
    // Reasoning
    const reasoningCell = document.createElement('td');
    reasoningCell.className = 'reasoning-cell';
    // Remove the max display length limitation and let it flow naturally
    if (item.reasoning) {
        reasoningCell.textContent = item.reasoning;
        // Keep the full text as tooltip for easier reading
        reasoningCell.title = item.reasoning;
    } else {
        reasoningCell.textContent = '—';
    }
    row.appendChild(reasoningCell);
    
    // Add click event to show citation details (but not for rows in the expander panel)
    row.style.cursor = 'pointer';
    
    // Store the citation ID on the row for easy access
    row.setAttribute('data-citation-id', item.id);
    
    return row;
}

// Function to highlight a node in the network
function highlightNetworkNode(nodeId) {
    // Get the first network container
    const containerId = Object.keys(window.citationNetworkState)[0];
    if (!containerId) return;
    
    const networkState = window.citationNetworkState[containerId];
    if (!networkState || !networkState.data) return;
    
    // Find the node in the data
    const node = networkState.data.nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    // Call showDetails (which is defined in renderNetwork)
    const nodeElement = d3.select(`.nodes g`).filter(d => d.id === nodeId).node();
    if (nodeElement) {
        // Scroll to the network visualization
        document.getElementById(containerId).scrollIntoView({ behavior: 'smooth' });
        
        // Highlight the node by triggering a click event
        nodeElement.dispatchEvent(new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window
        }));
    }
}

// Function to sort table data by field and order
function sortData(data, field, order) {
    return [...data].sort((a, b) => {
        // First prioritize by treatment (POSITIVE, NEGATIVE, CAUTION above NEUTRAL)
        const treatmentPriority = {
            'POSITIVE': 0,
            'NEGATIVE': 1,
            'CAUTION': 2,
            'NEUTRAL': 3,
            undefined: 4
        };
        
        const aTreatment = treatmentPriority[a.treatment] !== undefined ? treatmentPriority[a.treatment] : 4;
        const bTreatment = treatmentPriority[b.treatment] !== undefined ? treatmentPriority[b.treatment] : 4;
        
        // If treatments are different, prioritize by treatment
        if (aTreatment !== bTreatment) {
            return aTreatment - bTreatment;
        }
        
        // If treatments are the same, prioritize by section (MAJORITY > CONCURRING > DISSENTING > OTHER)
        const sectionPriority = {
            'MAJORITY': 0,
            'CONCURRING': 1,
            'DISSENTING': 2,
            'OTHER': 3,
            undefined: 4
        };
        
        const sectionA = a.section ? a.section.toUpperCase() : 'OTHER';
        const sectionB = b.section ? b.section.toUpperCase() : 'OTHER';
        
        const aSectionPriority = sectionPriority[sectionA] !== undefined ? sectionPriority[sectionA] : 3;
        const bSectionPriority = sectionPriority[sectionB] !== undefined ? sectionPriority[sectionB] : 3;
        
        // If sections are different, prioritize by section
        if (aSectionPriority !== bSectionPriority) {
            return aSectionPriority - bSectionPriority;
        }
        
        // If both treatments and sections are the same, sort by the selected field
        let valueA, valueB;
        
        switch (field) {
            case 'citation_string':
                valueA = a.citation_string || '';
                valueB = b.citation_string || '';
                break;
            case 'relevance':
                valueA = a.relevance !== undefined ? a.relevance : -1;
                valueB = b.relevance !== undefined ? b.relevance : -1;
                break;
            case 'treatment':
                // Sort order: POSITIVE > NEUTRAL > CAUTION > NEGATIVE
                const treatmentOrder = {
                    'POSITIVE': 3,
                    'NEUTRAL': 2,
                    'CAUTION': 1,
                    'NEGATIVE': 0
                };
                valueA = treatmentOrder[a.treatment] !== undefined ? treatmentOrder[a.treatment] : -1;
                valueB = treatmentOrder[b.treatment] !== undefined ? treatmentOrder[b.treatment] : -1;
                break;
            case 'section':
                // Sort order: MAJORITY > CONCURRING > DISSENTING > OTHER
                const sectionOrder = {
                    'MAJORITY': 3,
                    'CONCURRING': 2,
                    'DISSENTING': 1,
                    'OTHER': 0
                };
                valueA = sectionOrder[sectionA] !== undefined ? sectionOrder[sectionA] : -1;
                valueB = sectionOrder[sectionB] !== undefined ? sectionOrder[sectionB] : -1;
                break;
            case 'reasoning':
                valueA = a.reasoning || '';
                valueB = b.reasoning || '';
                break;
            default:
                valueA = a[field] || '';
                valueB = b[field] || '';
        }
        
        // String comparison for strings
        if (typeof valueA === 'string' && typeof valueB === 'string') {
            return order === 'asc' ? 
                valueA.localeCompare(valueB) : 
                valueB.localeCompare(valueA);
        }
        
        // Numeric comparison for numbers
        return order === 'asc' ? valueA - valueB : valueB - valueA;
    });
}

// Function to group data by type
function groupDataByType(data) {
    const grouped = {};
    
    data.forEach(item => {
        if (!grouped[item.standardType]) {
            grouped[item.standardType] = [];
        }
        grouped[item.standardType].push(item);
    });
    
    return grouped;
}

// Function to group similar items within a type
function groupSimilarItems(items) {
    const similarityThreshold = 0.7; // Minimum similarity to group items
    const processed = new Set();
    const result = [];
    
    for (let i = 0; i < items.length; i++) {
        if (processed.has(i)) continue;
        
        const currentItem = items[i];
        const similarIndices = [i];
        processed.add(i);
        
        // Find similar items
        for (let j = i + 1; j < items.length; j++) {
            if (processed.has(j)) continue;
            
            // Use the imported calculateSimilarity function
            const similarity = calculateSimilarity(
                currentItem.citation_string || '',
                items[j].citation_string || ''
            );
            
            if (similarity >= similarityThreshold) {
                similarIndices.push(j);
                processed.add(j);
            }
        }
        
        // If we found similar items, create a group
        if (similarIndices.length > 1) {
            const similarItems = similarIndices.map(idx => items[idx]);
            result.push({
                similarItems: true,
                items: similarItems
            });
        } else {
            // Otherwise add as a regular item
            result.push(currentItem);
        }
    }
    
    return result;
}

// Function to update sort indicators
function updateSortIndicators(sortField, sortOrder) {
    // Reset all sort icons
    document.querySelectorAll('th[data-sort] .sort-icon').forEach(icon => {
        icon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4" />`;
    });
    
    // Update the active sort icon
    const activeHeader = document.querySelector(`th[data-sort="${sortField}"] .sort-icon`);
    if (activeHeader) {
        if (sortOrder === 'asc') {
            activeHeader.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4" />`;
        } else {
            activeHeader.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 4v12m0 0l4-4m-4 4l-4-4" />`;
        }
    }
}

// Set up event listeners for table interactions
function setupTableEventListeners(data, config) {
    // Current state
    let currentSort = { field: 'citation_string', order: 'asc' };
    let currentFilter = '';
    let currentTypeFilter = 'all';
    let groupSimilar = false;
    
    // Sort headers
    document.querySelectorAll('th[data-sort]').forEach(header => {
        header.addEventListener('click', function() {
            const sortField = this.getAttribute('data-sort');
            let sortOrder = 'asc';
            
            // Toggle sort order if clicking the same header
            if (sortField === currentSort.field) {
                sortOrder = currentSort.order === 'asc' ? 'desc' : 'asc';
            }
            
            // Update current sort
            currentSort = { field: sortField, order: sortOrder };
            
            // Re-render table
            const currentDirection = window.citationNetworkState[Object.keys(window.citationNetworkState)[0]].direction;
            const tableData = prepareTableData(data, currentDirection, config);
            renderTable(tableData, config, sortField, sortOrder, currentFilter, currentTypeFilter, groupSimilar);
        });
    });
    
    // Search input
    const searchInput = document.getElementById('citation-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            currentFilter = this.value;
            
            // Re-render table
            const currentDirection = window.citationNetworkState[Object.keys(window.citationNetworkState)[0]].direction;
            const tableData = prepareTableData(data, currentDirection, config);
            renderTable(
                tableData,
                config,
                currentSort.field, 
                currentSort.order, 
                currentFilter,
                currentTypeFilter,
                groupSimilar
            );
        });
    }
    
    // Type filter
    document.querySelectorAll('#type-filter-options a').forEach(option => {
        option.addEventListener('click', function() {
            // Update active state
            document.querySelectorAll('#type-filter-options a').forEach(opt => {
                opt.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update filter
            currentTypeFilter = this.getAttribute('data-type');
            
            // Re-render table
            const currentDirection = window.citationNetworkState[Object.keys(window.citationNetworkState)[0]].direction;
            const tableData = prepareTableData(data, currentDirection, config);
            renderTable(
                tableData,
                config,
                currentSort.field, 
                currentSort.order, 
                currentFilter,
                currentTypeFilter,
                groupSimilar
            );
        });
    });
    
    // Group similar toggle
    const groupSimilarToggle = document.getElementById('group-similar-toggle');
    if (groupSimilarToggle) {
        groupSimilarToggle.addEventListener('change', function() {
            groupSimilar = this.checked;
            
            // Re-render table
            const currentDirection = window.citationNetworkState[Object.keys(window.citationNetworkState)[0]].direction;
            const tableData = prepareTableData(data, currentDirection, config);
            renderTable(
                tableData,
                config,
                currentSort.field, 
                currentSort.order, 
                currentFilter,
                currentTypeFilter,
                groupSimilar
            );
        });
    }
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