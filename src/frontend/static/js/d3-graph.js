function createCitationGraph(data, container) {
    // Clear any existing visualization
    d3.select(container).html("");

    const width = document.querySelector(container).clientWidth;
    const height = document.querySelector(container).clientHeight;

    // Create SVG
    const svg = d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Extract nodes and links from the API data
    const nodes = data.nodes;
    const links = data.edges.map(e => ({
        source: e.source,
        target: e.target,
        treatment: e.treatment
    }));

    // Create a force simulation
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    // Define arrow marker for links
    svg.append("defs").selectAll("marker")
        .data(["arrow"])
        .enter().append("marker")
        .attr("id", d => d)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5");

    // Draw links
    const link = svg.append("g")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .attr("stroke", d => getLinkColor(d.treatment))
        .attr("stroke-width", 1.5)
        .attr("marker-end", "url(#arrow)");

    // Draw nodes
    const node = svg.append("g")
        .selectAll("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("r", d => d.type === "center" ? 8 : 5)
        .attr("fill", d => getNodeColor(d.type))
        .call(drag(simulation));

    // Add text labels
    const text = svg.append("g")
        .selectAll("text")
        .data(nodes)
        .enter().append("text")
        .text(d => shortenText(d.label))
        .attr("font-size", 10)
        .attr("dx", 12)
        .attr("dy", 4);

    // Add tooltips
    node.append("title")
        .text(d => `${d.label}\n${d.court_id || ''}\n${d.date_filed || ''}`);

    // Update positions on simulation tick
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        text
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    });

    // Add interactivity - node click opens case
    node.on("click", (event, d) => {
        window.location.href = `/case/${d.id}`;
    });

    // Add zoom behavior
    svg.call(d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
            svg.selectAll("g").attr("transform", event.transform);
        }));

    // Helper functions
    function shortenText(text) {
        return text.length > 20 ? text.substring(0, 17) + "..." : text;
    }

    function getNodeColor(type) {
        switch (type) {
            case "center": return "#3182CE"; // blue
            case "cited": return "#38A169";  // green
            case "citing": return "#E53E3E"; // red
            default: return "#718096";       // gray
        }
    }

    function getLinkColor(treatment) {
        switch (treatment) {
            case "POSITIVE": return "#38A169"; // green
            case "NEGATIVE": return "#E53E3E";  // red
            case "CAUTION": return "#F6AD55";  // orange
            default: return "#A0AEC0";         // gray
        }
    }

    // Implement drag behavior
    function drag(simulation) {
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

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
}

// Expose the createCitationGraph function to be used elsewhere
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = { createCitationGraph };
} else {
    window.createCitationGraph = createCitationGraph;
}