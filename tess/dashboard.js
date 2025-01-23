// Simulated metrics data
function generateMetrics() {
    return {
        totalUsers: Math.floor(Math.random() * 1000) + 100,
        sales: Math.floor(Math.random() * 500) + 50,
        revenue: (Math.random() * 10000).toFixed(2),
    };
}

// Update metrics in the DOM
function updateMetrics() {
    const metrics = generateMetrics();
    d3.select("#metric1").text(metrics.totalUsers);
    d3.select("#metric2").text(metrics.sales);
    d3.select("#metric3").text(`$${metrics.revenue}`);
}

// Simulated chart data
function generateData(count = 10) {
    return Array.from({ length: count }, (_, i) => ({
        x: i + 1,
        y: Math.floor(Math.random() * 100) + 10,
    }));
}

// Bar Chart
function renderBarChart(data) {
    const svg = d3.select("#barChart").html("").append("svg");
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = svg.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = svg.node().getBoundingClientRect().height - margin.top - margin.bottom;

    const x = d3.scaleBand()
        .domain(data.map(d => d.x))
        .range([0, width])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.y)])
        .nice()
        .range([height, 0]);

    const chart = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    chart.selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", d => x(d.x))
        .attr("y", d => y(d.y))
        .attr("width", x.bandwidth())
        .attr("height", d => height - y(d.y))
        .attr("fill", "#007acc");

    chart.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));
    chart.append("g").call(d3.axisLeft(y));
}

// Line Chart
function renderLineChart(data) {
    const svg = d3.select("#lineChart").html("").append("svg");
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const width = svg.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = svg.node().getBoundingClientRect().height - margin.top - margin.bottom;

    const x = d3.scaleLinear()
        .domain(d3.extent(data, d => d.x))
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.y)])
        .nice()
        .range([height, 0]);

    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y));

    const chart = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    chart.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#007acc")
        .attr("stroke-width", 2)
        .attr("d", line);

    chart.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).ticks(data.length));
    chart.append("g").call(d3.axisLeft(y));
}

// Initialize the dashboard
function updateDashboard() {
    updateMetrics();
    const barData = generateData();
    const lineData = generateData();
    renderBarChart(barData);
    renderLineChart(lineData);
}

// Initial render
updateDashboard();
setInterval(updateDashboard, 5000); // Update every 5 seconds
