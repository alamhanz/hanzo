import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
export function renderLineChart(containerId, data) {
    const svg = d3.select(`#${containerId}`).html("").append("svg");
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
