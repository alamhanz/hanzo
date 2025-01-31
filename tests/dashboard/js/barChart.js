import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
export function renderBarChart(containerId, data) {
    const svg = d3.select(`#${containerId}`).html("").append("svg");
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
