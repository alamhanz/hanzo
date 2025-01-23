import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
export function updateMetrics(containerId, metrics) {
    const container = d3.select(`#${containerId}`);
    container.selectAll(".metric").remove(); // Clear existing metrics

    metrics.forEach(metric => {
        const metricDiv = container.append("div").attr("class", "metric");
        metricDiv.append("div").attr("class", "value").text(metric.value);
        metricDiv.append("div").text(metric.label);
    });
}
