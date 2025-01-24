import { renderBarChart } from './barChart.js';
import { renderLineChart } from './lineChart.js';
import { updateMetrics } from './metrics.js';


function generateMetrics() {
    return [
        { label: "Total Users", value: Math.floor(Math.random() * 1000) + 100 },
        { label: "Sales", value: Math.floor(Math.random() * 500) + 50 },
        { label: "Revenue", value: `$${(Math.random() * 10000).toFixed(2)}` }
    ];
}

function generateData(count = 10) {
    return Array.from({ length: count }, (_, i) => ({
        x: i + 1,
        y: Math.floor(Math.random() * 100) + 10,
    }));
}

function updateDashboard() {
    const metricsData = generateMetrics();
    updateMetrics("metricsContainer", metricsData);

    const barData = generateData();
    renderBarChart("barChartContainer", barData);

    const lineData = generateData();
    renderLineChart("lineChartContainer", lineData);
}

updateDashboard();
setInterval(updateDashboard, 5000);
