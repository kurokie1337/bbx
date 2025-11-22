# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-time Observability Dashboard
Web-based dashboard for monitoring BBX workflows
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from .observability import get_observability


app = FastAPI(title="BBX Observability Dashboard")

# WebSocket connections
active_connections: list[WebSocket] = []


@app.get("/")
async def dashboard():
    """Serve dashboard HTML"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>BBX Observability Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            font-size: 2rem;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-size: 0.875rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
        }

        .card-value {
            font-size: 2rem;
            font-weight: bold;
            color: #fff;
        }

        .card-subtitle {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 5px;
        }

        .metric-positive {
            color: #10b981;
        }

        .metric-negative {
            color: #ef4444;
        }

        .chart-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            height: 300px;
        }

        .logs-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            background: #0f172a;
        }

        .log-debug { border-left: 3px solid #64748b; }
        .log-info { border-left: 3px solid #3b82f6; }
        .log-warning { border-left: 3px solid #f59e0b; }
        .log-error { border-left: 3px solid #ef4444; }
        .log-critical { border-left: 3px solid #dc2626; }

        .timestamp {
            color: #64748b;
            margin-right: 10px;
        }

        .level {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-right: 10px;
        }

        .level-debug { background: #374151; color: #9ca3af; }
        .level-info { background: #1e40af; color: #dbeafe; }
        .level-warning { background: #b45309; color: #fef3c7; }
        .level-error { background: #991b1b; color: #fecaca; }
        .level-critical { background: #7f1d1d; color: #fecaca; }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-running { background: #3b82f6; animation: pulse 2s infinite; }
        .status-success { background: #10b981; }
        .status-error { background: #ef4444; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .trace-tree {
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
        }

        .trace-span {
            padding: 4px 8px;
            margin: 4px 0;
            background: #0f172a;
            border-radius: 4px;
            cursor: pointer;
        }

        .trace-span:hover {
            background: #334155;
        }

        .span-indent-0 { margin-left: 0; }
        .span-indent-1 { margin-left: 20px; }
        .span-indent-2 { margin-left: 40px; }
        .span-indent-3 { margin-left: 60px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ BBX Observability Dashboard</h1>

        <div class="grid">
            <div class="card">
                <div class="card-title">Total Executions</div>
                <div class="card-value" id="total-executions">0</div>
                <div class="card-subtitle">Workflow nodes executed</div>
            </div>

            <div class="card">
                <div class="card-title">Error Rate</div>
                <div class="card-value metric-negative" id="error-rate">0%</div>
                <div class="card-subtitle">Failed executions</div>
            </div>

            <div class="card">
                <div class="card-title">Avg Duration</div>
                <div class="card-value" id="avg-duration">0ms</div>
                <div class="card-subtitle">Average execution time</div>
            </div>

            <div class="card">
                <div class="card-title">Active Traces</div>
                <div class="card-value metric-positive" id="active-traces">0</div>
                <div class="card-subtitle">Currently running</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Execution Timeline</h3>
            <canvas id="timeline-chart"></canvas>
        </div>

        <div class="grid" style="grid-template-columns: 1fr 1fr;">
            <div>
                <h3 style="margin-bottom: 15px;">Recent Traces</h3>
                <div class="card" id="traces-container">
                    <div class="trace-tree" id="trace-tree">
                        No active traces
                    </div>
                </div>
            </div>

            <div>
                <h3 style="margin-bottom: 15px;">Live Logs</h3>
                <div class="logs-container" id="logs-container">
                    <!-- Logs will be inserted here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        const logsContainer = document.getElementById('logs-container');
        const maxLogs = 50;

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = () => {
                setTimeout(connect, 1000);
            };
        }

        function updateDashboard(data) {
            // Update metrics
            const metrics = data.metrics || {};
            const counters = metrics.counters || {};
            const histograms = metrics.histograms || {};

            // Total executions
            const executions = Object.entries(counters)
                .filter(([k, v]) => k.includes('executions'))
                .reduce((sum, [k, v]) => sum + v, 0);
            document.getElementById('total-executions').textContent = Math.round(executions);

            // Error rate
            document.getElementById('error-rate').textContent =
                (data.error_rate || 0).toFixed(2) + '%';

            // Avg duration
            document.getElementById('avg-duration').textContent =
                (data.avg_duration || 0).toFixed(2) + 'ms';

            // Active traces
            document.getElementById('active-traces').textContent =
                data.active_traces || 0;

            // Update logs
            if (data.recent_logs) {
                updateLogs(data.recent_logs);
            }

            // Update traces
            if (data.traces) {
                updateTraces(data.traces);
            }
        }

        function updateLogs(logs) {
            const container = document.getElementById('logs-container');

            logs.slice(-maxLogs).forEach(log => {
                const entry = document.createElement('div');
                entry.className = `log-entry log-${log.level.toLowerCase()}`;

                const timestamp = new Date(log.timestamp * 1000).toISOString();
                const level = log.level.toUpperCase();

                entry.innerHTML = `
                    <span class="timestamp">${timestamp}</span>
                    <span class="level level-${log.level.toLowerCase()}">${level}</span>
                    ${log.message}
                `;

                container.insertBefore(entry, container.firstChild);
            });

            // Keep only max logs
            while (container.children.length > maxLogs) {
                container.removeChild(container.lastChild);
            }
        }

        function updateTraces(traces) {
            const container = document.getElementById('trace-tree');

            if (!traces || traces.length === 0) {
                container.textContent = 'No active traces';
                return;
            }

            const recent = traces.slice(-10);
            container.innerHTML = recent.map(trace => {
                const status = trace.status || 'running';
                const duration = trace.duration_ms ? `${trace.duration_ms.toFixed(2)}ms` : 'running';

                return `
                    <div class="trace-span">
                        <span class="status-indicator status-${status}"></span>
                        ${trace.name} (${duration})
                    </div>
                `;
            }).join('');
        }

        // Connect on load
        connect();

        // Refresh every 1 second
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('refresh');
            }
        }, 1000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Wait for client message (refresh request)
            await websocket.receive_text()

            # Send dashboard data
            obs = get_observability()
            data = obs.get_dashboard_data()

            await websocket.send_json(data)

    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics"""
    obs = get_observability()
    return obs.metrics.get_summary()


@app.get("/api/traces")
async def get_traces():
    """Get all traces"""

@app.get("/api/export/{format}")
async def export_data(format: str):
    """Export telemetry data"""
    obs = get_observability()

    if format == "prometheus":
        from .observability import PrometheusExporter
        exporter = PrometheusExporter()
        data = obs.get_dashboard_data()
        return exporter(data)

    elif format == "json":
        await obs.export_all()
        return {"status": "exported"}

    else:
        return {"error": "Unknown format"}


def start_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Start the observability dashboard"""
    print("\n🚀 Starting BBX Observability Dashboard")
    print(f"📊 Dashboard: http://{host}:{port}")
    print(f"📈 Metrics API: http://{host}:{port}/api/metrics")
    print(f"🔍 Traces API: http://{host}:{port}/api/traces")
    print(f"📝 Logs API: http://{host}:{port}/api/logs\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_dashboard()
