"""
Real-time Monitoring Dashboard API
Level 1 Upgrade: Live tracking and visualization of ERA DAL operations

Provides a FastAPI/Flask-compatible REST API for:
- Real-time run progress monitoring
- Historical analytics
- Model performance metrics
- System health checks
"""

import json
import logging
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


@dataclass
class RunEvent:
    """Single run event for tracking"""
    timestamp: str
    event_type: str  # 'start', 'solver_complete', 'arbiter_complete', 'consensus', 'rebuttal', 'final'
    problem_id: str
    run_number: int
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ProblemSession:
    """Tracking session for a single problem"""
    problem_id: str
    problem_text: str
    pool_name: str
    started_at: str
    status: str = "running"  # 'running', 'completed', 'error'
    total_runs: int = 0
    completed_runs: int = 0
    events: List[RunEvent] = field(default_factory=list)
    final_result: Optional[Dict[str, Any]] = None
    ended_at: Optional[str] = None


class DashboardTracker:
    """
    Centralized tracker for all ERA DAL operations.
    Thread-safe singleton for collecting metrics.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.sessions: Dict[str, ProblemSession] = {}
        self.recent_events: deque = deque(maxlen=1000)
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self.global_stats = {
            "total_problems": 0,
            "total_runs": 0,
            "total_api_calls": 0,
            "avg_latency_ms": 0,
            "success_rate": 0.0,
            "started_at": datetime.now().isoformat()
        }
        self._event_lock = threading.Lock()
    
    def _generate_problem_id(self, problem_text: str) -> str:
        """Generate unique problem ID"""
        import hashlib
        hash_str = hashlib.md5(f"{problem_text}{time.time()}".encode()).hexdigest()[:8]
        return f"prob_{hash_str}"
    
    def start_session(
        self,
        problem_text: str,
        pool_name: str,
        total_runs: int
    ) -> str:
        """Start tracking a new problem session"""
        with self._event_lock:
            problem_id = self._generate_problem_id(problem_text)
            
            session = ProblemSession(
                problem_id=problem_id,
                problem_text=problem_text[:500],  # Truncate for storage
                pool_name=pool_name,
                started_at=datetime.now().isoformat(),
                total_runs=total_runs
            )
            
            self.sessions[problem_id] = session
            self.global_stats["total_problems"] += 1
            
            self._add_event(RunEvent(
                timestamp=datetime.now().isoformat(),
                event_type="start",
                problem_id=problem_id,
                run_number=0,
                data={"pool": pool_name, "total_runs": total_runs}
            ))
            
            logger.info(f"Dashboard: Started session {problem_id}")
            return problem_id
    
    def record_solver_complete(
        self,
        problem_id: str,
        run_number: int,
        model_id: str,
        answer_preview: str,
        confidence: float,
        latency_ms: int
    ):
        """Record a solver completion"""
        with self._event_lock:
            self._add_event(RunEvent(
                timestamp=datetime.now().isoformat(),
                event_type="solver_complete",
                problem_id=problem_id,
                run_number=run_number,
                data={
                    "model_id": model_id,
                    "answer_preview": answer_preview[:100],
                    "confidence": confidence,
                    "latency_ms": latency_ms
                }
            ))
            
            # Update model stats
            if model_id not in self.model_stats:
                self.model_stats[model_id] = {
                    "calls": 0,
                    "total_latency": 0,
                    "avg_confidence": 0,
                    "confidence_sum": 0
                }
            
            stats = self.model_stats[model_id]
            stats["calls"] += 1
            stats["total_latency"] += latency_ms
            stats["confidence_sum"] += confidence
            stats["avg_confidence"] = stats["confidence_sum"] / stats["calls"]
            
            self.global_stats["total_api_calls"] += 1
    
    def record_run_complete(
        self,
        problem_id: str,
        run_number: int,
        decision_mode: str,
        final_answer_preview: str
    ):
        """Record a complete run (all stages done)"""
        with self._event_lock:
            if problem_id in self.sessions:
                self.sessions[problem_id].completed_runs += 1
            
            self._add_event(RunEvent(
                timestamp=datetime.now().isoformat(),
                event_type="run_complete",
                problem_id=problem_id,
                run_number=run_number,
                data={
                    "decision_mode": decision_mode,
                    "answer_preview": final_answer_preview[:100]
                }
            ))
            
            self.global_stats["total_runs"] += 1
    
    def end_session(
        self,
        problem_id: str,
        status: str,
        final_result: Optional[Dict[str, Any]] = None
    ):
        """End a problem session"""
        with self._event_lock:
            if problem_id in self.sessions:
                session = self.sessions[problem_id]
                session.status = status
                session.ended_at = datetime.now().isoformat()
                session.final_result = final_result
            
            self._add_event(RunEvent(
                timestamp=datetime.now().isoformat(),
                event_type="final",
                problem_id=problem_id,
                run_number=0,
                data={"status": status}
            ))
    
    def _add_event(self, event: RunEvent):
        """Add event to recent events queue"""
        self.recent_events.append(event)
    
    def get_session(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get session details"""
        if problem_id in self.sessions:
            session = self.sessions[problem_id]
            return {
                "problem_id": session.problem_id,
                "problem_text": session.problem_text,
                "pool_name": session.pool_name,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "status": session.status,
                "progress": f"{session.completed_runs}/{session.total_runs}",
                "events_count": len(session.events)
            }
        return None
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events"""
        events = list(self.recent_events)[-limit:]
        return [asdict(e) for e in events]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        result = {}
        for model_id, stats in self.model_stats.items():
            result[model_id] = {
                "calls": stats["calls"],
                "avg_latency_ms": stats["total_latency"] / stats["calls"] if stats["calls"] > 0 else 0,
                "avg_confidence": stats["avg_confidence"]
            }
        return result
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete dashboard summary"""
        active_sessions = sum(1 for s in self.sessions.values() if s.status == "running")
        completed_sessions = sum(1 for s in self.sessions.values() if s.status == "completed")
        
        return {
            "status": "healthy",
            "uptime_since": self.global_stats["started_at"],
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "total_problems": self.global_stats["total_problems"],
            "total_runs": self.global_stats["total_runs"],
            "total_api_calls": self.global_stats["total_api_calls"],
            "models_used": len(self.model_stats),
            "model_stats": self.get_model_stats()
        }


# Singleton tracker instance
tracker = DashboardTracker()


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for dashboard API"""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        response_data = None
        
        if path == "/api/health":
            response_data = {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        elif path == "/api/dashboard":
            response_data = tracker.get_dashboard_summary()
        
        elif path == "/api/events":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", [50])[0])
            response_data = {"events": tracker.get_recent_events(limit)}
        
        elif path == "/api/models":
            response_data = {"models": tracker.get_model_stats()}
        
        elif path.startswith("/api/session/"):
            problem_id = path.split("/")[-1]
            session = tracker.get_session(problem_id)
            if session:
                response_data = session
            else:
                self.send_error(404, "Session not found")
                return
        
        elif path == "/" or path == "/dashboard":
            # Serve simple HTML dashboard
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self._get_dashboard_html().encode())
            return
        
        else:
            self.send_error(404, "Not found")
            return
        
        # Send JSON response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_data, indent=2).encode())
    
    def _get_dashboard_html(self) -> str:
        """Generate simple dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERA DAL Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .stat-card { transition: transform 0.2s; }
        .stat-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-blue-400">🎯 ERA DAL Dashboard</h1>
            <p class="text-gray-400">Real-time Monitoring & Analytics</p>
        </header>
        
        <div id="stats" class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="stat-card bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Total Problems</div>
                <div id="total-problems" class="text-2xl font-bold text-blue-400">-</div>
            </div>
            <div class="stat-card bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Total Runs</div>
                <div id="total-runs" class="text-2xl font-bold text-green-400">-</div>
            </div>
            <div class="stat-card bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">API Calls</div>
                <div id="api-calls" class="text-2xl font-bold text-yellow-400">-</div>
            </div>
            <div class="stat-card bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Models Used</div>
                <div id="models-used" class="text-2xl font-bold text-purple-400">-</div>
            </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-xl font-semibold mb-4">📊 Model Performance</h2>
                <div id="model-stats" class="space-y-2"></div>
            </div>
            
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-xl font-semibold mb-4">📝 Recent Events</h2>
                <div id="events" class="space-y-2 max-h-64 overflow-y-auto"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function fetchData() {
            try {
                const [dashRes, eventsRes] = await Promise.all([
                    fetch('/api/dashboard'),
                    fetch('/api/events?limit=10')
                ]);
                
                const dash = await dashRes.json();
                const events = await eventsRes.json();
                
                document.getElementById('total-problems').textContent = dash.total_problems;
                document.getElementById('total-runs').textContent = dash.total_runs;
                document.getElementById('api-calls').textContent = dash.total_api_calls;
                document.getElementById('models-used').textContent = dash.models_used;
                
                // Model stats
                const modelDiv = document.getElementById('model-stats');
                modelDiv.innerHTML = Object.entries(dash.model_stats || {}).map(([model, stats]) => `
                    <div class="flex justify-between items-center bg-gray-700 rounded p-2">
                        <span class="text-sm truncate" title="${model}">${model.split('/').pop()}</span>
                        <span class="text-xs text-gray-400">${stats.calls} calls, ${Math.round(stats.avg_latency_ms)}ms avg</span>
                    </div>
                `).join('') || '<p class="text-gray-500">No model data yet</p>';
                
                // Events
                const eventsDiv = document.getElementById('events');
                eventsDiv.innerHTML = (events.events || []).map(e => `
                    <div class="bg-gray-700 rounded p-2 text-sm">
                        <span class="text-blue-400">[${e.event_type}]</span>
                        <span class="text-gray-400">${e.problem_id}</span>
                        ${e.run_number ? `<span class="text-green-400">Run ${e.run_number}</span>` : ''}
                    </div>
                `).join('') || '<p class="text-gray-500">No events yet</p>';
                
            } catch (err) {
                console.error('Fetch error:', err);
            }
        }
        
        fetchData();
        setInterval(fetchData, 3000);  // Refresh every 3 seconds
    </script>
</body>
</html>
"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def start_dashboard_server(port: int = 8080) -> threading.Thread:
    """
    Start the dashboard server in a background thread.
    
    Args:
        port: Port to listen on
        
    Returns:
        Server thread
    """
    def run_server():
        with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
            logger.info(f"Dashboard server running at http://localhost:{port}")
            httpd.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread


# CLI helper
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Starting ERA DAL Dashboard on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        httpd.serve_forever()
