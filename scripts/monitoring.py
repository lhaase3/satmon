#!/usr/bin/env python3
"""
Production Monitoring and Observability for SatMon
Real-time performance tracking and alerting
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('satmon_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceMetrics:
    """Performance metrics for anomaly detection algorithms"""
    algorithm: str
    execution_time: float
    throughput: float  # points per second
    memory_peak: float  # MB
    accuracy_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SystemMonitor:
    """Monitor system resources and performance"""
    
    def __init__(self, monitoring_interval: int = 5):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.is_monitoring = False
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }
        
    async def start_monitoring(self):
        """Start continuous system monitoring"""
        self.is_monitoring = True
        logger.info("🔍 Starting system monitoring...")
        
        while self.is_monitoring:
            try:
                snapshot = self.capture_metrics()
                self.metrics_history.append(snapshot)
                
                # Check for alerts
                await self.check_alerts(snapshot)
                
                # Log periodic status
                if len(self.metrics_history) % 12 == 0:  # Every minute if 5s interval
                    await self.log_status_summary()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def capture_metrics(self) -> MetricSnapshot:
        """Capture current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Network connections
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            connections = 0
        
        return MetricSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_sent=network_io.bytes_sent if network_io else 0,
            network_recv=network_io.bytes_recv if network_io else 0,
            active_connections=connections
        )
    
    async def check_alerts(self, snapshot: MetricSnapshot):
        """Check if any metrics exceed alert thresholds"""
        alerts = []
        
        if snapshot.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"🚨 HIGH CPU: {snapshot.cpu_percent:.1f}%")
        
        if snapshot.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"🚨 HIGH MEMORY: {snapshot.memory_percent:.1f}%")
        
        # Check disk usage
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        if disk_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"🚨 HIGH DISK USAGE: {disk_percent:.1f}%")
        
        for alert in alerts:
            logger.warning(alert)
    
    async def log_status_summary(self):
        """Log periodic status summary"""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)[-12:]  # Last minute
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_memory_mb = np.mean([m.memory_mb for m in recent_metrics])
        
        logger.info(
            f"📊 System Status - CPU: {avg_cpu:.1f}%, "
            f"Memory: {avg_memory:.1f}% ({avg_memory_mb:.0f}MB), "
            f"Samples: {len(self.metrics_history)}"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary over monitoring period"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        metrics = list(self.metrics_history)
        
        return {
            "monitoring_duration_minutes": len(metrics) * self.monitoring_interval / 60,
            "avg_cpu_percent": np.mean([m.cpu_percent for m in metrics]),
            "max_cpu_percent": max(m.cpu_percent for m in metrics),
            "avg_memory_percent": np.mean([m.memory_percent for m in metrics]),
            "max_memory_mb": max(m.memory_mb for m in metrics),
            "samples_collected": len(metrics),
            "last_updated": metrics[-1].timestamp.isoformat()
        }

class AlgorithmProfiler:
    """Profile anomaly detection algorithm performance"""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
        self.active_profiling: Dict[str, Dict] = {}
    
    def start_profiling(self, algorithm: str, data_size: int) -> str:
        """Start profiling an algorithm execution"""
        session_id = f"{algorithm}_{int(time.time())}"
        
        self.active_profiling[session_id] = {
            'algorithm': algorithm,
            'data_size': data_size,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        return session_id
    
    def end_profiling(self, session_id: str, accuracy_metrics: Optional[Dict] = None) -> PerformanceMetrics:
        """End profiling and calculate metrics"""
        if session_id not in self.active_profiling:
            raise ValueError(f"No active profiling session: {session_id}")
        
        session = self.active_profiling[session_id]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - session['start_time']
        throughput = session['data_size'] / execution_time if execution_time > 0 else 0
        memory_peak = end_memory - session['start_memory']
        
        metrics = PerformanceMetrics(
            algorithm=session['algorithm'],
            execution_time=execution_time,
            throughput=throughput,
            memory_peak=max(0, memory_peak),
            accuracy_score=accuracy_metrics.get('accuracy') if accuracy_metrics else None,
            precision=accuracy_metrics.get('precision') if accuracy_metrics else None,
            recall=accuracy_metrics.get('recall') if accuracy_metrics else None,
            f1_score=accuracy_metrics.get('f1_score') if accuracy_metrics else None
        )
        
        self.performance_history.append(metrics)
        del self.active_profiling[session_id]
        
        logger.info(
            f"🎯 {metrics.algorithm} Performance: "
            f"{execution_time:.3f}s, {throughput:.0f} points/s, "
            f"{memory_peak:.1f}MB peak"
        )
        
        return metrics
    
    def get_algorithm_benchmark(self, algorithm: str) -> Dict[str, Any]:
        """Get benchmark statistics for an algorithm"""
        algo_metrics = [m for m in self.performance_history if m.algorithm == algorithm]
        
        if not algo_metrics:
            return {"status": "no_data", "algorithm": algorithm}
        
        return {
            "algorithm": algorithm,
            "executions": len(algo_metrics),
            "avg_execution_time": np.mean([m.execution_time for m in algo_metrics]),
            "avg_throughput": np.mean([m.throughput for m in algo_metrics]),
            "avg_memory_peak": np.mean([m.memory_peak for m in algo_metrics]),
            "best_throughput": max(m.throughput for m in algo_metrics),
            "best_execution_time": min(m.execution_time for m in algo_metrics),
            "accuracy_stats": {
                "avg_precision": np.mean([m.precision for m in algo_metrics if m.precision is not None]) if any(m.precision for m in algo_metrics) else None,
                "avg_recall": np.mean([m.recall for m in algo_metrics if m.recall is not None]) if any(m.recall for m in algo_metrics) else None,
                "avg_f1": np.mean([m.f1_score for m in algo_metrics if m.f1_score is not None]) if any(m.f1_score for m in algo_metrics) else None
            }
        }

class HealthChecker:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.health_checks = {
            'database_connection': self.check_database,
            'disk_space': self.check_disk_space,
            'memory_usage': self.check_memory,
            'api_responsiveness': self.check_api_health,
            'data_pipeline': self.check_data_pipeline
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for check_name, check_func in self.health_checks.items():
            try:
                status, details = await check_func()
                results['checks'][check_name] = {
                    'status': status,
                    'details': details
                }
                
                if status != 'healthy':
                    results['overall_status'] = 'degraded'
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'details': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        # Log overall status
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'unhealthy': '❌'
        }
        
        logger.info(f"{status_emoji[results['overall_status']]} Health Check: {results['overall_status'].upper()}")
        
        return results
    
    async def check_database(self) -> tuple:
        """Check database connectivity"""
        try:
            # This would normally test actual database connection
            # For demo, simulate check
            await asyncio.sleep(0.1)  # Simulate connection test
            return 'healthy', 'Database connection successful'
        except Exception as e:
            return 'unhealthy', f'Database connection failed: {e}'
    
    async def check_disk_space(self) -> tuple:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 10:
                return 'unhealthy', f'Low disk space: {free_percent:.1f}% free'
            elif free_percent < 20:
                return 'degraded', f'Moderate disk space: {free_percent:.1f}% free'
            else:
                return 'healthy', f'Adequate disk space: {free_percent:.1f}% free'
                
        except Exception as e:
            return 'error', f'Disk check failed: {e}'
    
    async def check_memory(self) -> tuple:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return 'unhealthy', f'High memory usage: {memory.percent:.1f}%'
            elif memory.percent > 80:
                return 'degraded', f'Moderate memory usage: {memory.percent:.1f}%'
            else:
                return 'healthy', f'Normal memory usage: {memory.percent:.1f}%'
                
        except Exception as e:
            return 'error', f'Memory check failed: {e}'
    
    async def check_api_health(self) -> tuple:
        """Check API responsiveness"""
        try:
            # This would normally test actual API endpoints
            await asyncio.sleep(0.05)  # Simulate API call
            return 'healthy', 'API endpoints responding normally'
        except Exception as e:
            return 'unhealthy', f'API health check failed: {e}'
    
    async def check_data_pipeline(self) -> tuple:
        """Check data pipeline status"""
        try:
            # Check if required files exist
            required_files = ['scripts/detect_zscore.py', 'scripts/detect_isoforest.py']
            missing_files = []
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                return 'degraded', f'Missing files: {missing_files}'
            else:
                return 'healthy', 'Data pipeline components available'
                
        except Exception as e:
            return 'error', f'Pipeline check failed: {e}'

class MonitoringDashboard:
    """In-memory monitoring dashboard"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.profiler = AlgorithmProfiler()
        self.health_checker = HealthChecker()
        self.dashboard_data = {}
    
    async def start(self):
        """Start monitoring services"""
        logger.info("🚀 Starting SatMon Monitoring Dashboard...")
        
        # Start system monitoring in background
        monitor_task = asyncio.create_task(self.system_monitor.start_monitoring())
        
        # Periodic health checks every 5 minutes
        health_task = asyncio.create_task(self.periodic_health_checks())
        
        # Dashboard update task
        dashboard_task = asyncio.create_task(self.update_dashboard())
        
        try:
            await asyncio.gather(monitor_task, health_task, dashboard_task)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down monitoring...")
            self.system_monitor.is_monitoring = False
    
    async def periodic_health_checks(self):
        """Run health checks periodically"""
        while self.system_monitor.is_monitoring:
            try:
                health_results = await self.health_checker.run_health_check()
                self.dashboard_data['health'] = health_results
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def update_dashboard(self):
        """Update dashboard data periodically"""
        while self.system_monitor.is_monitoring:
            try:
                self.dashboard_data.update({
                    'system_performance': self.system_monitor.get_performance_summary(),
                    'algorithm_benchmarks': {
                        'zscore': self.profiler.get_algorithm_benchmark('zscore'),
                        'isolation_forest': self.profiler.get_algorithm_benchmark('isolation_forest'),
                        'lstm': self.profiler.get_algorithm_benchmark('lstm')
                    },
                    'last_updated': datetime.now().isoformat()
                })
                
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(30)
    
    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Get current dashboard data snapshot"""
        return self.dashboard_data.copy()
    
    def export_metrics(self, filepath: str):
        """Export current metrics to JSON file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_metrics': [m.to_dict() for m in self.system_monitor.metrics_history],
            'performance_metrics': [m.to_dict() for m in self.profiler.performance_history],
            'dashboard_snapshot': self.dashboard_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Metrics exported to {filepath}")

# Example usage and testing
async def demo_monitoring():
    """Demonstrate monitoring capabilities"""
    print("🔍 SatMon Monitoring Demo")
    print("=" * 50)
    
    dashboard = MonitoringDashboard()
    
    # Simulate some algorithm profiling
    print("\n📈 Simulating Algorithm Profiling...")
    
    # Profile Z-Score algorithm
    session_id = dashboard.profiler.start_profiling('zscore', 1000)
    await asyncio.sleep(0.5)  # Simulate processing time
    dashboard.profiler.end_profiling(session_id, {
        'precision': 0.364,
        'recall': 1.0,
        'f1_score': 0.533
    })
    
    # Profile Isolation Forest
    session_id = dashboard.profiler.start_profiling('isolation_forest', 1000)
    await asyncio.sleep(0.3)
    dashboard.profiler.end_profiling(session_id, {
        'precision': 0.095,
        'recall': 0.5,
        'f1_score': 0.159
    })
    
    print("\n🏥 Running Health Check...")
    health_results = await dashboard.health_checker.run_health_check()
    
    print(f"\nHealth Status: {health_results['overall_status'].upper()}")
    for check, result in health_results['checks'].items():
        status_emoji = {'healthy': '✅', 'degraded': '⚠️', 'unhealthy': '❌', 'error': '🔴'}
        print(f"  {status_emoji.get(result['status'], '❓')} {check}: {result['details']}")
    
    print("\n📊 Algorithm Benchmarks:")
    zscore_bench = dashboard.profiler.get_algorithm_benchmark('zscore')
    if zscore_bench.get('executions', 0) > 0:
        print(f"  Z-Score: {zscore_bench['avg_execution_time']:.3f}s avg, {zscore_bench['avg_throughput']:.0f} points/s")
    
    iso_bench = dashboard.profiler.get_algorithm_benchmark('isolation_forest')
    if iso_bench.get('executions', 0) > 0:
        print(f"  Isolation Forest: {iso_bench['avg_execution_time']:.3f}s avg, {iso_bench['avg_throughput']:.0f} points/s")
    
    # Export metrics
    dashboard.export_metrics('monitoring_demo_export.json')
    print("\n💾 Metrics exported to monitoring_demo_export.json")

if __name__ == "__main__":
    asyncio.run(demo_monitoring())