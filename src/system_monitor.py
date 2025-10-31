"""
System monitoring and health check utilities.
"""

import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class SystemMonitor:
    """Monitor system health and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    def record_request(self, response_time: float, success: bool = True):
        """Record API request metrics."""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if not success:
            self.error_count += 1
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        uptime = datetime.now() - self.start_time
        
        # Calculate average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        # Calculate error rate
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check model files
        model_status = self._check_model_files()
        
        return {
            'status': 'healthy',
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime).split('.')[0],
            'requests_total': self.request_count,
            'requests_per_minute': self._calculate_rpm(),
            'error_count': self.error_count,
            'error_rate_percent': round(error_rate, 2),
            'average_response_time_ms': round(avg_response_time * 1000, 2),
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'model_status': model_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_model_files(self) -> Dict[str, Any]:
        """Check status of model files."""
        model_paths = {
            'dna_classifier': 'models/trained_models/dna_classifier.pkl',
            'image_classifier': 'models/trained_models/image_classifier.pkl',
            'fusion_model': 'models/trained_models/fusion_model.pkl',
            'ensemble_model': 'models/trained_models/ensemble_model.pkl'
        }
        
        model_status = {}
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    stat = os.stat(path)
                    model_status[name] = {
                        'exists': True,
                        'size_mb': round(stat.st_size / (1024*1024), 2),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                except Exception as e:
                    model_status[name] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                model_status[name] = {
                    'exists': False,
                    'error': 'File not found'
                }
        
        return model_status
    
    def _calculate_rpm(self) -> float:
        """Calculate requests per minute."""
        if self.request_count == 0:
            return 0.0
        
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        return round(self.request_count / uptime_minutes, 2) if uptime_minutes > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last hour."""
        recent_times = self.response_times[-60:] if len(self.response_times) >= 60 else self.response_times
        
        if not recent_times:
            return {
                'recent_requests': 0,
                'average_response_time_ms': 0,
                'min_response_time_ms': 0,
                'max_response_time_ms': 0,
                'p95_response_time_ms': 0
            }
        
        recent_times_ms = [t * 1000 for t in recent_times]
        recent_times_ms.sort()
        
        p95_index = int(len(recent_times_ms) * 0.95)
        
        return {
            'recent_requests': len(recent_times),
            'average_response_time_ms': round(sum(recent_times_ms) / len(recent_times_ms), 2),
            'min_response_time_ms': round(min(recent_times_ms), 2),
            'max_response_time_ms': round(max(recent_times_ms), 2),
            'p95_response_time_ms': round(recent_times_ms[p95_index] if p95_index < len(recent_times_ms) else recent_times_ms[-1], 2)
        }

# Global system monitor instance
system_monitor = SystemMonitor()

def check_api_endpoints() -> Dict[str, Any]:
    """Check status of all API endpoints."""
    import requests
    
    endpoints = {
        'main': '/health',
        'dna': '/api/dna/health',
        'image': '/api/image/health',
        'prediction': '/api/prediction/health'
    }
    
    results = {}
    base_url = 'http://localhost:8000'
    
    for name, endpoint in endpoints.items():
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            response_time = time.time() - start_time
            
            results[name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time_ms': round(response_time * 1000, 2),
                'error': None
            }
        except Exception as e:
            results[name] = {
                'status': 'unhealthy',
                'status_code': None,
                'response_time_ms': None,
                'error': str(e)
            }
    
    return results

def generate_health_report() -> Dict[str, Any]:
    """Generate comprehensive health report."""
    return {
        'system_health': system_monitor.get_system_health(),
        'api_endpoints': check_api_endpoints(),
        'performance_summary': system_monitor.get_performance_summary(),
        'generated_at': datetime.now().isoformat()
    }

def save_health_report(filename: str = None):
    """Save health report to file."""
    if filename is None:
        filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = generate_health_report()
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filename
