#!/usr/bin/env python3
"""
Simple monitoring dashboard for AutoML Framework
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any


class MonitoringDashboard:
    """Simple CLI monitoring dashboard"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        async with self.session.get(f"{self.api_url}/api/v1/monitoring/health") as response:
            return await response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        async with self.session.get(f"{self.api_url}/api/v1/monitoring/metrics") as response:
            return await response.json()
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        async with self.session.get(f"{self.api_url}/api/v1/monitoring/metrics/system") as response:
            return await response.json()
    
    async def get_alerts(self) -> Dict[str, Any]:
        """Get active alerts"""
        async with self.session.get(f"{self.api_url}/api/v1/monitoring/alerts") as response:
            return await response.json()
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        async with self.session.get(f"{self.api_url}/api/v1/monitoring/status") as response:
            return await response.json()
    
    def print_header(self):
        """Print dashboard header"""
        print("\n" + "="*80)
        print(f"AutoML Framework Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def print_health_status(self, health_data: Dict[str, Any]):
        """Print health status"""
        print(f"\n🏥 HEALTH STATUS: {health_data['overall_status'].upper()}")
        print("-" * 40)
        
        for check_name, check_data in health_data['checks'].items():
            status_emoji = {
                'healthy': '✅',
                'unhealthy': '❌',
                'degraded': '⚠️',
                'unknown': '❓'
            }.get(check_data['status'], '❓')
            
            print(f"{status_emoji} {check_name}: {check_data['status']} ({check_data['response_time_ms']:.1f}ms)")
            if check_data['status'] != 'healthy':
                print(f"   Message: {check_data['message']}")
    
    def print_system_metrics(self, system_data: Dict[str, Any]):
        """Print system metrics"""
        print(f"\n💻 SYSTEM RESOURCES")
        print("-" * 40)
        
        # CPU
        cpu_emoji = "🔥" if system_data['cpu_percent'] > 80 else "✅"
        print(f"{cpu_emoji} CPU: {system_data['cpu_percent']:.1f}%")
        
        # Memory
        mem_emoji = "🔥" if system_data['memory_percent'] > 80 else "✅"
        print(f"{mem_emoji} Memory: {system_data['memory_percent']:.1f}% ({system_data['memory_used_gb']:.1f}GB / {system_data['memory_total_gb']:.1f}GB)")
        
        # Disk
        disk_emoji = "🔥" if system_data['disk_percent'] > 90 else "✅"
        print(f"{disk_emoji} Disk: {system_data['disk_percent']:.1f}% ({system_data['disk_used_gb']:.1f}GB / {system_data['disk_total_gb']:.1f}GB)")
        
        # GPU
        if system_data['gpu_utilization']:
            print(f"🎮 GPUs: {len(system_data['gpu_utilization'])} available")
            for i, util in enumerate(system_data['gpu_utilization']):
                gpu_emoji = "🔥" if util > 90 else "✅"
                mem_used = system_data['gpu_memory_used'][i] if i < len(system_data['gpu_memory_used']) else 0
                mem_total = system_data['gpu_memory_total'][i] if i < len(system_data['gpu_memory_total']) else 0
                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                print(f"   {gpu_emoji} GPU {i}: {util:.1f}% util, {mem_percent:.1f}% mem ({mem_used:.0f}MB / {mem_total:.0f}MB)")
    
    def print_api_metrics(self, metrics_data: Dict[str, Any]):
        """Print API metrics"""
        print(f"\n🌐 API METRICS")
        print("-" * 40)
        
        total_requests = metrics_data['counters'].get('api.requests.total', 0)
        requests_per_minute = metrics_data['api_requests_per_minute']
        
        print(f"📊 Total Requests: {total_requests:,}")
        print(f"⚡ Requests/min: {requests_per_minute:.1f}")
        
        # Error rate
        error_requests = sum(v for k, v in metrics_data['counters'].items() if 'error' in k.lower())
        error_rate = (error_requests / max(total_requests, 1)) * 100
        error_emoji = "🔥" if error_rate > 5 else "✅"
        print(f"{error_emoji} Error Rate: {error_rate:.2f}%")
    
    def print_alerts(self, alerts_data: list):
        """Print active alerts"""
        if not alerts_data:
            print(f"\n🔔 ALERTS: None active ✅")
            return
        
        print(f"\n🚨 ACTIVE ALERTS: {len(alerts_data)}")
        print("-" * 40)
        
        for alert in alerts_data:
            severity_emoji = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }.get(alert['severity'], '❓')
            
            print(f"{severity_emoji} {alert['name']} ({alert['severity'].upper()})")
            print(f"   {alert['message']}")
            print(f"   Time: {alert['timestamp']}")
    
    async def run_dashboard(self, refresh_interval: int = 30):
        """Run the monitoring dashboard"""
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")
                
                self.print_header()
                
                try:
                    # Get all monitoring data
                    health_data = await self.get_health_status()
                    system_data = await self.get_system_metrics()
                    metrics_data = await self.get_metrics()
                    alerts_data = await self.get_alerts()
                    
                    # Print sections
                    self.print_health_status(health_data)
                    self.print_system_metrics(system_data)
                    self.print_api_metrics(metrics_data)
                    self.print_alerts(alerts_data)
                    
                except aiohttp.ClientError as e:
                    print(f"\n❌ Error connecting to API: {e}")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                
                print(f"\n🔄 Refreshing in {refresh_interval} seconds... (Ctrl+C to exit)")
                await asyncio.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped by user")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Framework Monitoring Dashboard")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    async with MonitoringDashboard(args.api_url) as dashboard:
        await dashboard.run_dashboard(args.refresh)


if __name__ == "__main__":
    asyncio.run(main())