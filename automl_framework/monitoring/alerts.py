"""
Alert management system for AutoML Framework
"""

import asyncio
import smtplib
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from automl_framework.utils.logging import get_logger
from automl_framework.core.config import get_config

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    timestamp: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel:
    """Base class for alert notification channels"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification"""
        raise NotImplementedError


class EmailAlertChannel(AlertChannel):
    """Email alert notification channel"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, 
                 from_email: str, to_emails: List[str]):
        super().__init__("email")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[AutoML Alert - {alert.severity.value.upper()}] {alert.name}"
            
            # Email body
            body = f"""
AutoML Framework Alert

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Source: {alert.source}
Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            logger.info(f"Alert email sent successfully: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}", alert_id=alert.id)
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook alert notification channel"""
    
    def __init__(self, name: str, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            import aiohttp
            
            payload = {
                'alert_id': alert.id,
                'name': alert.name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Alert webhook sent successfully: {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}: {alert.id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send alert webhook: {e}", alert_id=alert.id)
            return False


class SlackAlertChannel(AlertChannel):
    """Slack alert notification channel"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        super().__init__("slack")
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        try:
            import aiohttp
            
            # Color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
            }
            
            payload = {
                "channel": self.channel,
                "username": "AutoML Alert Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#ff0000"),
                        "title": f"{alert.name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": True
                            },
                            {
                                "title": "Alert ID",
                                "value": alert.id,
                                "short": True
                            }
                        ],
                        "footer": "AutoML Framework",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Alert sent to Slack successfully: {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack webhook returned status {response.status}: {alert.id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send alert to Slack: {e}", alert_id=alert.id)
            return False


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_channels: List[AlertChannel] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Register default alert rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        # High CPU usage alert
        self.register_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda metrics: metrics.get('system.cpu.percent', 0) > 90,
            severity=AlertSeverity.HIGH,
            message_template="High CPU usage detected: {cpu_percent:.1f}%",
            cooldown_minutes=10
        ))
        
        # High memory usage alert
        self.register_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda metrics: metrics.get('system.memory.percent', 0) > 90,
            severity=AlertSeverity.HIGH,
            message_template="High memory usage detected: {memory_percent:.1f}%",
            cooldown_minutes=10
        ))
        
        # High disk usage alert
        self.register_rule(AlertRule(
            name="high_disk_usage",
            condition=lambda metrics: metrics.get('system.disk.percent', 0) > 95,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical disk usage detected: {disk_percent:.1f}%",
            cooldown_minutes=5
        ))
        
        # Database connection failure
        self.register_rule(AlertRule(
            name="database_connection_failure",
            condition=lambda metrics: metrics.get('database.connection_errors', 0) > 0,
            severity=AlertSeverity.CRITICAL,
            message_template="Database connection failures detected",
            cooldown_minutes=5
        ))
        
        # Training job failures
        self.register_rule(AlertRule(
            name="high_training_failure_rate",
            condition=lambda metrics: (
                metrics.get('training.failed_jobs', 0) > 0 and
                metrics.get('training.failed_jobs', 0) / max(metrics.get('training.total_jobs', 1), 1) > 0.5
            ),
            severity=AlertSeverity.HIGH,
            message_template="High training job failure rate detected",
            cooldown_minutes=15
        ))
        
        # GPU temperature alert
        self.register_rule(AlertRule(
            name="gpu_overheating",
            condition=lambda metrics: any(
                temp > 85 for key, temp in metrics.items()
                if 'gpu' in key and 'temperature' in key
            ),
            severity=AlertSeverity.HIGH,
            message_template="GPU overheating detected",
            cooldown_minutes=5
        ))
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def unregister_rule(self, name: str):
        """Unregister an alert rule"""
        if name in self.alert_rules:
            del self.alert_rules[name]
            logger.info(f"Unregistered alert rule: {name}")
    
    def add_channel(self, channel: AlertChannel):
        """Add alert notification channel"""
        self.alert_channels.append(channel)
        logger.info(f"Added alert channel: {channel.name}")
    
    def remove_channel(self, channel_name: str):
        """Remove alert notification channel"""
        self.alert_channels = [
            ch for ch in self.alert_channels 
            if ch.name != channel_name
        ]
        logger.info(f"Removed alert channel: {channel_name}")
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown period
                if rule_name in self.last_alert_times:
                    time_since_last = datetime.now() - self.last_alert_times[rule_name]
                    if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                        continue
                
                # Evaluate condition
                if rule.condition(metrics):
                    await self._trigger_alert(rule, metrics)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(datetime.now().timestamp())}"
        
        # Format message
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = rule.message_template
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            source="automl_framework",
            timestamp=datetime.now(),
            metadata={
                'rule_metadata': rule.metadata,
                'triggering_metrics': metrics
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = datetime.now()
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(
            f"Alert triggered: {rule.name}",
            alert_id=alert_id,
            severity=rule.severity.value,
            message=message
        )
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through all channels"""
        if not self.alert_channels:
            logger.warning(f"No alert channels configured for alert: {alert.id}")
            return
        
        tasks = []
        for channel in self.alert_channels:
            tasks.append(channel.send_alert(alert))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        logger.info(
            f"Alert notifications sent: {success_count}/{len(self.alert_channels)}",
            alert_id=alert.id
        )
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id}", acknowledged_by=acknowledged_by)
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(hours=24)
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'active_alerts_count': len(active_alerts),
            'recent_alerts_count': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'alert_channels_count': len(self.alert_channels),
            'alert_rules_count': len(self.alert_rules),
            'timestamp': datetime.now().isoformat()
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def initialize_alert_system():
    """Initialize alert system with configuration"""
    alert_manager = get_alert_manager()
    
    try:
        config = get_config()
        
        # Add email channel if configured
        if hasattr(config, 'email_alerts') and config.email_alerts.enabled:
            email_channel = EmailAlertChannel(
                smtp_host=config.email_alerts.smtp_host,
                smtp_port=config.email_alerts.smtp_port,
                username=config.email_alerts.username,
                password=config.email_alerts.password,
                from_email=config.email_alerts.from_email,
                to_emails=config.email_alerts.to_emails
            )
            alert_manager.add_channel(email_channel)
        
        # Add Slack channel if configured
        if hasattr(config, 'slack_alerts') and config.slack_alerts.enabled:
            slack_channel = SlackAlertChannel(
                webhook_url=config.slack_alerts.webhook_url,
                channel=config.slack_alerts.channel
            )
            alert_manager.add_channel(slack_channel)
        
        logger.info("Alert system initialized")
        
    except Exception as e:
        logger.warning(f"Failed to initialize some alert channels: {e}")
    
    return alert_manager