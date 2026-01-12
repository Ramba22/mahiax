"""
Alerting System for MAHIA Dashboard V3
Implements Slack/Discord webhook integration for critical training events
"""

import json
import time
import threading
import requests
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import hashlib
import hmac

class AlertingSystem:
    """Alerting system with webhook integration for critical training events"""
    
    def __init__(self, 
                 slack_webhook_url: Optional[str] = None,
                 discord_webhook_url: Optional[str] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 deduplication_window: int = 300):  # 5 minutes
        """
        Initialize the alerting system
        
        Args:
            slack_webhook_url: Slack webhook URL for alerts
            discord_webhook_url: Discord webhook URL for alerts
            alert_thresholds: Custom thresholds for different alert types
            deduplication_window: Time window (seconds) to prevent duplicate alerts
        """
        self.slack_webhook_url = slack_webhook_url
        self.discord_webhook_url = discord_webhook_url
        self.deduplication_window = deduplication_window
        
        # Alert thresholds
        self.alert_thresholds = {
            "gradient_explosion": 10.0,  # Gradient norm threshold
            "vram_leak": 0.9,  # VRAM usage threshold (90%)
            "training_stall": 300,  # Training stall threshold (seconds)
            "loss_spike": 2.0,  # Loss increase multiplier
            "accuracy_drop": 0.1,  # Accuracy decrease threshold
            "entropy_collapse": 0.1,  # Entropy threshold
            "learning_rate_anomaly": 10.0,  # LR change multiplier
        }
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
            
        # Alert storage
        self.alerts = deque(maxlen=1000)
        self.alert_history = deque(maxlen=10000)
        self.lock = threading.Lock()
        
        # Alert handlers
        self.alert_handlers = []
        
        # Deduplication cache
        self.alert_cache = {}
        
        print("🚨 AlertingSystem initialized")
        print(f"   Slack webhook: {'✅ Configured' if slack_webhook_url else '❌ Not configured'}")
        print(f"   Discord webhook: {'✅ Configured' if discord_webhook_url else '❌ Not configured'}")
        print(f"   Deduplication window: {deduplication_window}s")
        
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Add a custom alert handler
        
        Args:
            handler: Function to call when an alert is triggered
        """
        self.alert_handlers.append(handler)
        print(f"✅ Added custom alert handler. Total handlers: {len(self.alert_handlers)}")
        
    def remove_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Remove a custom alert handler
        
        Args:
            handler: Handler function to remove
        """
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            print(f"🗑️  Removed custom alert handler. Total handlers: {len(self.alert_handlers)}")
            
    def _is_duplicate_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if this is a duplicate alert within the deduplication window
        
        Args:
            alert: Alert to check
            
        Returns:
            True if duplicate, False otherwise
        """
        # Create a hash of the alert content for deduplication
        alert_content = f"{alert.get('type', '')}_{alert.get('component', '')}_{alert.get('message', '')}"
        alert_hash = hashlib.md5(alert_content.encode()).hexdigest()
        
        current_time = time.time()
        
        # Check if we've seen this alert recently
        if alert_hash in self.alert_cache:
            last_time = self.alert_cache[alert_hash]
            if current_time - last_time < self.deduplication_window:
                return True
                
        # Update cache
        self.alert_cache[alert_hash] = current_time
        
        # Clean up old cache entries
        expired_hashes = [
            hash_key for hash_key, timestamp in self.alert_cache.items()
            if current_time - timestamp > self.deduplication_window
        ]
        for hash_key in expired_hashes:
            del self.alert_cache[hash_key]
            
        return False
        
    def trigger_alert(self, 
                     alert_type: str,
                     message: str,
                     severity: str = "warning",
                     component: str = "unknown",
                     details: Optional[Dict[str, Any]] = None,
                     send_webhook: bool = True):
        """
        Trigger an alert
        
        Args:
            alert_type: Type of alert (e.g., "gradient_explosion", "vram_leak")
            message: Alert message
            severity: Alert severity ("info", "warning", "error", "critical")
            component: Component that triggered the alert
            details: Additional details about the alert
            send_webhook: Whether to send webhook notifications
        """
        alert = {
            "id": f"{alert_type}_{int(time.time() * 1000000)}",
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "component": component,
            "details": details or {}
        }
        
        # Check for duplicates
        if self._is_duplicate_alert(alert):
            print(f"🔄 Duplicate alert suppressed: {alert_type}")
            return
            
        # Store alert
        with self.lock:
            self.alerts.append(alert)
            self.alert_history.append(alert)
            
        # Print to console
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }.get(severity, "🔔")
        
        print(f"{severity_emoji} [{severity.upper()}] {component}: {message}")
        if details:
            print(f"   Details: {details}")
            
        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"⚠️  Error in custom alert handler: {e}")
                
        # Send webhook notifications
        if send_webhook:
            self._send_webhook_alert(alert)
            
    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """
        Send alert via webhook integrations
        
        Args:
            alert: Alert to send
        """
        # Send to Slack if configured
        if self.slack_webhook_url:
            try:
                self._send_slack_alert(alert)
            except Exception as e:
                print(f"⚠️  Failed to send Slack alert: {e}")
                
        # Send to Discord if configured
        if self.discord_webhook_url:
            try:
                self._send_discord_alert(alert)
            except Exception as e:
                print(f"⚠️  Failed to send Discord alert: {e}")
                
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """
        Send alert to Slack via webhook
        
        Args:
            alert: Alert to send
        """
        if not self.slack_webhook_url:
            return
            
        # Map severity to color
        color_map = {
            "info": "#439FE0",
            "warning": "#FFA500",
            "error": "#E01E5A",
            "critical": "#8B0000"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert["severity"], "#439FE0"),
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"🚨 MAHIA Training Alert: {alert['type']}"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {alert['severity'].upper()}\n*Component:* {alert['component']}\n*Message:* {alert['message']}"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Time:* {time.ctime(alert['timestamp'])}"
                            }
                        }
                    ]
                }
            ]
        }
        
        if alert["details"]:
            details_text = "\n".join([f"*{k}:* {v}" for k, v in alert["details"].items()])
            payload["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{details_text}"
                }
            })
            
        response = requests.post(self.slack_webhook_url, json=payload, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Slack webhook failed with status {response.status_code}")
            
    def _send_discord_alert(self, alert: Dict[str, Any]):
        """
        Send alert to Discord via webhook
        
        Args:
            alert: Alert to send
        """
        if not self.discord_webhook_url:
            return
            
        # Map severity to emoji
        emoji_map = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:"
        }
        
        emoji = emoji_map.get(alert["severity"], ":bell:")
        
        # Create embed
        embed = {
            "title": f"{emoji} MAHIA Training Alert: {alert['type']}",
            "description": alert["message"],
            "color": {
                "info": 0x439FE0,
                "warning": 0xFFA500,
                "error": 0xE01E5A,
                "critical": 0x8B0000
            }.get(alert["severity"], 0x439FE0),
            "fields": [
                {
                    "name": "Severity",
                    "value": alert["severity"].upper(),
                    "inline": True
                },
                {
                    "name": "Component",
                    "value": alert["component"],
                    "inline": True
                },
                {
                    "name": "Time",
                    "value": time.ctime(alert["timestamp"]),
                    "inline": True
                }
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(alert["timestamp"]))
        }
        
        if alert["details"]:
            details_value = "\n".join([f"**{k}:** {v}" for k, v in alert["details"].items()])
            embed["fields"].append({
                "name": "Details",
                "value": details_value
            })
            
        payload = {
            "embeds": [embed],
            "username": "MAHIA Alert System",
            "avatar_url": "https://cdn.discordapp.com/embed/avatars/0.png"
        }
        
        response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
        if response.status_code not in [200, 204]:
            raise Exception(f"Discord webhook failed with status {response.status_code}")
            
    def check_gradient_explosion(self, gradient_norm: float, threshold: Optional[float] = None):
        """
        Check for gradient explosion
        
        Args:
            gradient_norm: Current gradient norm
            threshold: Custom threshold (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["gradient_explosion"]
            
        if threshold is not None and gradient_norm > threshold:
            self.trigger_alert(
                alert_type="gradient_explosion",
                message=f"Gradient explosion detected! Norm: {gradient_norm:.4f} > {threshold}",
                severity="critical",
                component="optimizer",
                details={
                    "gradient_norm": gradient_norm,
                    "threshold": threshold
                }
            )
            
    def check_vram_leak(self, vram_usage_gb: float, vram_total_gb: float, 
                       threshold: Optional[float] = None):
        """
        Check for VRAM leak
        
        Args:
            vram_usage_gb: Current VRAM usage in GB
            vram_total_gb: Total VRAM in GB
            threshold: Custom threshold (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["vram_leak"]
            
        usage_ratio = vram_usage_gb / vram_total_gb if vram_total_gb > 0 else 0.0
        
        if threshold is not None and usage_ratio > threshold:
            self.trigger_alert(
                alert_type="vram_leak",
                message=f"High VRAM usage detected! {usage_ratio*100:.1f}% used",
                severity="warning",
                component="memory",
                details={
                    "vram_usage_gb": vram_usage_gb,
                    "vram_total_gb": vram_total_gb,
                    "usage_ratio": usage_ratio
                }
            )
            
    def check_training_stall(self, last_update_time: float, 
                           threshold: Optional[float] = None):
        """
        Check for training stall
        
        Args:
            last_update_time: Timestamp of last training update
            threshold: Custom threshold in seconds (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["training_stall"]
            
        time_since_update = time.time() - last_update_time
        
        if threshold is not None and time_since_update > threshold:
            self.trigger_alert(
                alert_type="training_stall",
                message=f"Training appears stalled! No updates for {time_since_update:.1f}s",
                severity="error",
                component="training_loop",
                details={
                    "stall_duration": time_since_update,
                    "threshold": threshold
                }
            )
            
    def check_loss_spike(self, current_loss: float, previous_loss: float,
                        threshold: Optional[float] = None):
        """
        Check for loss spike
        
        Args:
            current_loss: Current loss value
            previous_loss: Previous loss value
            threshold: Custom threshold multiplier (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["loss_spike"]
            
        if threshold is not None and previous_loss > 0 and current_loss > previous_loss * threshold:
            self.trigger_alert(
                alert_type="loss_spike",
                message=f"Loss spike detected! {previous_loss:.6f} → {current_loss:.6f}",
                severity="warning",
                component="loss_function",
                details={
                    "previous_loss": previous_loss,
                    "current_loss": current_loss,
                    "increase_ratio": current_loss / previous_loss if previous_loss > 0 else float('inf')
                }
            )
            
    def check_accuracy_drop(self, current_accuracy: float, previous_accuracy: float,
                          threshold: Optional[float] = None):
        """
        Check for accuracy drop
        
        Args:
            current_accuracy: Current accuracy value
            previous_accuracy: Previous accuracy value
            threshold: Custom threshold (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["accuracy_drop"]
            
        if threshold is not None and previous_accuracy > 0 and (previous_accuracy - current_accuracy) > threshold:
            self.trigger_alert(
                alert_type="accuracy_drop",
                message=f"Accuracy drop detected! {previous_accuracy:.4f} → {current_accuracy:.4f}",
                severity="warning",
                component="evaluation",
                details={
                    "previous_accuracy": previous_accuracy,
                    "current_accuracy": current_accuracy,
                    "drop_amount": previous_accuracy - current_accuracy
                }
            )
            
    def check_entropy_collapse(self, entropy: float, threshold: Optional[float] = None):
        """
        Check for entropy collapse
        
        Args:
            entropy: Current entropy value
            threshold: Custom threshold (uses default if None)
        """
        if threshold is None:
            threshold = self.alert_thresholds["entropy_collapse"]
            
        if threshold is not None and entropy < threshold:
            self.trigger_alert(
                alert_type="entropy_collapse",
                message=f"Entropy collapse detected! Entropy: {entropy:.6f} < {threshold}",
                severity="critical",
                component="entropy_monitor",
                details={
                    "entropy": entropy,
                    "threshold": threshold
                }
            )

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        with self.lock:
            return list(self.alerts)[-limit:]
            
    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get alerts by severity
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of alerts with specified severity
        """
        with self.lock:
            return [alert for alert in self.alerts if alert.get("severity") == severity]
            
    def get_alerts_by_type(self, alert_type: str) -> List[Dict[str, Any]]:
        """
        Get alerts by type
        
        Args:
            alert_type: Alert type to filter by
            
        Returns:
            List of alerts with specified type
        """
        with self.lock:
            return [alert for alert in self.alerts if alert.get("type") == alert_type]
            
    def clear_alerts(self):
        """Clear all alerts"""
        with self.lock:
            self.alerts.clear()
        print("🗑️  All alerts cleared")
        
    def generate_alert_report(self) -> str:
        """
        Generate a summary report of alerts
        
        Returns:
            Formatted report string
        """
        with self.lock:
            total_alerts = len(self.alert_history)
            recent_alerts = list(self.alerts)
            
        # Count alerts by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        # Count alerts by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert.get("type", "unknown")
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            
        report = f"""
🚨 MAHIA Alerting System Report
============================

📊 Summary:
   Total Alerts (Session): {total_alerts}
   Recent Alerts: {len(recent_alerts)}
   
📈 By Severity:
"""
        
        for severity, count in severity_counts.items():
            report += f"   {severity.upper()}: {count}\n"
            
        report += "\n🔍 By Type:\n"
        
        for alert_type, count in type_counts.items():
            report += f"   {alert_type}: {count}\n"
            
        if recent_alerts:
            report += "\n🔔 Recent Alerts:\n"
            # Show last 5 alerts
            for alert in recent_alerts[-5:]:
                report += f"   [{alert['severity'].upper()}] {alert['type']} - {alert['message']}\n"
                
        return report

# Global instance
_alerting_system = None

def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system

# Example usage
def example_alerting():
    """Example of alerting system usage"""
    print("🔧 Setting up alerting system example...")
    
    # Create alerting system
    alerting = get_alerting_system()
    
    # Configure with webhook URLs (these are just examples)
    alerting.slack_webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    alerting.discord_webhook_url = "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK"
    
    # Trigger some example alerts
    alerting.trigger_alert(
        alert_type="training_started",
        message="Training session started successfully",
        severity="info",
        component="training_manager"
    )
    
    # Check for various conditions
    alerting.check_gradient_explosion(15.0)  # This should trigger an alert
    alerting.check_vram_leak(8.5, 10.0)  # This should not trigger an alert
    alerting.check_entropy_collapse(0.05)  # This should trigger an alert
    
    # Print report
    print(alerting.generate_alert_report())
    
    print("✅ Alerting system example completed!")

if __name__ == "__main__":
    example_alerting()