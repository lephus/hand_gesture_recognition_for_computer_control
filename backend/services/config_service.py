#!/usr/bin/env python3
"""
Configuration Service
Manages gesture-to-action mappings and user preferences
Author: [Your Name]
Date: October 2025
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigService:
    """Configuration management service"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration service
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "gesture_actions": {
                "one_finger": "chrome",
                "two_fingers": "firefox", 
                "three_fingers": "vscode",
                "four_fingers": "terminal",
                "five_fingers": "calculator",
                "rotate_clockwise": "volume_up",
                "rotate_counterclockwise": "volume_down",
                "x_gesture": "close_window",
                "swipe_left": "previous_tab",
                "swipe_right": "next_tab",
                "neutral": "none"
            },
            "gesture_settings": {
                "confidence_threshold": 0.8,
                "action_cooldown": 1.0,
                "enable_gestures": {
                    "one_finger": True,
                    "two_fingers": True,
                    "three_fingers": True,
                    "four_fingers": True,
                    "five_fingers": True,
                    "rotate_clockwise": True,
                    "rotate_counterclockwise": True,
                    "x_gesture": True,
                    "swipe_left": True,
                    "swipe_right": True,
                    "neutral": False
                }
            },
            "system_settings": {
                "fps_limit": 15,
                "inference_batch_size": 1,
                "log_level": "INFO",
                "enable_performance_tracking": True
            },
            "ui_settings": {
                "show_confidence": True,
                "show_gesture_name": True,
                "show_action_feedback": True,
                "theme": "dark"
            }
        }
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with default config
                    self._merge_config(self.config, loaded_config)
                print(f"✅ Configuration loaded from: {self.config_path}")
            else:
                print(f"📄 Creating default configuration: {self.config_path}")
                self.save_config()
        except Exception as e:
            print(f"⚠️  Error loading config: {e}, using defaults")
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]):
        """Merge loaded config with default config"""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def get_gesture_action(self, gesture: str) -> str:
        """Get action for specific gesture"""
        return self.config['gesture_actions'].get(gesture, 'none')
    
    def set_gesture_action(self, gesture: str, action: str) -> bool:
        """Set action for specific gesture"""
        try:
            if gesture not in self.config['gesture_actions']:
                return False
            
            self.config['gesture_actions'][gesture] = action
            return True
        except Exception as e:
            print(f"❌ Error setting gesture action: {e}")
            return False
    
    def is_gesture_enabled(self, gesture: str) -> bool:
        """Check if gesture is enabled"""
        return self.config['gesture_settings']['enable_gestures'].get(gesture, False)
    
    def enable_gesture(self, gesture: str, enabled: bool = True) -> bool:
        """Enable or disable gesture"""
        try:
            if gesture not in self.config['gesture_settings']['enable_gestures']:
                return False
            
            self.config['gesture_settings']['enable_gestures'][gesture] = enabled
            return True
        except Exception as e:
            print(f"❌ Error enabling/disabling gesture: {e}")
            return False
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold"""
        return self.config['gesture_settings']['confidence_threshold']
    
    def set_confidence_threshold(self, threshold: float) -> bool:
        """Set confidence threshold"""
        try:
            if 0.0 <= threshold <= 1.0:
                self.config['gesture_settings']['confidence_threshold'] = threshold
                return True
            return False
        except Exception as e:
            print(f"❌ Error setting confidence threshold: {e}")
            return False
    
    def get_action_cooldown(self) -> float:
        """Get action cooldown in seconds"""
        return self.config['gesture_settings']['action_cooldown']
    
    def set_action_cooldown(self, cooldown: float) -> bool:
        """Set action cooldown in seconds"""
        try:
            if cooldown >= 0:
                self.config['gesture_settings']['action_cooldown'] = cooldown
                return True
            return False
        except Exception as e:
            print(f"❌ Error setting action cooldown: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Validate configuration
            if not self._validate_config(new_config):
                return False
            
            # Merge new config
            self._merge_config(self.config, new_config)
            return True
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        try:
            # Check required sections
            required_sections = ['gesture_actions', 'gesture_settings', 'system_settings', 'ui_settings']
            for section in required_sections:
                if section not in config:
                    print(f"❌ Missing required section: {section}")
                    return False
            
            # Validate gesture actions
            valid_actions = [
                'chrome', 'firefox', 'safari', 'vscode', 'terminal', 'calculator',
                'volume_up', 'volume_down', 'close_window', 'next_tab', 'previous_tab', 'none'
            ]
            
            for gesture, action in config.get('gesture_actions', {}).items():
                if action not in valid_actions:
                    print(f"❌ Invalid action for {gesture}: {action}")
                    return False
            
            # Validate confidence threshold
            confidence = config.get('gesture_settings', {}).get('confidence_threshold', 0.8)
            if not 0.0 <= confidence <= 1.0:
                print(f"❌ Invalid confidence threshold: {confidence}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Configuration saved to: {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config = self._load_default_config()
            return self.save_config()
        except Exception as e:
            print(f"❌ Error resetting config: {e}")
            return False
    
    def get_available_actions(self) -> list:
        """Get list of available actions"""
        return [
            'chrome', 'firefox', 'safari', 'vscode', 'terminal', 'calculator',
            'volume_up', 'volume_down', 'close_window', 'next_tab', 'previous_tab', 'none'
        ]
    
    def get_available_gestures(self) -> list:
        """Get list of available gestures"""
        return list(self.config['gesture_actions'].keys())
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to file"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Configuration exported to: {export_file}")
            return True
        except Exception as e:
            print(f"❌ Error exporting config: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """Import configuration from file"""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                print(f"❌ Import file not found: {import_file}")
                return False
            
            with open(import_file, 'r') as f:
                imported_config = json.load(f)
            
            if self._validate_config(imported_config):
                self.config = imported_config
                return self.save_config()
            else:
                print("❌ Invalid configuration file")
                return False
        except Exception as e:
            print(f"❌ Error importing config: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Configuration Service")
    print("=" * 50)
    
    # Initialize config service
    config_service = ConfigService("test_config.json")
    
    # Test getting config
    config = config_service.get_config()
    print(f"📄 Current config sections: {list(config.keys())}")
    
    # Test gesture actions
    print(f"\n🎮 Gesture Actions:")
    for gesture in config_service.get_available_gestures():
        action = config_service.get_gesture_action(gesture)
        enabled = config_service.is_gesture_enabled(gesture)
        print(f"   {gesture}: {action} ({'enabled' if enabled else 'disabled'})")
    
    # Test settings
    print(f"\n⚙️  Settings:")
    print(f"   Confidence threshold: {config_service.get_confidence_threshold()}")
    print(f"   Action cooldown: {config_service.get_action_cooldown()}s")
    
    # Test updating config
    print(f"\n🔄 Testing config update...")
    new_config = {
        "gesture_actions": {
            "one_finger": "firefox",
            "two_fingers": "chrome"
        },
        "gesture_settings": {
            "confidence_threshold": 0.9
        }
    }
    
    if config_service.update_config(new_config):
        print("✅ Config updated successfully")
        print(f"   One finger now opens: {config_service.get_gesture_action('one_finger')}")
        print(f"   New confidence threshold: {config_service.get_confidence_threshold()}")
    else:
        print("❌ Failed to update config")
    
    # Clean up test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
        print("\n🧹 Test config file cleaned up")
    
    print("\n✅ Configuration service test completed")
