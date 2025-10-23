#!/usr/bin/env python3
"""
OS Controller Service
Handles system-level commands for gesture actions
Author: [Your Name]
Date: October 2025
"""

import os
import time
import platform
import subprocess
from typing import Dict, Any, Optional
import pyautogui
import pynput
from pynput.keyboard import Key, Listener
from pynput.mouse import Button, Listener as MouseListener

class OSController:
    """Cross-platform OS control service"""
    
    def __init__(self):
        """Initialize OS controller"""
        self.system = platform.system().lower()
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second cooldown between actions
        
        # Disable pyautogui failsafe for gesture control
        pyautogui.FAILSAFE = False
        
        print(f"🖥️  OS Controller initialized for: {self.system}")
        print(f"⚙️  Action cooldown: {self.action_cooldown}s")
    
    def execute_action(self, gesture: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action based on gesture
        
        Args:
            gesture: Recognized gesture
            config: Configuration mapping gestures to actions
            
        Returns:
            Action execution result
        """
        try:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_action_time < self.action_cooldown:
                return {
                    'success': False,
                    'message': 'Action cooldown active',
                    'gesture': gesture
                }
            
            # Get action mapping from config
            action_mapping = config.get('gesture_actions', {})
            action = action_mapping.get(gesture, 'none')
            
            if action == 'none':
                return {
                    'success': True,
                    'message': f'No action configured for gesture: {gesture}',
                    'gesture': gesture,
                    'action': action
                }
            
            # Execute action based on type
            result = self._execute_gesture_action(gesture, action)
            
            # Update last action time
            self.last_action_time = current_time
            
            return {
                'success': result['success'],
                'message': result['message'],
                'gesture': gesture,
                'action': action,
                'timestamp': current_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error executing action: {str(e)}',
                'gesture': gesture,
                'error': str(e)
            }
    
    def _execute_gesture_action(self, gesture: str, action: str) -> Dict[str, Any]:
        """Execute specific gesture action"""
        
        # Finger count gestures (1-5) - Launch applications
        if gesture in ['one_finger', 'two_fingers', 'three_fingers', 'four_fingers', 'five_fingers']:
            return self._launch_application(action)
        
        # Rotation gestures - Volume control
        elif gesture == 'rotate_clockwise':
            return self._volume_up()
        elif gesture == 'rotate_counterclockwise':
            return self._volume_down()
        
        # X gesture - Close window
        elif gesture == 'x_gesture':
            return self._close_window()
        
        # Swipe gestures - Tab navigation
        elif gesture == 'swipe_left':
            return self._previous_tab()
        elif gesture == 'swipe_right':
            return self._next_tab()
        
        # Neutral gesture - No action
        elif gesture == 'neutral':
            return {
                'success': True,
                'message': 'Neutral gesture - no action'
            }
        
        else:
            return {
                'success': False,
                'message': f'Unknown gesture: {gesture}'
            }
    
    def _launch_application(self, app_name: str) -> Dict[str, Any]:
        """Launch application based on system"""
        try:
            if self.system == 'windows':
                # Windows applications
                app_commands = {
                    'chrome': 'start chrome',
                    'firefox': 'start firefox',
                    'notepad': 'start notepad',
                    'calculator': 'start calc',
                    'explorer': 'start explorer',
                    'cmd': 'start cmd',
                    'vscode': 'start code'
                }
            elif self.system == 'darwin':  # macOS
                app_commands = {
                    'chrome': 'open -a "Google Chrome"',
                    'firefox': 'open -a Firefox',
                    'safari': 'open -a Safari',
                    'calculator': 'open -a Calculator',
                    'finder': 'open -a Finder',
                    'terminal': 'open -a Terminal',
                    'vscode': 'open -a "Visual Studio Code"'
                }
            else:  # Linux
                app_commands = {
                    'chrome': 'google-chrome',
                    'firefox': 'firefox',
                    'gedit': 'gedit',
                    'calculator': 'gnome-calculator',
                    'terminal': 'gnome-terminal',
                    'vscode': 'code'
                }
            
            command = app_commands.get(app_name.lower(), app_name)
            
            if self.system == 'linux':
                # For Linux, run in background
                subprocess.Popen([command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # For Windows and macOS
                os.system(command)
            
            return {
                'success': True,
                'message': f'Launched application: {app_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to launch {app_name}: {str(e)}'
            }
    
    def _volume_up(self) -> Dict[str, Any]:
        """Increase system volume"""
        try:
            if self.system == 'windows':
                # Windows volume up
                pyautogui.press('volumeup')
            elif self.system == 'darwin':
                # macOS volume up
                pyautogui.press('volumeup')
            else:
                # Linux volume up
                subprocess.run(['amixer', 'set', 'Master', '5%+'], capture_output=True)
            
            return {
                'success': True,
                'message': 'Volume increased'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to increase volume: {str(e)}'
            }
    
    def _volume_down(self) -> Dict[str, Any]:
        """Decrease system volume"""
        try:
            if self.system == 'windows':
                # Windows volume down
                pyautogui.press('volumedown')
            elif self.system == 'darwin':
                # macOS volume down
                pyautogui.press('volumedown')
            else:
                # Linux volume down
                subprocess.run(['amixer', 'set', 'Master', '5%-'], capture_output=True)
            
            return {
                'success': True,
                'message': 'Volume decreased'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to decrease volume: {str(e)}'
            }
    
    def _close_window(self) -> Dict[str, Any]:
        """Close active window"""
        try:
            if self.system == 'windows':
                # Windows: Alt + F4
                pyautogui.hotkey('alt', 'f4')
            elif self.system == 'darwin':
                # macOS: Cmd + W
                pyautogui.hotkey('cmd', 'w')
            else:
                # Linux: Alt + F4
                pyautogui.hotkey('alt', 'f4')
            
            return {
                'success': True,
                'message': 'Window closed'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to close window: {str(e)}'
            }
    
    def _next_tab(self) -> Dict[str, Any]:
        """Switch to next tab"""
        try:
            if self.system == 'windows':
                # Windows: Ctrl + Tab
                pyautogui.hotkey('ctrl', 'tab')
            elif self.system == 'darwin':
                # macOS: Cmd + Option + Right
                pyautogui.hotkey('cmd', 'option', 'right')
            else:
                # Linux: Ctrl + Tab
                pyautogui.hotkey('ctrl', 'tab')
            
            return {
                'success': True,
                'message': 'Switched to next tab'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to switch to next tab: {str(e)}'
            }
    
    def _previous_tab(self) -> Dict[str, Any]:
        """Switch to previous tab"""
        try:
            if self.system == 'windows':
                # Windows: Ctrl + Shift + Tab
                pyautogui.hotkey('ctrl', 'shift', 'tab')
            elif self.system == 'darwin':
                # macOS: Cmd + Option + Left
                pyautogui.hotkey('cmd', 'option', 'left')
            else:
                # Linux: Ctrl + Shift + Tab
                pyautogui.hotkey('ctrl', 'shift', 'tab')
            
            return {
                'success': True,
                'message': 'Switched to previous tab'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to switch to previous tab: {str(e)}'
            }
    
    def test_actions(self) -> Dict[str, Any]:
        """Test all available actions"""
        print("🧪 Testing OS controller actions...")
        
        test_results = {}
        
        # Test volume controls
        print("   Testing volume controls...")
        test_results['volume_up'] = self._volume_up()
        time.sleep(0.5)
        test_results['volume_down'] = self._volume_down()
        
        # Test tab navigation
        print("   Testing tab navigation...")
        test_results['next_tab'] = self._next_tab()
        time.sleep(0.5)
        test_results['previous_tab'] = self._previous_tab()
        
        # Test window close (be careful!)
        print("   Testing window close...")
        # Note: This will close the current window, so we'll skip it in testing
        # test_results['close_window'] = self._close_window()
        
        print("✅ OS controller test completed")
        return test_results

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing OS Controller Service")
    print("=" * 50)
    
    # Initialize controller
    controller = OSController()
    
    # Test actions
    test_results = controller.test_actions()
    
    # Print results
    print("\n📊 Test Results:")
    for action, result in test_results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {action}: {result['message']}")
    
    print("\n⚠️  Note: Some actions may have been executed on your system!")
    print("   Make sure you're in a safe environment for testing.")
