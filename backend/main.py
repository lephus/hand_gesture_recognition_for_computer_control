#!/usr/bin/env python3
"""
Hand Gesture Recognition Backend
FastAPI server with WebSocket support for real-time gesture control
Author: [Your Name]
Date: October 2025
"""

import os
import sys
import json
import time
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# FastAPI and WebSocket imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import socketio

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom services
from services.gesture_inference import GestureInference
from services.os_controller import OSController
from services.config_service import ConfigService

# Initialize FastAPI app
app = FastAPI(
    title="Hand Gesture Recognition API",
    description="Real-time hand gesture recognition for computer control",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
socket_app = socketio.ASGIApp(sio, app)

# Global services
gesture_inference: Optional[GestureInference] = None
os_controller: Optional[OSController] = None
config_service: Optional[ConfigService] = None

# Performance tracking
frame_count = 0
start_time = time.time()
fps_counter = 0
last_fps_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global gesture_inference, os_controller, config_service
    
    print("🚀 Starting Hand Gesture Recognition Backend...")
    
    try:
        # Initialize configuration service
        config_service = ConfigService()
        print("✅ Configuration service initialized")
        
        # Initialize OS controller
        os_controller = OSController()
        print("✅ OS controller initialized")
        
        # Initialize gesture inference (will be loaded when model is available)
        model_path = Path("models/best_model_v1.h5")
        if model_path.exists():
            gesture_inference = GestureInference(str(model_path))
            print("✅ Gesture inference service initialized")
        else:
            print("⚠️  Model not found - inference will be disabled")
            print(f"   Expected model at: {model_path}")
        
        print("🎉 Backend startup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hand Gesture Recognition API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "gesture_inference": gesture_inference is not None,
            "os_controller": os_controller is not None,
            "config_service": config_service is not None
        }
    }

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    if config_service is None:
        raise HTTPException(status_code=500, detail="Configuration service not initialized")
    
    return config_service.get_config()

@app.post("/api/config")
async def update_config(config: Dict[str, Any]):
    """Update configuration"""
    if config_service is None:
        raise HTTPException(status_code=500, detail="Configuration service not initialized")
    
    try:
        config_service.update_config(config)
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    global frame_count, fps_counter, last_fps_time
    
    current_time = time.time()
    uptime = current_time - start_time
    
    # Calculate current FPS
    if current_time - last_fps_time >= 1.0:
        fps_counter = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time
    
    return {
        "uptime": uptime,
        "frames_processed": frame_count,
        "current_fps": fps_counter,
        "services": {
            "gesture_inference": gesture_inference is not None,
            "os_controller": os_controller is not None,
            "config_service": config_service is not None
        }
    }

# WebSocket event handlers
@sio.on('connect')
async def on_connect(sid, environ):
    """Handle client connection"""
    print(f"🔗 Client connected: {sid}")
    await sio.emit('connected', {'message': 'Connected to gesture recognition server'}, room=sid)

@sio.on('disconnect')
async def on_disconnect(sid):
    """Handle client disconnection"""
    print(f"🔌 Client disconnected: {sid}")

@sio.on('frame')
async def handle_frame(sid, data):
    """Handle incoming video frame"""
    global frame_count, fps_counter, last_fps_time
    
    try:
        frame_count += 1
        
        # Check if inference service is available
        if gesture_inference is None:
            await sio.emit('error', {'message': 'Gesture inference not available'}, room=sid)
            return
        
        # Decode base64 image
        if 'frame' not in data:
            await sio.emit('error', {'message': 'No frame data provided'}, room=sid)
            return
        
        # Decode base64 to image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            await sio.emit('error', {'message': 'Invalid image data'}, room=sid)
            return
        
        # Perform gesture recognition
        start_inference = time.time()
        gesture_result = gesture_inference.predict(frame)
        inference_time = (time.time() - start_inference) * 1000  # Convert to ms
        
        # Execute action if gesture is recognized
        if gesture_result['gesture'] != 'neutral' and gesture_result['confidence'] > 0.8:
            if os_controller is not None:
                action_result = os_controller.execute_action(
                    gesture_result['gesture'], 
                    config_service.get_config() if config_service else {}
                )
                gesture_result['action_executed'] = action_result
        
        # Send result back to client
        await sio.emit('gesture_result', {
            'gesture': gesture_result['gesture'],
            'confidence': gesture_result['confidence'],
            'inference_time': inference_time,
            'timestamp': time.time()
        }, room=sid)
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        await sio.emit('error', {'message': f'Frame processing error: {str(e)}'}, room=sid)

@sio.on('test_gesture')
async def test_gesture(sid, data):
    """Test gesture recognition with sample data"""
    if gesture_inference is None:
        await sio.emit('error', {'message': 'Gesture inference not available'}, room=sid)
        return
    
    try:
        # Create a test frame (you can modify this for testing)
        test_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = gesture_inference.predict(test_frame)
        
        await sio.emit('test_result', {
            'gesture': result['gesture'],
            'confidence': result['confidence'],
            'message': 'Test gesture recognition completed'
        }, room=sid)
        
    except Exception as e:
        await sio.emit('error', {'message': f'Test error: {str(e)}'}, room=sid)

def main():
    """Main function to run the server"""
    print("=" * 60)
    print("  Hand Gesture Recognition Backend Server")
    print("  Computer Vision Project - DUT")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
