#!/usr/bin/env python3
"""
Gesture Inference Service
Handles real-time gesture recognition using trained CNN model
Author: [Your Name]
Date: October 2025
"""

import os
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional
import tensorflow as tf
from pathlib import Path

class GestureInference:
    """Real-time gesture inference service"""
    
    def __init__(self, model_path: str):
        """
        Initialize gesture inference service
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.gesture_classes = [
            'one_finger', 'two_fingers', 'three_fingers',
            'four_fingers', 'five_fingers', 'neutral',
            'rotate_clockwise', 'rotate_counterclockwise',
            'x_gesture', 'swipe_left', 'swipe_right'
        ]
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"🔄 Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
            print("✅ Model loaded successfully!")
            
            # Print model summary
            print(f"📊 Model input shape: {self.model.input_shape}")
            print(f"📊 Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model inference
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame ready for model
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (224x224)
            resized = cv2.resize(rgb_frame, (224, 224))
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch_frame = np.expand_dims(normalized, axis=0)
            
            return batch_frame
            
        except Exception as e:
            print(f"❌ Error preprocessing frame: {e}")
            raise
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict gesture from frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with gesture prediction results
        """
        if self.model is None:
            return {
                'gesture': 'model_not_loaded',
                'confidence': 0.0,
                'inference_time': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Make prediction
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_gesture = self.gesture_classes[predicted_class_idx]
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Track performance
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            # Keep only last 100 inference times for rolling average
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return {
                'gesture': predicted_gesture,
                'confidence': confidence,
                'inference_time': inference_time,
                'all_predictions': {
                    class_name: float(pred) 
                    for class_name, pred in zip(self.gesture_classes, predictions[0])
                }
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {
                'gesture': 'error',
                'confidence': 0.0,
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {
                'total_inferences': 0,
                'average_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0
            }
        
        return {
            'total_inferences': self.total_inferences,
            'average_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'recent_avg_inference_time': np.mean(self.inference_times[-10:]) if len(self.inference_times) >= 10 else np.mean(self.inference_times)
        }
    
    def test_inference(self, test_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Test inference with a sample frame
        
        Args:
            test_frame: Optional test frame, creates random frame if None
            
        Returns:
            Test results
        """
        if test_frame is None:
            # Create a random test frame
            test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print("🧪 Testing gesture inference...")
        result = self.predict(test_frame)
        print(f"✅ Test result: {result['gesture']} (confidence: {result['confidence']:.3f})")
        
        return result

# Example usage and testing
if __name__ == "__main__":
    # Test the inference service
    model_path = "model/trained_models/gesture_model.h5"
    
    if os.path.exists(model_path):
        print("🚀 Testing Gesture Inference Service")
        print("=" * 50)
        
        # Initialize service
        inference_service = GestureInference(model_path)
        
        # Test with random frame
        test_result = inference_service.test_inference()
        
        # Print performance stats
        stats = inference_service.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Total inferences: {stats['total_inferences']}")
        print(f"   Average inference time: {stats['average_inference_time']:.2f}ms")
        
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Please train the model first using the Kaggle training script")
