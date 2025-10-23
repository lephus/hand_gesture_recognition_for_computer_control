#!/usr/bin/env python3
"""
Test script for hand gesture recognition model
Tests the trained model with webcam input or sample images
Author: [Your Name]
Date: October 2025
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gesture_inference import GestureInference

def test_model_with_webcam():
    """Test model with live webcam input"""
    print("🎥 Testing model with webcam...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Initialize model
    model_path = "models/best_model_v1.h5"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        inference_service = GestureInference(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("📹 Webcam initialized. Starting gesture recognition...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam")
                break
            
            frame_count += 1
            
            # Perform gesture recognition
            result = inference_service.predict(frame)
            
            # Calculate FPS
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                print(f"📊 FPS: {fps:.1f}")
                frame_count = 0
                start_time = current_time
            
            # Display results on frame
            gesture = result['gesture']
            confidence = result['confidence']
            inference_time = result.get('inference_time', 0)
            
            # Draw text on frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"test_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved as: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        stats = inference_service.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Total inferences: {stats['total_inferences']}")
        print(f"   Average inference time: {stats['average_inference_time']:.2f}ms")
        print(f"   Min inference time: {stats['min_inference_time']:.2f}ms")
        print(f"   Max inference time: {stats['max_inference_time']:.2f}ms")

def test_model_with_sample_images():
    """Test model with sample images"""
    print("🖼️  Testing model with sample images...")
    
    # Initialize model
    model_path = "models/best_model_v1.h5"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        inference_service = GestureInference(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample test images
    test_images = []
    
    # Create different colored rectangles as test images
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for i, color in enumerate(colors):
        # Create a test image
        test_img = np.full((224, 224, 3), color, dtype=np.uint8)
        test_images.append(test_img)
    
    print(f"🧪 Testing with {len(test_images)} sample images...")
    
    for i, test_img in enumerate(test_images):
        print(f"\n📸 Testing image {i+1}/{len(test_images)}")
        
        # Perform prediction
        result = inference_service.predict(test_img)
        
        print(f"   Gesture: {result['gesture']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Inference time: {result.get('inference_time', 0):.2f}ms")
        
        # Show all predictions
        if 'all_predictions' in result:
            print("   All predictions:")
            for gesture, conf in result['all_predictions'].items():
                if conf > 0.1:  # Only show predictions with confidence > 10%
                    print(f"     {gesture}: {conf:.3f}")
    
    # Print performance stats
    stats = inference_service.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Total inferences: {stats['total_inferences']}")
    print(f"   Average inference time: {stats['average_inference_time']:.2f}ms")

def test_model_basic():
    """Basic model test without camera"""
    print("🧪 Basic model test...")
    
    # Try different model paths
    model_paths = [
        "models/new_model.h5",
        "models/best_model_v1.h5",
        "models/fixed_model.h5"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"🔄 Trying model: {model_path}")
            try:
                inference_service = GestureInference(model_path)
                print("✅ Model loaded successfully!")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")
                continue
    
    if not model_loaded:
        print("❌ No working model found!")
        print("💡 Please run: python create_new_model.py")
        return
    
    try:
        # Test with random frame
        test_result = inference_service.test_inference()
        
        # Print performance stats
        stats = inference_service.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Total inferences: {stats['total_inferences']}")
        print(f"   Average inference time: {stats['average_inference_time']:.2f}ms")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("  Hand Gesture Recognition Model Test")
    print("  Computer Vision Project - DUT")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. Basic model test (no camera)")
    print("2. Test with sample images")
    print("3. Test with webcam (live)")
    print("4. All tests")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            test_model_basic()
        elif choice == "2":
            test_model_with_sample_images()
        elif choice == "3":
            test_model_with_webcam()
        elif choice == "4":
            print("\n🔄 Running all tests...")
            test_model_basic()
            print("\n" + "="*50)
            test_model_with_sample_images()
            print("\n" + "="*50)
            test_model_with_webcam()
        else:
            print("❌ Invalid choice. Running basic test...")
            test_model_basic()
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()
