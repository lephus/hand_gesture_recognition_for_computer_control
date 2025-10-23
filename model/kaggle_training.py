#!/usr/bin/env python3
"""
Hand Gesture Recognition Model Training
Kaggle Notebook for Sprint 1 - HaGRID Dataset
Author: [Your Name]
Date: October 2025

Dataset Structure Expected:
/kaggle/input/hagrid-sample/other/default/1/hagrid-sample-30k-384p/
└── hagrid_30k/
    ├── train_val_one/          -> one_finger
    ├── train_val_two_up/       -> two_fingers  
    ├── train_val_three/        -> three_fingers
    ├── train_val_four/         -> four_fingers
    ├── train_val_palm/         -> five_fingers
    ├── train_val_fist/         -> neutral
    └── train_val_mute/         -> x_gesture
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d

class GestureModelTrainer:
    """CNN Model Trainer for Hand Gesture Recognition"""
    
    def __init__(self, data_path, model_path, img_size=224):
        """
        Initialize trainer
        
        Args:
            data_path: Path to dataset directory
            model_path: Path to save trained model
            img_size: Input image size (224x224)
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.num_classes = 7  # 7 gesture classes (we have data for 7 classes)
        self.batch_size = 8   # Reduced for better gradient updates
        self.epochs = 50      # Keep epochs for convergence
        
        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Data path: {self.data_path}")
        print(f"💾 Model path: {self.model_path}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        
    def check_dataset_structure(self):
        """Check HaGRID dataset structure"""
        print(f"\n🔍 Checking HaGRID dataset structure...")
        
        # Path to HaGRID dataset
        hagrid_path = self.data_path / "hagrid_30k"
        
        if not hagrid_path.exists():
            print(f"❌ HaGRID dataset not found at {hagrid_path}")
            print(f"💡 Expected path: {self.data_path}/hagrid_30k/")
            return False
        
        print(f"✅ HaGRID dataset found at {hagrid_path}")
        
        # Check for gesture folders
        expected_folders = [
            'train_val_one', 'train_val_two_up', 'train_val_three',
            'train_val_four', 'train_val_palm', 'train_val_fist', 'train_val_mute'
        ]
        
        found_folders = []
        for folder in expected_folders:
            folder_path = hagrid_path / folder
            if folder_path.exists():
                image_count = len(list(folder_path.glob('*.jpg'))) + len(list(folder_path.glob('*.png')))
                found_folders.append(folder)
                print(f"   ✅ {folder}: {image_count} images")
            else:
                print(f"   ❌ {folder}: not found")
        
        if len(found_folders) == 0:
            print(f"❌ No gesture folders found!")
            return False
        
        print(f"✅ Found {len(found_folders)} gesture folders")
        
        # Check data quality
        print(f"\n🔍 Checking data quality...")
        total_images = 0
        for folder in found_folders:
            folder_path = hagrid_path / folder
            image_count = len(list(folder_path.glob('*.jpg'))) + len(list(folder_path.glob('*.png')))
            total_images += image_count
            print(f"   {folder}: {image_count} images")
        
        print(f"📊 Total images available: {total_images}")
        
        if total_images < 1000:
            print(f"⚠️  Warning: Low number of images ({total_images}). Consider using more data.")
        
        return True
        
    def load_dataset(self):
        """Load and organize HaGRID dataset from Kaggle"""
        print("\n📊 Loading HaGRID dataset...")
        
        # Define gesture mapping from HaGRID to our classes
        self.hagrid_to_our_mapping = {
            'one': 'one_finger',
            'two_up': 'two_fingers', 
            'three': 'three_fingers',
            'four': 'four_fingers',
            'palm': 'five_fingers',
            'fist': 'neutral',
            'mute': 'x_gesture'
        }
        
        # Our target classes
        self.gesture_classes = list(self.hagrid_to_our_mapping.values())
        
        # Load images and labels
        images = []
        labels = []
        class_counts = {}
        
        # Path to HaGRID dataset
        hagrid_path = self.data_path / "hagrid_30k"
        
        if not hagrid_path.exists():
            print(f"❌ HaGRID dataset not found at {hagrid_path}")
            print(f"💡 Expected path: /kaggle/input/hagrid-sample/other/default/1/hagrid-sample-30k-384p/hagrid_30k")
            raise ValueError("HaGRID dataset path not found!")
        
        print(f"📁 Loading from HaGRID dataset: {hagrid_path}")
        
        # Process each HaGRID gesture class
        for hagrid_class, our_class in self.hagrid_to_our_mapping.items():
            class_path = hagrid_path / f"train_val_{hagrid_class}"
            
            if not class_path.exists():
                print(f"⚠️  Warning: {hagrid_class} not found at {class_path}")
                continue
            
            # Get all image files
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            class_counts[our_class] = len(image_files)
            
            print(f"   📸 {hagrid_class} -> {our_class}: {len(image_files)} images")
            
            # Load images (limit to avoid memory issues)
            max_images_per_class = 800  # Increased for better training
            images_to_load = image_files[:max_images_per_class]
            
            for img_path in images_to_load:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"❌ Could not load image: {img_path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(our_class)
                    
                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")
                    continue
        
        if len(images) == 0:
            raise ValueError("No images loaded! Check your dataset path and structure.")
        
        # Convert to numpy arrays
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.gesture_classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to integers
        self.label_indices = np.array([self.label_to_idx[label] for label in self.labels])
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"📈 Total images: {len(self.images)}")
        print(f"📈 Image shape: {self.images.shape}")
        print(f"📈 Number of classes: {len(self.gesture_classes)}")
        print(f"📈 Classes: {self.gesture_classes}")
        
        # Show class distribution
        print(f"\n📊 Class distribution:")
        for class_name in self.gesture_classes:
            count = class_counts.get(class_name, 0)
            print(f"   {class_name}: {count} images")
            
        return self.images, self.label_indices
    
    def create_data_generators(self, train_images, train_labels, val_images, val_labels):
        """Create data generators with augmentation"""
        print("\n🔄 Creating data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=10,  # Minimal rotation for hand gestures
            width_shift_range=0.1,  # Minimal shift
            height_shift_range=0.1,  # Minimal shift
            shear_range=0.05,  # Minimal shear
            zoom_range=0.1,  # Minimal zoom
            horizontal_flip=False,  # Disabled for hand gestures
            brightness_range=[0.95, 1.05],  # Minimal brightness change
            fill_mode='nearest'
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            train_images, train_labels,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            val_images, val_labels,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        print("✅ Data generators created!")
        return train_generator, val_generator
    
    def build_model(self):
        """Build CNN model using transfer learning"""
        print("\n🏗️  Building model architecture...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze most base model layers, unfreeze last few layers
        base_model.trainable = True
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),  # Reduced dropout
            layers.Dense(128, activation='relu'),  # Smaller network
            layers.Dropout(0.1),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with higher learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.002),  # Higher learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, model, train_generator, val_generator):
        """Train the model"""
        print("\n🚀 Starting model training...")
        
        # Define callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                filepath=self.model_path / 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # Increased patience for better convergence
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Less aggressive reduction
                patience=8,  # Increased patience
                min_lr=1e-5,
                verbose=1
            )
        ]
        
        # Train model with fine-tuning
        print("🔄 Starting initial training...")
        history = model.fit(
            train_generator,
            epochs=self.epochs // 2,  # First half with frozen layers
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Fine-tuning: unfreeze more layers and reduce learning rate
        print("🔄 Starting fine-tuning...")
        for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
            layer.trainable = True
            
        # Recompile with lower learning rate for fine-tuning
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training with fine-tuning
        fine_tune_history = model.fit(
            train_generator,
            epochs=self.epochs - (self.epochs // 2),  # Remaining epochs
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
        
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self, model, test_images, test_labels):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        # Make predictions
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"🎯 Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        report = classification_report(
            test_labels, predicted_classes,
            target_names=self.gesture_classes,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predicted_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.gesture_classes,
                   yticklabels=self.gesture_classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.model_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy, test_loss, cm
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("\n📈 Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_info(self, model, test_accuracy, test_loss, history):
        """Save model information and results"""
        print(f"\n💾 Saving model and results...")
        
        # Create comprehensive model info
        model_info = {
            'model_architecture': 'MobileNetV2 + Custom Head',
            'input_size': f"{self.img_size}x{self.img_size}",
            'num_classes': self.num_classes,
            'gesture_classes': self.gesture_classes,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'total_parameters': int(model.count_params()),
            'training_date': pd.Timestamp.now().isoformat(),
            'training_epochs': len(history.history['loss']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        # Save model info
        with open(self.model_path / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save model in multiple formats
        model.save(self.model_path / 'gesture_model.h5')
        model.save(self.model_path / 'gesture_model.keras')
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(self.model_path / 'training_history.csv', index=False)
        
        print(f"✅ Model saved to: {self.model_path}")
        print(f"📄 Model info saved to: {self.model_path / 'model_info.json'}")
        print(f"📊 Training history saved to: {self.model_path / 'training_history.csv'}")
        
        return model_info
    
    def print_training_results(self, model_info, test_accuracy, test_loss):
        """Print comprehensive training results"""
        print(f"\n" + "="*80)
        print(f"🎯 TRAINING RESULTS SUMMARY")
        print(f"="*80)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"   🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   📉 Test Loss: {test_loss:.4f}")
        print(f"   🏆 Best Validation Accuracy: {model_info['best_val_accuracy']:.4f} ({model_info['best_val_accuracy']*100:.2f}%)")
        
        print(f"\n📈 TRAINING METRICS:")
        print(f"   🔢 Total Parameters: {model_info['total_parameters']:,}")
        print(f"   📅 Training Date: {model_info['training_date']}")
        print(f"   🔄 Epochs Trained: {model_info['training_epochs']}")
        print(f"   📊 Final Train Accuracy: {model_info['final_train_accuracy']:.4f}")
        print(f"   📊 Final Val Accuracy: {model_info['final_val_accuracy']:.4f}")
        
        print(f"\n🎨 GESTURE CLASSES:")
        for i, gesture in enumerate(model_info['gesture_classes']):
            print(f"   {i+1}. {gesture}")
        
        print(f"\n💾 SAVED FILES:")
        print(f"   📁 Model: {self.model_path}/gesture_model.h5")
        print(f"   📁 Model (Keras): {self.model_path}/gesture_model.keras")
        print(f"   📄 Info: {self.model_path}/model_info.json")
        print(f"   📊 History: {self.model_path}/training_history.csv")
        print(f"   📈 Plots: {self.model_path}/training_history.png")
        print(f"   🔥 Confusion Matrix: {self.model_path}/confusion_matrix.png")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Download model files from Kaggle")
        print(f"   2. Test model with real-time inference")
        print(f"   3. Integrate with web application")
        print(f"   4. Deploy for production use")
        
        print(f"\n" + "="*80)
    
    def create_additional_visualizations(self, model, test_images, test_labels, history):
        """Create additional visualizations"""
        print(f"\n📊 Creating additional visualizations...")
        
        # 1. Class-wise accuracy
        self.plot_class_accuracy(model, test_images, test_labels)
        
        # 2. Learning curves with smoothing
        self.plot_smoothed_learning_curves(history)
        
        # 3. Model summary
        self.print_model_summary(model)
        
        print(f"✅ Additional visualizations created!")
    
    def plot_class_accuracy(self, model, test_images, test_labels):
        """Plot class-wise accuracy"""
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i, class_name in enumerate(self.gesture_classes):
            class_mask = test_labels == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == test_labels[class_mask])
                class_accuracies.append(class_accuracy)
            else:
                class_accuracies.append(0.0)
        
        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.gesture_classes, class_accuracies, color='skyblue', alpha=0.7)
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Gesture Classes', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.model_path / 'class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_smoothed_learning_curves(self, history):
        """Plot smoothed learning curves"""
        
        plt.figure(figsize=(15, 5))
        
        # Smooth the curves
        epochs = range(1, len(history.history['accuracy']) + 1)
        smooth_train_acc = gaussian_filter1d(history.history['accuracy'], sigma=1)
        smooth_val_acc = gaussian_filter1d(history.history['val_accuracy'], sigma=1)
        smooth_train_loss = gaussian_filter1d(history.history['loss'], sigma=1)
        smooth_val_loss = gaussian_filter1d(history.history['val_loss'], sigma=1)
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history.history['accuracy'], 'b-', alpha=0.3, label='Raw Train')
        plt.plot(epochs, history.history['val_accuracy'], 'r-', alpha=0.3, label='Raw Val')
        plt.plot(epochs, smooth_train_acc, 'b-', linewidth=2, label='Smooth Train')
        plt.plot(epochs, smooth_val_acc, 'r-', linewidth=2, label='Smooth Val')
        plt.title('Smoothed Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history.history['loss'], 'b-', alpha=0.3, label='Raw Train')
        plt.plot(epochs, history.history['val_loss'], 'r-', alpha=0.3, label='Raw Val')
        plt.plot(epochs, smooth_train_loss, 'b-', linewidth=2, label='Smooth Train')
        plt.plot(epochs, smooth_val_loss, 'r-', linewidth=2, label='Smooth Val')
        plt.title('Smoothed Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(epochs, history.history['lr'], 'g-', linewidth=2)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate Schedule')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.model_path / 'smoothed_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_model_summary(self, model):
        """Print detailed model summary"""
        print(f"\n📋 MODEL ARCHITECTURE SUMMARY:")
        print(f"="*60)
        
        # Model summary
        model.summary()
        
        # Layer information
        print(f"\n🔍 LAYER DETAILS:")
        for i, layer in enumerate(model.layers):
            print(f"   {i+1:2d}. {layer.name:20s} | {str(layer.output_shape):20s} | {layer.count_params():8,} params")
        
        print(f"\n📊 TOTAL PARAMETERS: {model.count_params():,}")
        print(f"💾 MODEL SIZE: {model.count_params() * 4 / (1024*1024):.2f} MB")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🚀 Starting Hand Gesture Recognition Training Pipeline")
        print("=" * 60)
        
        # Check dataset structure first
        if not self.check_dataset_structure():
            raise ValueError("Dataset structure check failed!")
        
        # Load dataset
        images, labels = self.load_dataset()
        
        # Split data
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        print(f"📊 Data split:")
        print(f"   Training: {len(train_images)} samples")
        print(f"   Validation: {len(val_images)} samples")
        print(f"   Test: {len(test_images)} samples")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(
            train_images, train_labels, val_images, val_labels
        )
        
        # Build model
        model = self.build_model()
        
        # Train model
        history = self.train_model(model, train_gen, val_gen)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate model
        test_accuracy, test_loss, cm = self.evaluate_model(model, test_images, test_labels)
        
        # Save model and info
        model_info = self.save_model_info(model, test_accuracy, test_loss, history)
        
        # Print comprehensive results
        self.print_training_results(model_info, test_accuracy, test_loss)
        
        # Create additional visualizations
        self.create_additional_visualizations(model, test_images, test_labels, history)
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
        
        return model, history, test_accuracy

def main():
    """Main function for Kaggle notebook"""
    # Set paths for HaGRID dataset on Kaggle
    data_path = "/kaggle/input/hagrid-sample/other/default/1/hagrid-sample-30k-384p"
    model_path = "/kaggle/working/models"
    
    print("🌐 Kaggle HaGRID Dataset Training")
    print("=" * 60)
    print(f"📁 Data path: {data_path}")
    print(f"💾 Model path: {model_path}")
    
    try:
        # Initialize trainer
        trainer = GestureModelTrainer(data_path, model_path)
        
        # Run training pipeline
        model, history, accuracy = trainer.run_training_pipeline()
        
        print("\n" + "=" * 60)
        print("🎯 SPRINT 1 COMPLETED!")
        print("📋 Next Steps for Sprint 2:")
        print("   1. Export model for web integration")
        print("   2. Create inference service")
        print("   3. Design web application architecture")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Check your dataset path and structure")
        print(f"💡 Expected dataset structure:")
        print(f"   {data_path}/hagrid_30k/")
        print(f"   ├── train_val_one/")
        print(f"   ├── train_val_two_up/")
        print(f"   ├── train_val_three/")
        print(f"   ├── train_val_four/")
        print(f"   ├── train_val_palm/")
        print(f"   ├── train_val_fist/")
        print(f"   └── train_val_mute/")

if __name__ == "__main__":
    main()
