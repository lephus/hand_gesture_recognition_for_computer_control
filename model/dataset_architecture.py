#!/usr/bin/env python3
"""
Dataset Architecture Analyzer
Phân tích kiến trúc và chất lượng dataset HaGRID
Author: LÊ HỮU PHÚ
Main Supervisors: HUỲNH HỮU HƯNG
CLASS: K50KHMT - THỊ GIÁC MÁY TÍNH
Date: October 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Phân tích kiến trúc và chất lượng dataset HaGRID"""
    
    def __init__(self, data_path):
        """
        Initialize dataset analyzer
        
        Args:
            data_path: Path to dataset directory
        """
        self.data_path = Path(data_path)
        self.hagrid_path = self.data_path / "hagrid_30k"
        
        # Analysis results
        self.dataset_info = {}
        self.class_distribution = {}
        self.image_quality_stats = {}
        self.annotations_info = {}
        
        print(f"📁 Dataset path: {self.data_path}")
        print(f"📁 HaGRID path: {self.hagrid_path}")
        
    def check_dataset_structure(self):
        """Kiểm tra cấu trúc dataset"""
        print("\n🔍 PHÂN TÍCH CẤU TRÚC DATASET")
        print("=" * 50)
        
        # Kiểm tra path chính
        if not self.data_path.exists():
            print(f"❌ Dataset path không tồn tại: {self.data_path}")
            return False
            
        print(f"✅ Dataset path tồn tại: {self.data_path}")
        
        # Kiểm tra HaGRID folder
        if not self.hagrid_path.exists():
            print(f"❌ HaGRID folder không tồn tại: {self.hagrid_path}")
            return False
            
        print(f"✅ HaGRID folder tồn tại: {self.hagrid_path}")
        
        # Kiểm tra các gesture folders
        expected_folders = [
            'train_val_call', 'train_val_dislike', 'train_val_fist', 'train_val_four',
            'train_val_like', 'train_val_mute', 'train_val_ok', 'train_val_one',
            'train_val_palm', 'train_val_peace', 'train_val_peace_inverted',
            'train_val_rock', 'train_val_stop', 'train_val_stop_inverted',
            'train_val_three', 'train_val_three2', 'train_val_two_up',
            'train_val_two_up_inverted'
        ]
        
        found_folders = []
        missing_folders = []
        
        for folder in expected_folders:
            folder_path = self.hagrid_path / folder
            if folder_path.exists():
                found_folders.append(folder)
                print(f"   ✅ {folder}")
            else:
                missing_folders.append(folder)
                print(f"   ❌ {folder}")
        
        print(f"\n📊 Tổng kết cấu trúc:")
        print(f"   ✅ Found: {len(found_folders)} folders")
        print(f"   ❌ Missing: {len(missing_folders)} folders")
        
        if missing_folders:
            print(f"   ⚠️  Missing folders: {missing_folders}")
        
        self.dataset_info['total_folders'] = len(found_folders)
        self.dataset_info['missing_folders'] = missing_folders
        self.dataset_info['found_folders'] = found_folders
        
        return len(found_folders) > 0
    
    def analyze_class_distribution(self):
        """Phân tích phân phối các lớp"""
        print("\n📊 PHÂN TÍCH PHÂN PHỐI CÁC LỚP")
        print("=" * 50)
        
        if not self.hagrid_path.exists():
            print("❌ HaGRID path không tồn tại!")
            return
        
        # Lấy danh sách các gesture classes
        gesture_classes = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
            'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
            'three', 'three_2', 'two_up', 'two_up_inverted'
        ]
        
        class_counts = {}
        total_images = 0
        
        print("📸 Số lượng ảnh theo từng lớp:")
        for gesture_class in gesture_classes:
            # Xác định path cho từng class
            if gesture_class == 'like':
                class_path = self.hagrid_path / f"train_{gesture_class}"
            else:
                class_path = self.hagrid_path / f"train_val_{gesture_class}"
            
            if class_path.exists():
                # Đếm số ảnh
                image_count = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
                class_counts[gesture_class] = image_count
                total_images += image_count
                print(f"   {gesture_class:20s}: {image_count:6d} images")
            else:
                class_counts[gesture_class] = 0
                print(f"   {gesture_class:20s}: {0:6d} images (NOT FOUND)")
        
        print(f"\n📊 Tổng kết phân phối:")
        print(f"   Total images: {total_images:,}")
        print(f"   Number of classes: {len(gesture_classes)}")
        print(f"   Average per class: {total_images/len(gesture_classes):.1f}")
        
        # Tìm class có ít ảnh nhất và nhiều ảnh nhất
        min_class = min(class_counts.items(), key=lambda x: x[1])
        max_class = max(class_counts.items(), key=lambda x: x[1])
        
        print(f"   Min images: {min_class[0]} ({min_class[1]} images)")
        print(f"   Max images: {max_class[0]} ({max_class[1]} images)")
        
        # Tính độ lệch chuẩn
        counts = list(class_counts.values())
        std_dev = np.std(counts)
        print(f"   Standard deviation: {std_dev:.1f}")
        
        self.class_distribution = class_counts
        self.dataset_info['total_images'] = total_images
        self.dataset_info['num_classes'] = len(gesture_classes)
        self.dataset_info['avg_per_class'] = total_images/len(gesture_classes)
        self.dataset_info['std_deviation'] = std_dev
        
        return class_counts
    
    def analyze_image_quality(self, sample_size=100):
        """Phân tích chất lượng ảnh"""
        print(f"\n🖼️  PHÂN TÍCH CHẤT LƯỢNG ẢNH (Sample: {sample_size})")
        print("=" * 50)
        
        if not self.hagrid_path.exists():
            print("❌ HaGRID path không tồn tại!")
            return
        
        # Lấy sample ảnh từ mỗi class
        gesture_classes = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
            'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
            'three', 'three_2', 'two_up', 'two_up_inverted'
        ]
        
        image_sizes = []
        image_formats = []
        corrupted_images = 0
        total_analyzed = 0
        
        for gesture_class in gesture_classes:
            if gesture_class == 'like':
                class_path = self.hagrid_path / f"train_{gesture_class}"
            else:
                class_path = self.hagrid_path / f"train_val_{gesture_class}"
            
            if not class_path.exists():
                continue
                
            # Lấy sample ảnh
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            sample_files = image_files[:sample_size]
            
            print(f"   📸 Analyzing {gesture_class}...")
            
            for img_path in sample_files:
                try:
                    # Load ảnh
                    img = cv2.imread(str(img_path))
                    if img is None:
                        corrupted_images += 1
                        continue
                    
                    # Lấy thông tin ảnh
                    height, width = img.shape[:2]
                    image_sizes.append((width, height))
                    
                    # Lấy format
                    format_type = img_path.suffix.lower()
                    image_formats.append(format_type)
                    
                    total_analyzed += 1
                    
                except Exception as e:
                    corrupted_images += 1
                    print(f"      ❌ Error loading {img_path}: {e}")
        
        # Phân tích kết quả
        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            
            print(f"\n📊 Kết quả phân tích chất lượng:")
            print(f"   Total analyzed: {total_analyzed}")
            print(f"   Corrupted images: {corrupted_images}")
            print(f"   Corruption rate: {corrupted_images/total_analyzed*100:.2f}%")
            
            print(f"\n📐 Kích thước ảnh:")
            print(f"   Width - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.1f}")
            print(f"   Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.1f}")
            
            # Phân tích format
            format_counts = Counter(image_formats)
            print(f"\n📄 Format phân phối:")
            for format_type, count in format_counts.items():
                print(f"   {format_type}: {count} images ({count/total_analyzed*100:.1f}%)")
            
            self.image_quality_stats = {
                'total_analyzed': total_analyzed,
                'corrupted_images': corrupted_images,
                'corruption_rate': corrupted_images/total_analyzed*100,
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'format_distribution': dict(format_counts)
            }
        
        return self.image_quality_stats
    
    def analyze_annotations(self):
        """Phân tích annotations (nếu có)"""
        print("\n📋 PHÂN TÍCH ANNOTATIONS")
        print("=" * 50)
        
        # Tìm file annotations
        annotation_files = []
        
        # Tìm file JSON
        json_files = list(self.data_path.glob("*.json"))
        if json_files:
            annotation_files.extend(json_files)
            print(f"📄 Found JSON files: {len(json_files)}")
            for f in json_files:
                print(f"   - {f.name}")
        
        # Tìm file parquet
        parquet_files = list(self.data_path.glob("*.parquet"))
        if parquet_files:
            annotation_files.extend(parquet_files)
            print(f"📊 Found Parquet files: {len(parquet_files)}")
            for f in parquet_files:
                print(f"   - {f.name}")
        
        if not annotation_files:
            print("⚠️  No annotation files found!")
            return
        
        # Phân tích file parquet nếu có
        parquet_file = None
        for f in parquet_files:
            if 'annotations' in f.name.lower():
                parquet_file = f
                break
        
        if parquet_file:
            try:
                print(f"\n📊 Analyzing {parquet_file.name}...")
                df = pd.read_parquet(parquet_file)
                
                print(f"   Rows: {len(df):,}")
                print(f"   Columns: {list(df.columns)}")
                
                # Phân tích cột chính
                if 'class' in df.columns:
                    class_counts = df['class'].value_counts()
                    print(f"\n📊 Class distribution in annotations:")
                    for class_name, count in class_counts.items():
                        print(f"   {class_name}: {count}")
                
                self.annotations_info = {
                    'file_path': str(parquet_file),
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'class_distribution': dict(class_counts) if 'class' in df.columns else {}
                }
                
            except Exception as e:
                print(f"❌ Error reading parquet file: {e}")
        
        return self.annotations_info
    
    def create_visualizations(self):
        """Tạo visualizations cho phân tích"""
        print("\n📊 TẠO VISUALIZATIONS")
        print("=" * 50)
        
        # 1. Class distribution bar chart
        if self.class_distribution:
            plt.figure(figsize=(15, 8))
            classes = list(self.class_distribution.keys())
            counts = list(self.class_distribution.values())
            
            bars = plt.bar(classes, counts, color='skyblue', alpha=0.7)
            plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
            plt.xlabel('Gesture Classes', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Thêm giá trị trên bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Image quality pie chart
        if self.image_quality_stats and 'format_distribution' in self.image_quality_stats:
            plt.figure(figsize=(10, 6))
            formats = list(self.image_quality_stats['format_distribution'].keys())
            counts = list(self.image_quality_stats['format_distribution'].values())
            
            plt.pie(counts, labels=formats, autopct='%1.1f%%', startangle=90)
            plt.title('Image Format Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('format_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n📋 BÁO CÁO TỔNG HỢP")
        print("=" * 60)
        
        print(f"📁 Dataset Path: {self.data_path}")
        print(f"📁 HaGRID Path: {self.hagrid_path}")
        
        if self.dataset_info:
            print(f"\n📊 Dataset Information:")
            print(f"   Total folders: {self.dataset_info.get('total_folders', 'N/A')}")
            print(f"   Total images: {self.dataset_info.get('total_images', 'N/A'):,}")
            print(f"   Number of classes: {self.dataset_info.get('num_classes', 'N/A')}")
            print(f"   Average per class: {self.dataset_info.get('avg_per_class', 'N/A'):.1f}")
            print(f"   Standard deviation: {self.dataset_info.get('std_deviation', 'N/A'):.1f}")
        
        if self.class_distribution:
            print(f"\n📸 Class Distribution:")
            for class_name, count in self.class_distribution.items():
                print(f"   {class_name:20s}: {count:6d} images")
        
        if self.image_quality_stats:
            print(f"\n🖼️  Image Quality:")
            print(f"   Total analyzed: {self.image_quality_stats.get('total_analyzed', 'N/A')}")
            print(f"   Corrupted images: {self.image_quality_stats.get('corrupted_images', 'N/A')}")
            print(f"   Corruption rate: {self.image_quality_stats.get('corruption_rate', 'N/A'):.2f}%")
            print(f"   Average size: {self.image_quality_stats.get('avg_width', 'N/A'):.0f}x{self.image_quality_stats.get('avg_height', 'N/A'):.0f}")
        
        if self.annotations_info:
            print(f"\n📋 Annotations:")
            print(f"   File: {self.annotations_info.get('file_path', 'N/A')}")
            print(f"   Total rows: {self.annotations_info.get('total_rows', 'N/A'):,}")
            print(f"   Columns: {self.annotations_info.get('columns', 'N/A')}")
        
        print(f"\n✅ Phân tích hoàn tất!")
    
    def run_full_analysis(self):
        """Chạy phân tích đầy đủ"""
        print("🚀 BẮT ĐẦU PHÂN TÍCH DATASET")
        print("=" * 60)
        
        # 1. Kiểm tra cấu trúc
        if not self.check_dataset_structure():
            print("❌ Dataset structure check failed!")
            return False
        
        # 2. Phân tích phân phối lớp
        self.analyze_class_distribution()
        
        # 3. Phân tích chất lượng ảnh
        self.analyze_image_quality()
        
        # 4. Phân tích annotations
        self.analyze_annotations()
        
        # 5. Tạo visualizations
        self.create_visualizations()
        
        # 6. Tạo báo cáo
        self.generate_report()
        
        print("\n🎉 PHÂN TÍCH HOÀN TẤT!")
        return True

def main():
    """Main function"""
    # Set path for Kaggle dataset
    data_path = "/kaggle/input/hagrid-sample/other/default/1/hagrid-sample-30k-384p"
    
    print("🔍 DATASET ARCHITECTURE ANALYZER")
    print("=" * 60)
    print(f"📁 Data path: {data_path}")
    
    try:
        # Initialize analyzer
        analyzer = DatasetAnalyzer(data_path)
        
        # Run full analysis
        success = analyzer.run_full_analysis()
        
        if success:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()