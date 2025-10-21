#!/usr/bin/env python3
"""
Simple Example Script

Demonstrates how to use the detector package with the new detect() function.
"""

import os
import sys
import json

# Add the parent directory to the path so we can import the detector package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import detect

def main():
    """Simple example of using the detector package."""
    
    print("🚀 Simple Detector Example")
    print("=" * 40)
    
    # Example 1: Detect objects in a specific image
    print("\n📸 Example 1: Detect objects in an image")
    print("-" * 40)
    
    # You can replace this with any image path
    image_path = "detector/default/default_img.jpg"
    
    try:
        result = detect(
            img_input=image_path,
            conf_thres=0.6,
            img_size=640,
            device="cpu"
        )
        
        # Result is already JSON, so print it directly
        print("📄 JSON Result:")
        print(json.dumps(result, indent=2))
            
    except FileNotFoundError:
        print(f"❌ Image not found: {image_path}")
        print("💡 Make sure the image file exists")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Detect objects in default image
    print("\n📸 Example 2: Detect objects in default image")
    print("-" * 40)
    
    try:
        result = detect(
            # No img_input parameter = use default image
            conf_thres=0.6,
            img_size=640,
            device="cpu"
        )
        
        # Result is already JSON, so print it directly
        print("📄 JSON Result:")
        print(json.dumps(result, indent=2))
            
    except FileNotFoundError:
        print("❌ Default image not found")
        print("💡 Make sure the default image exists in detector/default/")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Multi-frame detection with postprocessing
    print("\n📸 Example 3: Multi-frame detection with postprocessing")
    print("-" * 40)
    
    try:
        # Simulate multiple frames from the same camera
        camera_id = "camera_1"
        
        print(f"Simulating detection sequence for camera: {camera_id}")
        
        # Run detection multiple times to simulate temporal data
        results = []
        for frame_num in range(5):
            print(f"  Frame {frame_num + 1}...")
            result = detect(
                img_input=image_path,
                conf_thres=0.6,
                img_size=640,
                device="cpu",
                source_id=camera_id  # Use same source_id for temporal processing
            )
            results.append(result)
            
            # Small delay to simulate real-time processing
            import time
            time.sleep(0.1)
        
        print(f"\n📊 Results Summary:")
        print(f"  Total frames processed: {len(results)}")
        
        # Show detection counts per frame
        for i, result in enumerate(results):
            left_count = len(result["left"]["detections"])
            right_count = len(result["right"]["detections"])
            print(f"  Frame {i+1}: {left_count + right_count} detections (L:{left_count}, R:{right_count})")
        
        # Show final result
        final_result = results[-1]
        print(f"\n📄 Final Result (Frame {len(results)}):")
        print(json.dumps(final_result, indent=2))
        
        print(f"\n💡 Note: Enable postprocessing in config.yaml to see temporal stabilization effects")
        
    except Exception as e:
        print(f"❌ Error in multi-frame example: {e}")
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main()