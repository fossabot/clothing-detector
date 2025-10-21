"""
Postprocessing module for stabilizing clothing detections across frames.

This module provides temporal memory, confidence filtering, and context-aware
rules to create stable detection results from potentially noisy raw detections.
"""

import time
import logging
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations

from .config import config

# Setup logging
logger = logging.getLogger(__name__)


class DetectionCache:
    """Cache for storing detection history per source."""
    
    def __init__(self, maxlen: int = 5):
        """
        Initialize detection cache.
        
        Args:
            maxlen: Maximum number of frames to keep in history
        """
        self.maxlen = maxlen
        self.history: Dict[str, deque] = {}
        self.last_seen: Dict[str, Dict[str, float]] = {}  # source_id -> {class_name: timestamp}
    
    def add_detection(self, source_id: str, detections: List[Dict[str, Any]], timestamp: float):
        """
        Add detections to cache for a source.
        
        Args:
            source_id: Unique identifier for the source (e.g., camera ID)
            detections: List of detection dictionaries
            timestamp: Current timestamp
        """
        if source_id not in self.history:
            self.history[source_id] = deque(maxlen=self.maxlen)
            self.last_seen[source_id] = {}
        
        # Store detection frame
        frame_data = {
            "timestamp": timestamp,
            "detections": detections.copy()
        }
        self.history[source_id].append(frame_data)
        
        # Update last seen times for detected classes
        for detection in detections:
            class_name = detection["class_name"]
            self.last_seen[source_id][class_name] = timestamp
    
    def get_recent_detections(self, source_id: str, current_time: float, 
                            expiry_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """
        Get recent detections that haven't expired.
        
        Args:
            source_id: Source identifier
            current_time: Current timestamp
            expiry_seconds: How long to keep detections
            
        Returns:
            List of recent detection frames
        """
        if source_id not in self.history:
            return []
        
        recent_frames = []
        for frame in self.history[source_id]:
            if current_time - frame["timestamp"] <= expiry_seconds:
                recent_frames.append(frame)
        
        return recent_frames
    
    def cleanup_expired(self, source_id: str, current_time: float, expiry_seconds: float = 5.0):
        """
        Remove expired entries from cache.
        
        Args:
            source_id: Source identifier
            current_time: Current timestamp
            expiry_seconds: How long to keep detections
        """
        if source_id not in self.history:
            return
        
        # Clean up last_seen times
        expired_classes = []
        for class_name, last_time in self.last_seen[source_id].items():
            if current_time - last_time > expiry_seconds:
                expired_classes.append(class_name)
        
        for class_name in expired_classes:
            del self.last_seen[source_id][class_name]
    
    def get_last_seen_time(self, source_id: str, class_name: str) -> Optional[float]:
        """Get the last time a class was detected for a source."""
        if source_id in self.last_seen and class_name in self.last_seen[source_id]:
            return self.last_seen[source_id][class_name]
        return None


class PostProcessor:
    """Main postprocessing class for stabilizing detections."""
    
    def __init__(self, postprocessing_config=None):
        """
        Initialize postprocessor.
        
        Args:
            postprocessing_config: PostProcessingConfig instance, uses global config if None
        """
        self.config = postprocessing_config or config.postprocessing
        self.cache = DetectionCache(maxlen=self.config.history_size)
        
        # Load confidence thresholds and layer rules
        self.conf_thresholds = self.config.conf_thresholds
        self.valid_layers = self.config.valid_layers
        
        logger.info(f"PostProcessor initialized with config: enabled={self.config.enabled}")
    
    def postprocess_with_cache(self, source_id: str, raw_detections: List[Dict[str, Any]], 
                             timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for postprocessing detections.
        
        Args:
            source_id: Unique identifier for the source
            raw_detections: Raw detections from the model
            timestamp: Current timestamp (uses current time if None)
            
        Returns:
            Postprocessed detections
        """
        if not self.config.enabled:
            return raw_detections
        
        if timestamp is None:
            timestamp = time.time()
        
        # Add to cache
        self.cache.add_detection(source_id, raw_detections, timestamp)
        
        # Apply postprocessing pipeline
        filtered_detections = self._filter_by_confidence(raw_detections)
        aggregated_detections = self._aggregate_temporal(source_id, filtered_detections, timestamp)
        persistent_detections = self._apply_persistence(source_id, aggregated_detections, timestamp)
        final_detections = self._apply_cooccurrence_rules(persistent_detections)
        
        # Cleanup expired entries
        self.cache.cleanup_expired(source_id, timestamp, self.config.expiry_seconds)
        
        logger.debug(f"Postprocessed {len(raw_detections)} -> {len(final_detections)} detections for {source_id}")
        return final_detections
    
    def _filter_by_confidence(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections by class-specific confidence thresholds.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections
        """
        filtered = []
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Get threshold for this class, use default if not specified
            threshold = self.conf_thresholds.get(class_name, self.conf_thresholds.get("default", 0.6))
            
            if confidence >= threshold:
                filtered.append(detection)
            else:
                logger.debug(f"Filtered out {class_name} with confidence {confidence:.3f} < {threshold:.3f}")
        
        return filtered
    
    def _aggregate_temporal(self, source_id: str, current_detections: List[Dict[str, Any]], 
                          timestamp: float) -> List[Dict[str, Any]]:
        """
        Aggregate detections over recent frames.
        
        Args:
            source_id: Source identifier
            current_detections: Current frame detections
            timestamp: Current timestamp
            
        Returns:
            Aggregated detections
        """
        recent_frames = self.cache.get_recent_detections(
            source_id, timestamp, self.config.expiry_seconds
        )
        
        if len(recent_frames) < 2:
            return current_detections
        
        # Count class occurrences and sum confidences
        class_stats = {}
        for frame in recent_frames:
            for detection in frame["detections"]:
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        "count": 0,
                        "total_conf": 0.0,
                        "detection": detection  # Keep the most recent detection structure
                    }
                
                class_stats[class_name]["count"] += 1
                class_stats[class_name]["total_conf"] += confidence
        
        # Filter by presence ratio
        min_frames = max(1, int(len(recent_frames) * self.config.min_presence_ratio))
        aggregated = []
        
        for class_name, stats in class_stats.items():
            if stats["count"] >= min_frames:
                # Calculate average confidence
                avg_confidence = stats["total_conf"] / stats["count"]
                
                # Create aggregated detection
                detection = stats["detection"].copy()
                detection["confidence"] = avg_confidence
                detection["presence_ratio"] = stats["count"] / len(recent_frames)
                
                aggregated.append(detection)
                logger.debug(f"Aggregated {class_name}: {stats['count']}/{len(recent_frames)} frames, "
                           f"avg_conf={avg_confidence:.3f}")
        
        return aggregated
    
    def _apply_persistence(self, source_id: str, detections: List[Dict[str, Any]], 
                         timestamp: float) -> List[Dict[str, Any]]:
        """
        Apply persistence logic to keep detections alive during brief occlusions.
        
        Args:
            source_id: Source identifier
            detections: Current aggregated detections
            timestamp: Current timestamp
            
        Returns:
            Detections with persistence applied
        """
        persistent = []
        current_classes = {det["class_name"] for det in detections}
        
        # Add current detections
        persistent.extend(detections)
        
        # Check for classes that should be kept alive
        for class_name in self.cache.last_seen.get(source_id, {}):
            if class_name not in current_classes:
                last_seen_time = self.cache.get_last_seen_time(source_id, class_name)
                if last_seen_time and (timestamp - last_seen_time) <= self.config.expiry_seconds:
                    # Keep this class alive
                    # We need to reconstruct the detection from recent history
                    recent_frames = self.cache.get_recent_detections(
                        source_id, timestamp, self.config.expiry_seconds
                    )
                    
                    for frame in reversed(recent_frames):  # Start from most recent
                        for detection in frame["detections"]:
                            if detection["class_name"] == class_name:
                                # Create a "kept alive" detection with reduced confidence
                                kept_alive = detection.copy()
                                kept_alive["confidence"] *= 0.8  # Reduce confidence
                                kept_alive["kept_alive"] = True
                                persistent.append(kept_alive)
                                logger.debug(f"Keeping {class_name} alive (last seen {timestamp - last_seen_time:.1f}s ago)")
                                break
                        else:
                            continue
                        break
        
        return persistent
    
    def _apply_cooccurrence_rules(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply co-occurrence rules to validate layer compatibility.
        
        Args:
            detections: List of detections
            
        Returns:
            Detections with co-occurrence validation applied
        """
        if len(detections) < 2:
            return detections
        
        validated = []
        class_names = [det["class_name"] for det in detections]
        
        for detection in detections:
            class_name = detection["class_name"]
            is_valid = True
            
            # Check compatibility with other detected classes
            for other_class in class_names:
                if other_class != class_name:
                    # Check if other_class is in valid layers for this class
                    valid_layers = self.valid_layers.get(class_name, [])
                    if valid_layers and other_class not in valid_layers:
                        # This is a potential conflict, but we'll keep both and log it
                        logger.debug(f"Potential layer conflict: {class_name} with {other_class}")
                        # For now, we keep all detections but could implement more complex logic
            
            if is_valid:
                validated.append(detection)
        
        return validated
    
    def get_cache_stats(self, source_id: str) -> Dict[str, Any]:
        """
        Get cache statistics for a source.
        
        Args:
            source_id: Source identifier
            
        Returns:
            Dictionary with cache statistics
        """
        if source_id not in self.cache.history:
            return {"frames": 0, "classes": 0}
        
        frames = len(self.cache.history[source_id])
        classes = len(self.cache.last_seen.get(source_id, {}))
        
        return {
            "frames": frames,
            "classes": classes,
            "last_seen": self.cache.last_seen.get(source_id, {}).copy()
        }
    
    def clear_cache(self, source_id: Optional[str] = None):
        """
        Clear cache for a specific source or all sources.
        
        Args:
            source_id: Source to clear (None for all sources)
        """
        if source_id is None:
            self.cache.history.clear()
            self.cache.last_seen.clear()
            logger.info("Cleared all caches")
        else:
            if source_id in self.cache.history:
                del self.cache.history[source_id]
            if source_id in self.cache.last_seen:
                del self.cache.last_seen[source_id]
            logger.info(f"Cleared cache for source: {source_id}")


# Global postprocessor instance
_postprocessor = None

def get_postprocessor() -> PostProcessor:
    """Get the global postprocessor instance."""
    global _postprocessor
    if _postprocessor is None:
        _postprocessor = PostProcessor()
    return _postprocessor
