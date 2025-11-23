"""
Predictive Analytics Engine for JARVIS
Handles user behavior modeling, routine detection, and anomaly identification
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)


class UserBehaviorModel:
    """Models user behavior patterns and preferences"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.interactions: List[Dict[str, Any]] = []
        self.preferences: Dict[str, float] = defaultdict(float)
        self.activity_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.context_scores: Dict[str, float] = defaultdict(float)

    def record_interaction(
        self,
        action: str,
        context: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        success: bool = True,
        feedback_score: float = 0.0
    ):
        """Record a user interaction for behavior modeling"""
        timestamp = timestamp or datetime.now()

        interaction = {
            "action": action,
            "context": context,
            "timestamp": timestamp,
            "success": success,
            "feedback_score": feedback_score,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "location": context.get("location", "unknown")
        }

        self.interactions.append(interaction)
        self.activity_patterns[action].append(timestamp)

        # Update preferences based on success and feedback
        if success:
            self.preferences[action] += 0.1 + feedback_score
            for key, value in context.items():
                self.context_scores[f"{action}:{key}:{value}"] += 0.1

        logger.info(f"Recorded interaction: {action} at {timestamp}")

    def get_action_frequency(self, action: str) -> int:
        """Get frequency of a specific action"""
        return len(self.activity_patterns.get(action, []))

    def get_preference_score(self, action: str) -> float:
        """Get preference score for an action"""
        return self.preferences.get(action, 0.0)

    def get_active_hours(self) -> List[int]:
        """Get hours when user is most active"""
        if not self.interactions:
            return list(range(9, 18))  # Default 9am-6pm

        hour_counts = Counter(i["hour"] for i in self.interactions)
        avg_count = sum(hour_counts.values()) / len(hour_counts)

        return sorted([hour for hour, count in hour_counts.items() if count > avg_count])

    def get_context_affinity(self, context: Dict[str, Any]) -> float:
        """Calculate affinity score for a given context"""
        score = 0.0
        for key, value in context.items():
            context_key = f"{key}:{value}"
            for pref_key, pref_score in self.context_scores.items():
                if context_key in pref_key:
                    score += pref_score
        return score


class RoutineDetector:
    """Detects recurring patterns and routines in user behavior"""

    def __init__(self, min_occurrences: int = 3, time_window_days: int = 30):
        self.min_occurrences = min_occurrences
        self.time_window_days = time_window_days
        self.detected_routines: List[Dict[str, Any]] = []

    def detect_routines(
        self,
        behavior_model: UserBehaviorModel
    ) -> List[Dict[str, Any]]:
        """Detect recurring routines from user behavior"""
        routines = []

        for action, timestamps in behavior_model.activity_patterns.items():
            if len(timestamps) < self.min_occurrences:
                continue

            # Check for time-based patterns
            time_patterns = self._detect_time_patterns(timestamps)
            if time_patterns:
                routines.append({
                    "action": action,
                    "type": "time_based",
                    "pattern": time_patterns,
                    "confidence": self._calculate_confidence(timestamps, time_patterns),
                    "occurrences": len(timestamps)
                })

            # Check for sequential patterns
            sequence_patterns = self._detect_sequence_patterns(
                action, behavior_model.interactions
            )
            if sequence_patterns:
                routines.append({
                    "action": action,
                    "type": "sequence_based",
                    "pattern": sequence_patterns,
                    "confidence": 0.7,
                    "occurrences": len(timestamps)
                })

        self.detected_routines = routines
        logger.info(f"Detected {len(routines)} routines")
        return routines

    def _detect_time_patterns(
        self,
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Detect time-based patterns (daily, weekly, etc.)"""
        if len(timestamps) < 2:
            return None

        # Group by hour of day
        hour_counts = Counter(ts.hour for ts in timestamps)
        most_common_hour = hour_counts.most_common(1)[0]

        if most_common_hour[1] >= self.min_occurrences:
            # Check day of week pattern
            day_counts = Counter(ts.weekday() for ts in timestamps)
            common_days = [day for day, count in day_counts.items()
                          if count >= self.min_occurrences]

            return {
                "time_of_day": most_common_hour[0],
                "days_of_week": common_days if common_days else list(range(7)),
                "frequency": "daily" if len(common_days) >= 5 else "weekly"
            }

        return None

    def _detect_sequence_patterns(
        self,
        action: str,
        interactions: List[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """Detect action sequences (e.g., always do X after Y)"""
        sequences = []

        for i, interaction in enumerate(interactions):
            if interaction["action"] == action and i > 0:
                prev_action = interactions[i-1]["action"]
                time_diff = (interaction["timestamp"] -
                           interactions[i-1]["timestamp"]).total_seconds() / 60

                # If actions occur within 30 minutes, consider as sequence
                if time_diff <= 30:
                    sequences.append(prev_action)

        if sequences:
            sequence_counts = Counter(sequences)
            most_common = sequence_counts.most_common(1)[0]

            if most_common[1] >= self.min_occurrences:
                return [most_common[0], action]

        return None

    def _calculate_confidence(
        self,
        timestamps: List[datetime],
        pattern: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a detected pattern"""
        matching = 0
        total = len(timestamps)

        for ts in timestamps:
            if ts.hour == pattern["time_of_day"]:
                if ts.weekday() in pattern["days_of_week"]:
                    matching += 1

        return matching / total if total > 0 else 0.0


class AnomalyDetector:
    """Detects unusual or anomalous behavior patterns"""

    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self.baseline_established = False
        self.normal_patterns: Dict[str, Any] = {}

    def establish_baseline(self, behavior_model: UserBehaviorModel):
        """Establish baseline normal behavior"""
        if len(behavior_model.interactions) < 10:
            logger.warning("Insufficient data for baseline")
            return

        # Calculate normal activity patterns
        hourly_activity = defaultdict(int)
        action_frequencies = defaultdict(int)

        for interaction in behavior_model.interactions:
            hourly_activity[interaction["hour"]] += 1
            action_frequencies[interaction["action"]] += 1

        self.normal_patterns = {
            "hourly_activity": dict(hourly_activity),
            "action_frequencies": dict(action_frequencies),
            "avg_actions_per_day": len(behavior_model.interactions) / 30,
            "common_actions": set(action_frequencies.keys())
        }

        self.baseline_established = True
        logger.info("Baseline behavior established")

    def detect_anomalies(
        self,
        recent_interactions: List[Dict[str, Any]],
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Detect anomalous behavior in recent interactions"""
        if not self.baseline_established:
            return []

        anomalies = []
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent = [i for i in recent_interactions
                 if i["timestamp"] > cutoff_time]

        # Check for unusual activity times
        for interaction in recent:
            hour = interaction["hour"]
            normal_count = self.normal_patterns["hourly_activity"].get(hour, 0)

            if normal_count == 0:
                anomalies.append({
                    "type": "unusual_time",
                    "severity": "medium",
                    "description": f"Activity at unusual hour: {hour}",
                    "interaction": interaction,
                    "timestamp": interaction["timestamp"]
                })

        # Check for unusual action frequency
        recent_action_counts = Counter(i["action"] for i in recent)
        for action, count in recent_action_counts.items():
            normal_freq = self.normal_patterns["action_frequencies"].get(action, 0) / 30
            expected_in_window = normal_freq * (time_window_hours / 24)

            if count > expected_in_window * 3:  # 3x normal frequency
                anomalies.append({
                    "type": "unusual_frequency",
                    "severity": "high",
                    "description": f"Unusual frequency of {action}: {count} (expected ~{expected_in_window:.1f})",
                    "action": action,
                    "timestamp": datetime.now()
                })

        # Check for unknown actions
        for interaction in recent:
            if interaction["action"] not in self.normal_patterns["common_actions"]:
                anomalies.append({
                    "type": "new_action",
                    "severity": "low",
                    "description": f"New action type: {interaction['action']}",
                    "interaction": interaction,
                    "timestamp": interaction["timestamp"]
                })

        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies


class PredictiveAnalytics:
    """Main predictive analytics engine combining all components"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.behavior_model = UserBehaviorModel(user_id)
        self.routine_detector = RoutineDetector()
        self.anomaly_detector = AnomalyDetector()
        self.predictions: List[Dict[str, Any]] = []

    def record_interaction(
        self,
        action: str,
        context: Dict[str, Any],
        success: bool = True,
        feedback_score: float = 0.0
    ):
        """Record user interaction for analytics"""
        self.behavior_model.record_interaction(
            action=action,
            context=context,
            success=success,
            feedback_score=feedback_score
        )

        # Update anomaly baseline periodically
        if len(self.behavior_model.interactions) % 50 == 0:
            self.anomaly_detector.establish_baseline(self.behavior_model)

    def analyze_patterns(self) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis"""
        # Detect routines
        routines = self.routine_detector.detect_routines(self.behavior_model)

        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            self.behavior_model.interactions
        )

        # Get active hours
        active_hours = self.behavior_model.get_active_hours()

        return {
            "routines": routines,
            "anomalies": anomalies,
            "active_hours": active_hours,
            "total_interactions": len(self.behavior_model.interactions),
            "analyzed_at": datetime.now().isoformat()
        }

    def predict_next_action(
        self,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict likely next actions based on context and patterns"""
        predictions = []
        current_time = datetime.now()

        # Get routines that match current time
        for routine in self.routine_detector.detected_routines:
            if routine["type"] == "time_based":
                pattern = routine["pattern"]
                if (current_time.hour == pattern["time_of_day"] and
                    current_time.weekday() in pattern["days_of_week"]):
                    predictions.append({
                        "action": routine["action"],
                        "confidence": routine["confidence"],
                        "reason": "time_based_routine",
                        "details": pattern
                    })

        # Get actions with high context affinity
        for action in self.behavior_model.preferences.keys():
            affinity = self.behavior_model.get_context_affinity(current_context)
            if affinity > 0.5:
                predictions.append({
                    "action": action,
                    "confidence": min(affinity / 10, 1.0),
                    "reason": "context_affinity",
                    "affinity_score": affinity
                })

        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        self.predictions = predictions[:5]  # Top 5 predictions
        return self.predictions

    def get_insights(self) -> Dict[str, Any]:
        """Get actionable insights from analytics"""
        analysis = self.analyze_patterns()

        insights = {
            "productivity_peak_hours": self.behavior_model.get_active_hours(),
            "established_routines": len(analysis["routines"]),
            "recent_anomalies": len(analysis["anomalies"]),
            "top_actions": sorted(
                self.behavior_model.preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "recommendations": self._generate_recommendations(analysis)
        }

        return insights

    def _generate_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Routine-based recommendations
        for routine in analysis["routines"]:
            if routine["confidence"] > 0.7:
                recommendations.append(
                    f"Consider automating {routine['action']} - "
                    f"occurs regularly with {routine['confidence']:.0%} consistency"
                )

        # Anomaly-based recommendations
        high_severity_anomalies = [
            a for a in analysis["anomalies"] if a["severity"] == "high"
        ]
        if high_severity_anomalies:
            recommendations.append(
                f"Detected {len(high_severity_anomalies)} unusual activity patterns - "
                "review for potential issues"
            )

        # Activity optimization
        active_hours = analysis["active_hours"]
        if len(active_hours) > 0:
            recommendations.append(
                f"Peak productivity hours: {min(active_hours)}:00-{max(active_hours)}:00 - "
                "schedule important tasks during this time"
            )

        return recommendations

    def export_model(self) -> Dict[str, Any]:
        """Export the predictive model for persistence"""
        return {
            "user_id": self.user_id,
            "preferences": dict(self.behavior_model.preferences),
            "context_scores": dict(self.behavior_model.context_scores),
            "routines": self.routine_detector.detected_routines,
            "normal_patterns": self.anomaly_detector.normal_patterns,
            "interaction_count": len(self.behavior_model.interactions),
            "exported_at": datetime.now().isoformat()
        }

    def import_model(self, model_data: Dict[str, Any]):
        """Import a previously exported model"""
        self.behavior_model.preferences = defaultdict(
            float, model_data.get("preferences", {})
        )
        self.behavior_model.context_scores = defaultdict(
            float, model_data.get("context_scores", {})
        )
        self.routine_detector.detected_routines = model_data.get("routines", [])
        self.anomaly_detector.normal_patterns = model_data.get("normal_patterns", {})
        self.anomaly_detector.baseline_established = bool(
            model_data.get("normal_patterns")
        )

        logger.info(f"Imported model for user {self.user_id}")


# Example usage
if __name__ == "__main__":
    # Create analytics engine
    analytics = PredictiveAnalytics(user_id="user_123")

    # Simulate some interactions
    analytics.record_interaction(
        action="check_email",
        context={"location": "office", "device": "laptop"},
        success=True,
        feedback_score=0.8
    )

    analytics.record_interaction(
        action="schedule_meeting",
        context={"location": "office", "time": "morning"},
        success=True,
        feedback_score=0.9
    )

    # Analyze patterns
    patterns = analytics.analyze_patterns()
    print("Detected patterns:", json.dumps(patterns, indent=2, default=str))

    # Get predictions
    predictions = analytics.predict_next_action(
        current_context={"location": "office", "device": "laptop"}
    )
    print("\nPredictions:", json.dumps(predictions, indent=2))

    # Get insights
    insights = analytics.get_insights()
    print("\nInsights:", json.dumps(insights, indent=2, default=str))
