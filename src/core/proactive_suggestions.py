"""
Proactive Suggestion System for JARVIS
Generates context-aware, timely suggestions based on user behavior and patterns
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of proactive suggestions"""
    REMINDER = "reminder"  # Time-based reminders
    OPTIMIZATION = "optimization"  # Efficiency improvements
    OPPORTUNITY = "opportunity"  # New opportunities
    WARNING = "warning"  # Potential issues
    AUTOMATION = "automation"  # Automation recommendations
    INFORMATION = "information"  # Relevant information
    ROUTINE = "routine"  # Routine-based suggestions


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Suggestion:
    """Represents a proactive suggestion"""

    def __init__(
        self,
        suggestion_id: str,
        suggestion_type: SuggestionType,
        priority: SuggestionPriority,
        title: str,
        description: str,
        action: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ):
        self.suggestion_id = suggestion_id
        self.suggestion_type = suggestion_type
        self.priority = priority
        self.title = title
        self.description = description
        self.action = action or {}
        self.context = context or {}
        self.created_at = datetime.now()
        self.expires_at = expires_at
        self.accepted = False
        self.dismissed = False
        self.shown = False

    def is_expired(self) -> bool:
        """Check if suggestion has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def is_relevant(self, current_context: Dict[str, Any]) -> bool:
        """Check if suggestion is relevant in current context"""
        if self.is_expired():
            return False

        if self.dismissed:
            return False

        # Check context matching
        if self.context:
            for key, value in self.context.items():
                if key in current_context and current_context[key] != value:
                    return False

        return True

    def accept(self):
        """Mark suggestion as accepted"""
        self.accepted = True
        logger.info(f"Suggestion accepted: {self.suggestion_id}")

    def dismiss(self):
        """Mark suggestion as dismissed"""
        self.dismissed = True
        logger.info(f"Suggestion dismissed: {self.suggestion_id}")

    def mark_shown(self):
        """Mark suggestion as shown to user"""
        self.shown = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "suggestion_id": self.suggestion_id,
            "type": self.suggestion_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "accepted": self.accepted,
            "dismissed": self.dismissed,
            "shown": self.shown
        }


class SuggestionGenerator:
    """Generates suggestions based on various inputs"""

    def __init__(self):
        self.suggestion_counter = 0

    def generate_routine_suggestion(
        self,
        routine: Dict[str, Any],
        current_time: datetime
    ) -> Optional[Suggestion]:
        """Generate suggestion based on detected routine"""
        if routine["type"] != "time_based":
            return None

        pattern = routine["pattern"]
        action = routine["action"]

        # Check if it's time for this routine
        if (current_time.hour == pattern["time_of_day"] and
            current_time.weekday() in pattern["days_of_week"]):

            return self._create_suggestion(
                suggestion_type=SuggestionType.ROUTINE,
                priority=SuggestionPriority.MEDIUM,
                title=f"Time for {action}",
                description=f"Based on your routine, you typically {action} around this time",
                action={
                    "type": "execute_routine",
                    "routine_action": action
                },
                expires_at=current_time + timedelta(hours=1)
            )

        return None

    def generate_optimization_suggestion(
        self,
        inefficiency: Dict[str, Any]
    ) -> Suggestion:
        """Generate suggestion to improve efficiency"""
        return self._create_suggestion(
            suggestion_type=SuggestionType.OPTIMIZATION,
            priority=SuggestionPriority.HIGH,
            title=f"Optimize {inefficiency['task']}",
            description=f"You could save time by {inefficiency['improvement']}",
            action={
                "type": "apply_optimization",
                "details": inefficiency
            }
        )

    def generate_warning_suggestion(
        self,
        anomaly: Dict[str, Any]
    ) -> Suggestion:
        """Generate warning based on anomaly detection"""
        priority_map = {
            "low": SuggestionPriority.LOW,
            "medium": SuggestionPriority.MEDIUM,
            "high": SuggestionPriority.CRITICAL
        }

        return self._create_suggestion(
            suggestion_type=SuggestionType.WARNING,
            priority=priority_map.get(anomaly["severity"], SuggestionPriority.MEDIUM),
            title=f"Unusual Activity Detected",
            description=anomaly["description"],
            action={
                "type": "investigate_anomaly",
                "anomaly": anomaly
            },
            expires_at=datetime.now() + timedelta(hours=24)
        )

    def generate_automation_suggestion(
        self,
        repetitive_task: Dict[str, Any]
    ) -> Suggestion:
        """Generate suggestion to automate repetitive task"""
        return self._create_suggestion(
            suggestion_type=SuggestionType.AUTOMATION,
            priority=SuggestionPriority.HIGH,
            title=f"Automate {repetitive_task['task']}",
            description=f"This task has been performed {repetitive_task['count']} times. Consider automation.",
            action={
                "type": "create_automation",
                "task": repetitive_task
            }
        )

    def generate_opportunity_suggestion(
        self,
        opportunity: Dict[str, Any]
    ) -> Suggestion:
        """Generate suggestion for new opportunity"""
        return self._create_suggestion(
            suggestion_type=SuggestionType.OPPORTUNITY,
            priority=SuggestionPriority.MEDIUM,
            title=opportunity["title"],
            description=opportunity["description"],
            action={
                "type": "explore_opportunity",
                "details": opportunity
            },
            expires_at=opportunity.get("expires_at")
        )

    def generate_information_suggestion(
        self,
        info: Dict[str, Any],
        relevance_score: float
    ) -> Suggestion:
        """Generate informational suggestion"""
        priority = (
            SuggestionPriority.HIGH if relevance_score > 0.8
            else SuggestionPriority.MEDIUM if relevance_score > 0.5
            else SuggestionPriority.LOW
        )

        return self._create_suggestion(
            suggestion_type=SuggestionType.INFORMATION,
            priority=priority,
            title=info["title"],
            description=info["description"],
            action={
                "type": "show_information",
                "content": info.get("content")
            },
            expires_at=datetime.now() + timedelta(hours=6)
        )

    def generate_reminder_suggestion(
        self,
        reminder: Dict[str, Any]
    ) -> Suggestion:
        """Generate reminder suggestion"""
        return self._create_suggestion(
            suggestion_type=SuggestionType.REMINDER,
            priority=SuggestionPriority.HIGH,
            title=reminder["title"],
            description=reminder["description"],
            action={
                "type": "acknowledge_reminder",
                "reminder_id": reminder.get("id")
            },
            context=reminder.get("context", {}),
            expires_at=reminder.get("expires_at")
        )

    def _create_suggestion(
        self,
        suggestion_type: SuggestionType,
        priority: SuggestionPriority,
        title: str,
        description: str,
        action: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None
    ) -> Suggestion:
        """Create a new suggestion"""
        self.suggestion_counter += 1
        suggestion_id = f"sugg_{self.suggestion_counter}_{int(datetime.now().timestamp())}"

        return Suggestion(
            suggestion_id=suggestion_id,
            suggestion_type=suggestion_type,
            priority=priority,
            title=title,
            description=description,
            action=action,
            context=context,
            expires_at=expires_at
        )


class SuggestionRanker:
    """Ranks suggestions based on relevance and priority"""

    def __init__(self):
        self.priority_weights = {
            SuggestionPriority.CRITICAL: 1.0,
            SuggestionPriority.HIGH: 0.7,
            SuggestionPriority.MEDIUM: 0.4,
            SuggestionPriority.LOW: 0.2
        }

    def rank_suggestions(
        self,
        suggestions: List[Suggestion],
        current_context: Dict[str, Any],
        user_preferences: Optional[Dict[str, float]] = None
    ) -> List[Suggestion]:
        """Rank suggestions by relevance and priority"""
        scored_suggestions = []

        for suggestion in suggestions:
            if not suggestion.is_relevant(current_context):
                continue

            score = self._calculate_score(
                suggestion,
                current_context,
                user_preferences or {}
            )
            scored_suggestions.append((suggestion, score))

        # Sort by score (descending)
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in scored_suggestions]

    def _calculate_score(
        self,
        suggestion: Suggestion,
        current_context: Dict[str, Any],
        user_preferences: Dict[str, float]
    ) -> float:
        """Calculate relevance score for a suggestion"""
        # Base score from priority
        score = self.priority_weights.get(suggestion.priority, 0.5)

        # Boost based on context matching
        context_match = self._calculate_context_match(
            suggestion.context,
            current_context
        )
        score += context_match * 0.3

        # Adjust based on user preferences
        pref_key = f"suggestion_type:{suggestion.suggestion_type.value}"
        preference_score = user_preferences.get(pref_key, 0.5)
        score *= preference_score

        # Penalty for older suggestions
        age_hours = (datetime.now() - suggestion.created_at).total_seconds() / 3600
        age_penalty = max(0, 1.0 - (age_hours / 24))
        score *= age_penalty

        # Penalty if already shown but not acted upon
        if suggestion.shown:
            score *= 0.5

        return score

    def _calculate_context_match(
        self,
        suggestion_context: Dict[str, Any],
        current_context: Dict[str, Any]
    ) -> float:
        """Calculate how well contexts match"""
        if not suggestion_context:
            return 0.5  # Neutral if no context specified

        matches = 0
        total = len(suggestion_context)

        for key, value in suggestion_context.items():
            if key in current_context and current_context[key] == value:
                matches += 1

        return matches / total if total > 0 else 0.0


class ProactiveSuggestionSystem:
    """Main proactive suggestion system"""

    def __init__(self):
        self.generator = SuggestionGenerator()
        self.ranker = SuggestionRanker()
        self.active_suggestions: Dict[str, Suggestion] = {}
        self.suggestion_history: List[Dict[str, Any]] = []
        self.dismissed_patterns: Set[str] = set()

    def add_suggestion(self, suggestion: Suggestion) -> bool:
        """Add a new suggestion to the system"""
        # Check if similar suggestion was recently dismissed
        pattern = f"{suggestion.suggestion_type.value}:{suggestion.title}"
        if pattern in self.dismissed_patterns:
            logger.debug(f"Skipping suggestion - similar pattern dismissed: {pattern}")
            return False

        self.active_suggestions[suggestion.suggestion_id] = suggestion
        logger.info(f"Added suggestion: {suggestion.title} ({suggestion.priority.value})")
        return True

    def generate_from_routine(
        self,
        routine: Dict[str, Any],
        current_time: Optional[datetime] = None
    ):
        """Generate and add suggestion from routine"""
        current_time = current_time or datetime.now()
        suggestion = self.generator.generate_routine_suggestion(routine, current_time)

        if suggestion:
            self.add_suggestion(suggestion)

    def generate_from_anomaly(self, anomaly: Dict[str, Any]):
        """Generate and add warning suggestion from anomaly"""
        suggestion = self.generator.generate_warning_suggestion(anomaly)
        self.add_suggestion(suggestion)

    def generate_automation_opportunity(self, repetitive_task: Dict[str, Any]):
        """Generate automation suggestion"""
        suggestion = self.generator.generate_automation_suggestion(repetitive_task)
        self.add_suggestion(suggestion)

    def generate_optimization(self, inefficiency: Dict[str, Any]):
        """Generate optimization suggestion"""
        suggestion = self.generator.generate_optimization_suggestion(inefficiency)
        self.add_suggestion(suggestion)

    def generate_information(self, info: Dict[str, Any], relevance: float):
        """Generate information suggestion"""
        suggestion = self.generator.generate_information_suggestion(info, relevance)
        self.add_suggestion(suggestion)

    def get_suggestions(
        self,
        current_context: Dict[str, Any],
        user_preferences: Optional[Dict[str, float]] = None,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top suggestions for current context"""
        # Remove expired suggestions
        self._cleanup_expired()

        # Rank suggestions
        active_list = list(self.active_suggestions.values())
        ranked = self.ranker.rank_suggestions(
            active_list,
            current_context,
            user_preferences
        )

        # Mark as shown
        for suggestion in ranked[:max_suggestions]:
            suggestion.mark_shown()

        return [s.to_dict() for s in ranked[:max_suggestions]]

    def accept_suggestion(self, suggestion_id: str) -> bool:
        """Accept a suggestion"""
        if suggestion_id not in self.active_suggestions:
            return False

        suggestion = self.active_suggestions[suggestion_id]
        suggestion.accept()

        # Record in history
        self.suggestion_history.append({
            "suggestion_id": suggestion_id,
            "type": suggestion.suggestion_type.value,
            "accepted": True,
            "timestamp": datetime.now()
        })

        # Remove from active
        del self.active_suggestions[suggestion_id]
        return True

    def dismiss_suggestion(
        self,
        suggestion_id: str,
        remember_pattern: bool = False
    ) -> bool:
        """Dismiss a suggestion"""
        if suggestion_id not in self.active_suggestions:
            return False

        suggestion = self.active_suggestions[suggestion_id]
        suggestion.dismiss()

        # Remember pattern if requested
        if remember_pattern:
            pattern = f"{suggestion.suggestion_type.value}:{suggestion.title}"
            self.dismissed_patterns.add(pattern)
            logger.info(f"Remembering dismissed pattern: {pattern}")

        # Record in history
        self.suggestion_history.append({
            "suggestion_id": suggestion_id,
            "type": suggestion.suggestion_type.value,
            "accepted": False,
            "timestamp": datetime.now()
        })

        # Remove from active
        del self.active_suggestions[suggestion_id]
        return True

    def _cleanup_expired(self):
        """Remove expired suggestions"""
        expired = [
            sid for sid, sugg in self.active_suggestions.items()
            if sugg.is_expired()
        ]

        for sid in expired:
            del self.active_suggestions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired suggestions")

    def get_statistics(self) -> Dict[str, Any]:
        """Get suggestion system statistics"""
        total_shown = len(self.suggestion_history)
        accepted = sum(1 for h in self.suggestion_history if h["accepted"])

        by_type = {}
        for entry in self.suggestion_history:
            stype = entry["type"]
            if stype not in by_type:
                by_type[stype] = {"shown": 0, "accepted": 0}
            by_type[stype]["shown"] += 1
            if entry["accepted"]:
                by_type[stype]["accepted"] += 1

        return {
            "total_suggestions_shown": total_shown,
            "total_accepted": accepted,
            "acceptance_rate": accepted / total_shown if total_shown > 0 else 0.0,
            "active_suggestions": len(self.active_suggestions),
            "dismissed_patterns": len(self.dismissed_patterns),
            "by_type": by_type
        }

    def export_state(self) -> Dict[str, Any]:
        """Export system state"""
        return {
            "active_suggestions": [
                s.to_dict() for s in self.active_suggestions.values()
            ],
            "dismissed_patterns": list(self.dismissed_patterns),
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Create suggestion system
    system = ProactiveSuggestionSystem()

    # Generate various suggestions
    system.generate_from_routine(
        routine={
            "action": "check_email",
            "type": "time_based",
            "pattern": {
                "time_of_day": datetime.now().hour,
                "days_of_week": list(range(5)),
                "frequency": "daily"
            }
        }
    )

    system.generate_automation_opportunity({
        "task": "daily_report_generation",
        "count": 25,
        "time_savings": "30 minutes per day"
    })

    system.generate_from_anomaly({
        "type": "unusual_frequency",
        "severity": "medium",
        "description": "Unusual number of meetings scheduled today"
    })

    # Get suggestions
    suggestions = system.get_suggestions(
        current_context={"location": "office", "time": "morning"},
        max_suggestions=3
    )

    print("Top Suggestions:")
    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. [{sugg['priority']}] {sugg['title']}")
        print(f"   {sugg['description']}")

    # Get statistics
    stats = system.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
