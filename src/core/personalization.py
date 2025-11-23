"""
Personalization Engine for JARVIS
Handles adaptive learning, preference evolution, and A/B testing
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import random
import json
from enum import Enum

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for personalization"""
    EXPLORATION = "exploration"  # Try new things
    EXPLOITATION = "exploitation"  # Use known preferences
    BALANCED = "balanced"  # Mix of both


class PreferenceCategory(Enum):
    """Categories of user preferences"""
    COMMUNICATION = "communication"  # Email, messaging style
    SCHEDULING = "scheduling"  # Calendar preferences
    INFORMATION = "information"  # News, weather, updates
    AUTOMATION = "automation"  # Task automation preferences
    INTERACTION = "interaction"  # Voice, text, UI preferences


class PreferenceTracker:
    """Tracks and evolves user preferences over time"""

    def __init__(self, user_id: str, learning_rate: float = 0.1):
        self.user_id = user_id
        self.learning_rate = learning_rate
        self.preferences: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.preference_history: List[Dict[str, Any]] = []
        self.evolution_rate: Dict[str, float] = defaultdict(lambda: 0.0)

    def update_preference(
        self,
        category: str,
        preference_key: str,
        feedback_score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Update preference based on user feedback"""
        current_score = self.preferences[category][preference_key]

        # Apply learning rate with decay over time
        decay_factor = 1.0 / (1.0 + len(self.preference_history) * 0.001)
        adjusted_learning_rate = self.learning_rate * decay_factor

        # Update with exponential moving average
        new_score = (
            (1 - adjusted_learning_rate) * current_score +
            adjusted_learning_rate * feedback_score
        )

        self.preferences[category][preference_key] = new_score

        # Track evolution
        evolution = abs(new_score - current_score)
        self.evolution_rate[category] = (
            0.9 * self.evolution_rate[category] + 0.1 * evolution
        )

        # Record history
        self.preference_history.append({
            "category": category,
            "preference_key": preference_key,
            "old_score": current_score,
            "new_score": new_score,
            "feedback": feedback_score,
            "context": context or {},
            "timestamp": datetime.now()
        })

        logger.debug(
            f"Updated preference {category}:{preference_key}: "
            f"{current_score:.3f} -> {new_score:.3f}"
        )

    def get_preference(self, category: str, preference_key: str) -> float:
        """Get current preference score"""
        return self.preferences[category].get(preference_key, 0.5)  # Default to neutral

    def get_top_preferences(
        self,
        category: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top N preferences in a category"""
        prefs = self.preferences[category]
        return sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_evolution_rate(self, category: str) -> float:
        """Get rate of preference evolution for a category"""
        return self.evolution_rate.get(category, 0.0)

    def is_preference_stable(
        self,
        category: str,
        stability_threshold: float = 0.1
    ) -> bool:
        """Check if preferences in a category are stable"""
        return self.get_evolution_rate(category) < stability_threshold

    def predict_preference_drift(
        self,
        category: str,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """Predict if preferences are likely to change"""
        cutoff = datetime.now() - timedelta(days=time_window_days)
        recent_changes = [
            h for h in self.preference_history
            if h["category"] == category and h["timestamp"] > cutoff
        ]

        if not recent_changes:
            return {"drifting": False, "confidence": 0.0}

        # Calculate average change magnitude
        avg_change = sum(
            abs(h["new_score"] - h["old_score"])
            for h in recent_changes
        ) / len(recent_changes)

        return {
            "drifting": avg_change > 0.2,
            "confidence": min(avg_change * 5, 1.0),
            "recent_changes": len(recent_changes),
            "avg_change_magnitude": avg_change
        }


class ABTest:
    """A/B test for comparing different approaches"""

    def __init__(
        self,
        test_id: str,
        variants: List[str],
        metric: str,
        allocation_strategy: str = "even"
    ):
        self.test_id = test_id
        self.variants = variants
        self.metric = metric
        self.allocation_strategy = allocation_strategy
        self.variant_results: Dict[str, List[float]] = defaultdict(list)
        self.variant_counts: Dict[str, int] = defaultdict(int)
        self.started_at = datetime.now()
        self.completed = False

    def assign_variant(self, user_id: str) -> str:
        """Assign a variant to a user"""
        if self.allocation_strategy == "even":
            # Round-robin or random even distribution
            return random.choice(self.variants)
        elif self.allocation_strategy == "thompson":
            # Thompson sampling for multi-armed bandit
            return self._thompson_sampling()
        else:
            return self.variants[0]

    def _thompson_sampling(self) -> str:
        """Thompson sampling for adaptive allocation"""
        samples = {}

        for variant in self.variants:
            results = self.variant_results.get(variant, [1.0])  # Prior
            # Sample from beta distribution
            alpha = sum(results) + 1
            beta = len(results) - sum(results) + 1
            samples[variant] = random.betavariate(alpha, beta)

        return max(samples.items(), key=lambda x: x[1])[0]

    def record_result(self, variant: str, score: float):
        """Record result for a variant"""
        if variant not in self.variants:
            logger.warning(f"Unknown variant: {variant}")
            return

        self.variant_results[variant].append(score)
        self.variant_counts[variant] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics"""
        stats = {}

        for variant in self.variants:
            results = self.variant_results.get(variant, [])
            if results:
                stats[variant] = {
                    "mean": sum(results) / len(results),
                    "count": len(results),
                    "sum": sum(results),
                    "min": min(results),
                    "max": max(results)
                }
            else:
                stats[variant] = {
                    "mean": 0.0,
                    "count": 0,
                    "sum": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }

        return stats

    def get_winner(self, min_samples: int = 30) -> Optional[str]:
        """Determine winning variant if statistically significant"""
        stats = self.get_statistics()

        # Check if we have enough samples
        if any(s["count"] < min_samples for s in stats.values()):
            return None

        # Find variant with highest mean
        winner = max(stats.items(), key=lambda x: x[1]["mean"])

        # Simple significance check (95% confidence)
        winner_mean = winner[1]["mean"]
        winner_count = winner[1]["count"]

        for variant, stat in stats.items():
            if variant == winner[0]:
                continue

            # Check if difference is significant
            diff = winner_mean - stat["mean"]
            if diff < 0.1:  # Less than 10% improvement
                return None

        return winner[0]

    def complete_test(self) -> Dict[str, Any]:
        """Mark test as complete and return results"""
        self.completed = True
        winner = self.get_winner()

        return {
            "test_id": self.test_id,
            "variants": self.variants,
            "winner": winner,
            "statistics": self.get_statistics(),
            "duration_days": (datetime.now() - self.started_at).days,
            "completed_at": datetime.now()
        }


class ABTestingFramework:
    """Framework for managing A/B tests"""

    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: List[Dict[str, Any]] = []
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)

    def create_test(
        self,
        test_id: str,
        variants: List[str],
        metric: str,
        allocation_strategy: str = "even"
    ) -> ABTest:
        """Create a new A/B test"""
        test = ABTest(test_id, variants, metric, allocation_strategy)
        self.active_tests[test_id] = test
        logger.info(f"Created A/B test: {test_id} with variants {variants}")
        return test

    def get_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Get assigned variant for a user"""
        if test_id not in self.active_tests:
            return None

        # Check if user already has an assignment
        if user_id in self.user_assignments and test_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][test_id]

        # Assign new variant
        test = self.active_tests[test_id]
        variant = test.assign_variant(user_id)
        self.user_assignments[user_id][test_id] = variant

        return variant

    def record_result(self, test_id: str, user_id: str, score: float):
        """Record result for a user's assigned variant"""
        if test_id not in self.active_tests:
            logger.warning(f"Test not found: {test_id}")
            return

        variant = self.user_assignments[user_id].get(test_id)
        if not variant:
            logger.warning(f"No variant assigned for user {user_id} in test {test_id}")
            return

        test = self.active_tests[test_id]
        test.record_result(variant, score)

    def check_test_completion(
        self,
        test_id: str,
        min_samples: int = 30,
        max_duration_days: int = 30
    ) -> bool:
        """Check if test should be completed"""
        if test_id not in self.active_tests:
            return False

        test = self.active_tests[test_id]

        # Check duration
        duration = (datetime.now() - test.started_at).days
        if duration >= max_duration_days:
            self._complete_test(test_id)
            return True

        # Check if winner is clear
        winner = test.get_winner(min_samples)
        if winner:
            self._complete_test(test_id)
            return True

        return False

    def _complete_test(self, test_id: str):
        """Complete a test and store results"""
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]
        results = test.complete_test()
        self.completed_tests.append(results)
        del self.active_tests[test_id]

        logger.info(f"Completed test {test_id}, winner: {results['winner']}")

    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a completed test"""
        for result in self.completed_tests:
            if result["test_id"] == test_id:
                return result
        return None


class PersonalizationEngine:
    """Main personalization engine combining all components"""

    def __init__(
        self,
        user_id: str,
        learning_rate: float = 0.1,
        learning_mode: LearningMode = LearningMode.BALANCED
    ):
        self.user_id = user_id
        self.learning_mode = learning_mode
        self.preference_tracker = PreferenceTracker(user_id, learning_rate)
        self.ab_testing = ABTestingFramework()
        self.exploration_rate = 0.2  # 20% exploration by default
        self.adaptation_history: List[Dict[str, Any]] = []

    def set_learning_mode(self, mode: LearningMode):
        """Set the learning mode"""
        self.learning_mode = mode

        # Adjust exploration rate based on mode
        if mode == LearningMode.EXPLORATION:
            self.exploration_rate = 0.4
        elif mode == LearningMode.EXPLOITATION:
            self.exploration_rate = 0.05
        else:  # BALANCED
            self.exploration_rate = 0.2

        logger.info(f"Learning mode set to {mode.value}, exploration rate: {self.exploration_rate}")

    def should_explore(self) -> bool:
        """Determine if we should explore vs exploit"""
        return random.random() < self.exploration_rate

    def learn_from_interaction(
        self,
        category: str,
        option_chosen: str,
        outcome_score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Learn from user interaction"""
        self.preference_tracker.update_preference(
            category=category,
            preference_key=option_chosen,
            feedback_score=outcome_score,
            context=context
        )

        # Record adaptation
        self.adaptation_history.append({
            "category": category,
            "option": option_chosen,
            "score": outcome_score,
            "timestamp": datetime.now()
        })

    def recommend_option(
        self,
        category: str,
        available_options: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, str]:
        """Recommend an option based on learned preferences"""

        # Check if we should explore
        if self.should_explore() and len(available_options) > 1:
            # Exploration: choose randomly
            choice = random.choice(available_options)
            return choice, 0.5, "exploration"

        # Exploitation: choose based on preferences
        option_scores = {
            option: self.preference_tracker.get_preference(category, option)
            for option in available_options
        }

        best_option = max(option_scores.items(), key=lambda x: x[1])
        return best_option[0], best_option[1], "exploitation"

    def get_personalization_insights(self) -> Dict[str, Any]:
        """Get insights about current personalization state"""
        insights = {
            "user_id": self.user_id,
            "learning_mode": self.learning_mode.value,
            "exploration_rate": self.exploration_rate,
            "total_adaptations": len(self.adaptation_history),
            "categories": {}
        }

        # Analyze each category
        for category in PreferenceCategory:
            cat_name = category.value
            top_prefs = self.preference_tracker.get_top_preferences(cat_name, 3)
            evolution_rate = self.preference_tracker.get_evolution_rate(cat_name)
            is_stable = self.preference_tracker.is_preference_stable(cat_name)

            insights["categories"][cat_name] = {
                "top_preferences": [
                    {"option": opt, "score": score}
                    for opt, score in top_prefs
                ],
                "evolution_rate": evolution_rate,
                "is_stable": is_stable
            }

        return insights

    def adapt_to_feedback(self, feedback: Dict[str, Any]):
        """Adapt learning based on explicit user feedback"""
        if "learning_rate_adjustment" in feedback:
            adjustment = feedback["learning_rate_adjustment"]
            new_rate = max(0.01, min(0.5, self.preference_tracker.learning_rate + adjustment))
            self.preference_tracker.learning_rate = new_rate
            logger.info(f"Adjusted learning rate to {new_rate}")

        if "learning_mode" in feedback:
            mode_str = feedback["learning_mode"]
            try:
                new_mode = LearningMode(mode_str)
                self.set_learning_mode(new_mode)
            except ValueError:
                logger.warning(f"Invalid learning mode: {mode_str}")

    def export_personalization(self) -> Dict[str, Any]:
        """Export personalization data for persistence"""
        return {
            "user_id": self.user_id,
            "learning_mode": self.learning_mode.value,
            "exploration_rate": self.exploration_rate,
            "preferences": dict(self.preference_tracker.preferences),
            "evolution_rates": dict(self.preference_tracker.evolution_rate),
            "learning_rate": self.preference_tracker.learning_rate,
            "adaptation_count": len(self.adaptation_history),
            "exported_at": datetime.now().isoformat()
        }

    def import_personalization(self, data: Dict[str, Any]):
        """Import personalization data"""
        self.learning_mode = LearningMode(data.get("learning_mode", "balanced"))
        self.exploration_rate = data.get("exploration_rate", 0.2)
        self.preference_tracker.preferences = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data.get("preferences", {}).items()}
        )
        self.preference_tracker.evolution_rate = defaultdict(
            float,
            data.get("evolution_rates", {})
        )
        self.preference_tracker.learning_rate = data.get("learning_rate", 0.1)

        logger.info(f"Imported personalization for user {self.user_id}")


# Example usage
if __name__ == "__main__":
    # Create personalization engine
    engine = PersonalizationEngine(user_id="user_123")

    # Simulate learning from interactions
    engine.learn_from_interaction(
        category="communication",
        option_chosen="concise_style",
        outcome_score=0.9,
        context={"time": "morning"}
    )

    engine.learn_from_interaction(
        category="communication",
        option_chosen="detailed_style",
        outcome_score=0.6,
        context={"time": "morning"}
    )

    # Get recommendation
    option, confidence, strategy = engine.recommend_option(
        category="communication",
        available_options=["concise_style", "detailed_style", "balanced_style"]
    )
    print(f"Recommended: {option} (confidence: {confidence:.2f}, strategy: {strategy})")

    # Create A/B test
    ab_test = engine.ab_testing.create_test(
        test_id="email_tone_test",
        variants=["formal", "casual", "friendly"],
        metric="user_satisfaction"
    )

    # Get variant for user
    variant = engine.ab_testing.get_variant("email_tone_test", "user_123")
    print(f"A/B test variant: {variant}")

    # Record results
    engine.ab_testing.record_result("email_tone_test", "user_123", 0.85)

    # Get insights
    insights = engine.get_personalization_insights()
    print("\nPersonalization insights:")
    print(json.dumps(insights, indent=2, default=str))
