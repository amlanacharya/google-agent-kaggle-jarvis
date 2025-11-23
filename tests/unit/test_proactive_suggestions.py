"""
Unit tests for Proactive Suggestion System
"""

import pytest
from datetime import datetime, timedelta
from src.core.proactive_suggestions import (
    Suggestion,
    SuggestionGenerator,
    SuggestionRanker,
    ProactiveSuggestionSystem,
    SuggestionType,
    SuggestionPriority
)


class TestSuggestion:
    """Test Suggestion class"""

    def test_initialization(self):
        """Test suggestion initialization"""
        suggestion = Suggestion(
            suggestion_id="sugg_1",
            suggestion_type=SuggestionType.REMINDER,
            priority=SuggestionPriority.HIGH,
            title="Test Reminder",
            description="This is a test"
        )

        assert suggestion.suggestion_id == "sugg_1"
        assert suggestion.suggestion_type == SuggestionType.REMINDER
        assert suggestion.priority == SuggestionPriority.HIGH
        assert not suggestion.accepted
        assert not suggestion.dismissed

    def test_expiration(self):
        """Test suggestion expiration"""
        expires_at = datetime.now() + timedelta(hours=1)
        suggestion = Suggestion(
            suggestion_id="sugg_1",
            suggestion_type=SuggestionType.INFORMATION,
            priority=SuggestionPriority.MEDIUM,
            title="Test",
            description="Test",
            expires_at=expires_at
        )

        assert not suggestion.is_expired()

        # Create expired suggestion
        expired_suggestion = Suggestion(
            suggestion_id="sugg_2",
            suggestion_type=SuggestionType.INFORMATION,
            priority=SuggestionPriority.LOW,
            title="Test",
            description="Test",
            expires_at=datetime.now() - timedelta(hours=1)
        )

        assert expired_suggestion.is_expired()

    def test_relevance(self):
        """Test relevance checking"""
        suggestion = Suggestion(
            suggestion_id="sugg_1",
            suggestion_type=SuggestionType.ROUTINE,
            priority=SuggestionPriority.MEDIUM,
            title="Test",
            description="Test",
            context={"location": "office"}
        )

        assert suggestion.is_relevant({"location": "office"})
        assert not suggestion.is_relevant({"location": "home"})

    def test_accept_dismiss(self):
        """Test accepting and dismissing suggestions"""
        suggestion = Suggestion(
            suggestion_id="sugg_1",
            suggestion_type=SuggestionType.AUTOMATION,
            priority=SuggestionPriority.HIGH,
            title="Test",
            description="Test"
        )

        suggestion.accept()
        assert suggestion.accepted

        suggestion2 = Suggestion(
            suggestion_id="sugg_2",
            suggestion_type=SuggestionType.INFORMATION,
            priority=SuggestionPriority.LOW,
            title="Test",
            description="Test"
        )

        suggestion2.dismiss()
        assert suggestion2.dismissed
        assert not suggestion2.is_relevant({})  # Dismissed not relevant


class TestSuggestionGenerator:
    """Test SuggestionGenerator class"""

    def test_initialization(self):
        """Test generator initialization"""
        generator = SuggestionGenerator()
        assert generator.suggestion_counter == 0

    def test_generate_routine_suggestion(self):
        """Test routine suggestion generation"""
        generator = SuggestionGenerator()

        routine = {
            "action": "check_email",
            "type": "time_based",
            "pattern": {
                "time_of_day": datetime.now().hour,
                "days_of_week": list(range(7))
            }
        }

        suggestion = generator.generate_routine_suggestion(
            routine,
            datetime.now()
        )

        assert suggestion is not None
        assert suggestion.suggestion_type == SuggestionType.ROUTINE
        assert "check_email" in suggestion.title

    def test_generate_warning_suggestion(self):
        """Test warning suggestion generation"""
        generator = SuggestionGenerator()

        anomaly = {
            "type": "unusual_activity",
            "severity": "high",
            "description": "Unusual login pattern detected"
        }

        suggestion = generator.generate_warning_suggestion(anomaly)

        assert suggestion.suggestion_type == SuggestionType.WARNING
        assert suggestion.priority == SuggestionPriority.CRITICAL

    def test_generate_automation_suggestion(self):
        """Test automation suggestion generation"""
        generator = SuggestionGenerator()

        repetitive_task = {
            "task": "daily_report",
            "count": 30,
            "time_savings": "1 hour"
        }

        suggestion = generator.generate_automation_suggestion(repetitive_task)

        assert suggestion.suggestion_type == SuggestionType.AUTOMATION
        assert "automate" in suggestion.title.lower()

    def test_generate_information_suggestion(self):
        """Test information suggestion generation"""
        generator = SuggestionGenerator()

        info = {
            "title": "Market Update",
            "description": "Stock market opened higher",
            "content": "Detailed information..."
        }

        suggestion = generator.generate_information_suggestion(info, relevance_score=0.9)

        assert suggestion.suggestion_type == SuggestionType.INFORMATION
        assert suggestion.priority == SuggestionPriority.HIGH  # High relevance


class TestSuggestionRanker:
    """Test SuggestionRanker class"""

    def test_initialization(self):
        """Test ranker initialization"""
        ranker = SuggestionRanker()
        assert len(ranker.priority_weights) > 0

    def test_rank_suggestions(self):
        """Test ranking suggestions"""
        ranker = SuggestionRanker()

        suggestions = [
            Suggestion(
                "sugg_1",
                SuggestionType.INFORMATION,
                SuggestionPriority.LOW,
                "Low priority",
                "Test"
            ),
            Suggestion(
                "sugg_2",
                SuggestionType.WARNING,
                SuggestionPriority.CRITICAL,
                "Critical warning",
                "Test"
            ),
            Suggestion(
                "sugg_3",
                SuggestionType.ROUTINE,
                SuggestionPriority.MEDIUM,
                "Medium priority",
                "Test"
            )
        ]

        ranked = ranker.rank_suggestions(suggestions, {})

        # Critical should be first
        assert ranked[0].priority == SuggestionPriority.CRITICAL

    def test_context_matching(self):
        """Test context-based ranking"""
        ranker = SuggestionRanker()

        suggestion_office = Suggestion(
            "sugg_1",
            SuggestionType.ROUTINE,
            SuggestionPriority.MEDIUM,
            "Office task",
            "Test",
            context={"location": "office"}
        )

        suggestion_home = Suggestion(
            "sugg_2",
            SuggestionType.ROUTINE,
            SuggestionPriority.MEDIUM,
            "Home task",
            "Test",
            context={"location": "home"}
        )

        # Office context - office suggestion should rank higher
        ranked = ranker.rank_suggestions(
            [suggestion_office, suggestion_home],
            {"location": "office"}
        )

        assert ranked[0].suggestion_id == "sugg_1"


class TestProactiveSuggestionSystem:
    """Test ProactiveSuggestionSystem class"""

    def test_initialization(self):
        """Test system initialization"""
        system = ProactiveSuggestionSystem()

        assert system.generator is not None
        assert system.ranker is not None
        assert len(system.active_suggestions) == 0

    def test_add_suggestion(self):
        """Test adding suggestions"""
        system = ProactiveSuggestionSystem()

        suggestion = Suggestion(
            "sugg_1",
            SuggestionType.INFORMATION,
            SuggestionPriority.MEDIUM,
            "Test",
            "Description"
        )

        success = system.add_suggestion(suggestion)
        assert success
        assert "sugg_1" in system.active_suggestions

    def test_generate_from_routine(self):
        """Test generating suggestion from routine"""
        system = ProactiveSuggestionSystem()

        routine = {
            "action": "morning_routine",
            "type": "time_based",
            "pattern": {
                "time_of_day": datetime.now().hour,
                "days_of_week": list(range(7))
            }
        }

        system.generate_from_routine(routine)

        assert len(system.active_suggestions) > 0

    def test_generate_from_anomaly(self):
        """Test generating suggestion from anomaly"""
        system = ProactiveSuggestionSystem()

        anomaly = {
            "type": "unusual_frequency",
            "severity": "medium",
            "description": "Unusual activity detected"
        }

        system.generate_from_anomaly(anomaly)

        assert len(system.active_suggestions) > 0

        # Check it's a warning type
        suggestion = list(system.active_suggestions.values())[0]
        assert suggestion.suggestion_type == SuggestionType.WARNING

    def test_get_suggestions(self):
        """Test getting ranked suggestions"""
        system = ProactiveSuggestionSystem()

        # Add multiple suggestions
        for i in range(5):
            suggestion = Suggestion(
                f"sugg_{i}",
                SuggestionType.INFORMATION,
                SuggestionPriority.MEDIUM,
                f"Suggestion {i}",
                "Test"
            )
            system.add_suggestion(suggestion)

        suggestions = system.get_suggestions(
            current_context={},
            max_suggestions=3
        )

        assert len(suggestions) <= 3
        assert all("suggestion_id" in s for s in suggestions)

    def test_accept_suggestion(self):
        """Test accepting a suggestion"""
        system = ProactiveSuggestionSystem()

        suggestion = Suggestion(
            "sugg_1",
            SuggestionType.AUTOMATION,
            SuggestionPriority.HIGH,
            "Test",
            "Description"
        )

        system.add_suggestion(suggestion)
        success = system.accept_suggestion("sugg_1")

        assert success
        assert "sugg_1" not in system.active_suggestions
        assert len(system.suggestion_history) > 0

    def test_dismiss_suggestion(self):
        """Test dismissing a suggestion"""
        system = ProactiveSuggestionSystem()

        suggestion = Suggestion(
            "sugg_1",
            SuggestionType.INFORMATION,
            SuggestionPriority.LOW,
            "Test",
            "Description"
        )

        system.add_suggestion(suggestion)
        success = system.dismiss_suggestion("sugg_1", remember_pattern=True)

        assert success
        assert "sugg_1" not in system.active_suggestions
        assert len(system.dismissed_patterns) > 0

    def test_dismissed_pattern_blocking(self):
        """Test that dismissed patterns are blocked"""
        system = ProactiveSuggestionSystem()

        suggestion1 = Suggestion(
            "sugg_1",
            SuggestionType.ROUTINE,
            SuggestionPriority.MEDIUM,
            "Morning Coffee",
            "Test"
        )

        system.add_suggestion(suggestion1)
        system.dismiss_suggestion("sugg_1", remember_pattern=True)

        # Try to add similar suggestion
        suggestion2 = Suggestion(
            "sugg_2",
            SuggestionType.ROUTINE,
            SuggestionPriority.MEDIUM,
            "Morning Coffee",
            "Test"
        )

        success = system.add_suggestion(suggestion2)
        assert not success  # Should be blocked

    def test_cleanup_expired(self):
        """Test cleanup of expired suggestions"""
        system = ProactiveSuggestionSystem()

        # Add expired suggestion
        expired_suggestion = Suggestion(
            "sugg_1",
            SuggestionType.INFORMATION,
            SuggestionPriority.LOW,
            "Expired",
            "Test",
            expires_at=datetime.now() - timedelta(hours=1)
        )

        system.add_suggestion(expired_suggestion)
        assert "sugg_1" in system.active_suggestions

        # Get suggestions (triggers cleanup)
        system.get_suggestions({})

        assert "sugg_1" not in system.active_suggestions

    def test_statistics(self):
        """Test getting statistics"""
        system = ProactiveSuggestionSystem()

        # Add and accept some suggestions
        for i in range(3):
            suggestion = Suggestion(
                f"sugg_{i}",
                SuggestionType.INFORMATION,
                SuggestionPriority.MEDIUM,
                f"Test {i}",
                "Description"
            )
            system.add_suggestion(suggestion)

        system.accept_suggestion("sugg_0")
        system.dismiss_suggestion("sugg_1")

        stats = system.get_statistics()

        assert stats["total_suggestions_shown"] == 2
        assert stats["total_accepted"] == 1
        assert "acceptance_rate" in stats

    def test_export_state(self):
        """Test exporting system state"""
        system = ProactiveSuggestionSystem()

        suggestion = Suggestion(
            "sugg_1",
            SuggestionType.AUTOMATION,
            SuggestionPriority.HIGH,
            "Test",
            "Description"
        )

        system.add_suggestion(suggestion)

        state = system.export_state()

        assert "active_suggestions" in state
        assert "dismissed_patterns" in state
        assert "statistics" in state


@pytest.fixture
def populated_system():
    """Fixture providing a system with suggestions"""
    system = ProactiveSuggestionSystem()

    # Add various suggestions
    priorities = [
        SuggestionPriority.CRITICAL,
        SuggestionPriority.HIGH,
        SuggestionPriority.MEDIUM,
        SuggestionPriority.LOW
    ]

    types = [
        SuggestionType.REMINDER,
        SuggestionType.AUTOMATION,
        SuggestionType.INFORMATION,
        SuggestionType.WARNING
    ]

    for i, (priority, stype) in enumerate(zip(priorities, types)):
        suggestion = Suggestion(
            f"sugg_{i}",
            stype,
            priority,
            f"Suggestion {i}",
            f"Description {i}"
        )
        system.add_suggestion(suggestion)

    return system


def test_with_populated_system(populated_system):
    """Test using populated system fixture"""
    assert len(populated_system.active_suggestions) == 4

    suggestions = populated_system.get_suggestions({}, max_suggestions=5)
    assert len(suggestions) == 4

    # Critical should be first
    assert suggestions[0]["priority"] == "critical"
