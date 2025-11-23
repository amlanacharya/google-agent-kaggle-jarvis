"""
Unit tests for Personalization Engine
"""

import pytest
from src.core.personalization import (
    PreferenceTracker,
    ABTest,
    ABTestingFramework,
    PersonalizationEngine,
    LearningMode,
    PreferenceCategory
)


class TestPreferenceTracker:
    """Test PreferenceTracker class"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = PreferenceTracker("user_123", learning_rate=0.1)
        assert tracker.user_id == "user_123"
        assert tracker.learning_rate == 0.1
        assert len(tracker.preferences) == 0

    def test_update_preference(self):
        """Test preference updates"""
        tracker = PreferenceTracker("user_123", learning_rate=0.2)

        tracker.update_preference(
            category="communication",
            preference_key="concise_style",
            feedback_score=0.9
        )

        score = tracker.get_preference("communication", "concise_style")
        assert score > 0.5  # Should be above neutral

    def test_preference_evolution(self):
        """Test that preferences evolve over time"""
        tracker = PreferenceTracker("user_123", learning_rate=0.1)

        # Initial preference
        tracker.update_preference("test", "option_a", 0.8)
        score1 = tracker.get_preference("test", "option_a")

        # Update again
        tracker.update_preference("test", "option_a", 0.9)
        score2 = tracker.get_preference("test", "option_a")

        assert score2 > score1  # Should increase

    def test_top_preferences(self):
        """Test getting top preferences"""
        tracker = PreferenceTracker("user_123")

        # Add multiple preferences
        tracker.update_preference("category", "option_a", 0.9)
        tracker.update_preference("category", "option_b", 0.7)
        tracker.update_preference("category", "option_c", 0.5)

        top = tracker.get_top_preferences("category", top_n=2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]  # Sorted by score

    def test_preference_stability(self):
        """Test preference stability detection"""
        tracker = PreferenceTracker("user_123")

        # Add stable preference
        for _ in range(5):
            tracker.update_preference("test", "stable", 0.8)

        # Should be stable
        assert tracker.get_evolution_rate("test") < 0.2


class TestABTest:
    """Test ABTest class"""

    def test_initialization(self):
        """Test A/B test initialization"""
        test = ABTest(
            test_id="test_1",
            variants=["a", "b", "c"],
            metric="satisfaction"
        )

        assert test.test_id == "test_1"
        assert len(test.variants) == 3
        assert not test.completed

    def test_assign_variant(self):
        """Test variant assignment"""
        test = ABTest(
            test_id="test_1",
            variants=["a", "b"],
            metric="clicks",
            allocation_strategy="even"
        )

        variant = test.assign_variant("user_123")
        assert variant in ["a", "b"]

    def test_record_result(self):
        """Test recording results"""
        test = ABTest("test_1", ["a", "b"], "metric")

        test.record_result("a", 0.8)
        test.record_result("a", 0.9)
        test.record_result("b", 0.6)

        stats = test.get_statistics()
        assert stats["a"]["count"] == 2
        assert stats["b"]["count"] == 1

    def test_get_winner(self):
        """Test winner determination"""
        test = ABTest("test_1", ["a", "b"], "metric")

        # Add enough samples with clear winner
        for _ in range(40):
            test.record_result("a", 0.9)
        for _ in range(40):
            test.record_result("b", 0.6)

        winner = test.get_winner(min_samples=30)
        assert winner == "a"


class TestABTestingFramework:
    """Test ABTestingFramework class"""

    def test_create_test(self):
        """Test creating A/B test"""
        framework = ABTestingFramework()

        test = framework.create_test(
            test_id="test_1",
            variants=["control", "treatment"],
            metric="conversion"
        )

        assert test.test_id == "test_1"
        assert "test_1" in framework.active_tests

    def test_get_variant(self):
        """Test getting variant for user"""
        framework = ABTestingFramework()

        framework.create_test(
            test_id="test_1",
            variants=["a", "b"],
            metric="test"
        )

        variant1 = framework.get_variant("test_1", "user_123")
        variant2 = framework.get_variant("test_1", "user_123")

        # Should get same variant consistently
        assert variant1 == variant2

    def test_record_result(self):
        """Test recording test results"""
        framework = ABTestingFramework()

        framework.create_test("test_1", ["a", "b"], "metric")
        variant = framework.get_variant("test_1", "user_123")

        framework.record_result("test_1", "user_123", 0.8)

        test = framework.active_tests["test_1"]
        assert len(test.variant_results[variant]) > 0

    def test_complete_test(self):
        """Test completing a test"""
        framework = ABTestingFramework()

        test = framework.create_test("test_1", ["a", "b"], "metric")

        # Add many results
        for i in range(50):
            user_id = f"user_{i}"
            variant = framework.get_variant("test_1", user_id)
            score = 0.9 if variant == "a" else 0.5
            framework.record_result("test_1", user_id, score)

        # Check if should complete
        completed = framework.check_test_completion("test_1", min_samples=30)

        if completed:
            assert "test_1" not in framework.active_tests
            assert len(framework.completed_tests) > 0


class TestPersonalizationEngine:
    """Test PersonalizationEngine class"""

    def test_initialization(self):
        """Test engine initialization"""
        engine = PersonalizationEngine("user_123")

        assert engine.user_id == "user_123"
        assert engine.learning_mode == LearningMode.BALANCED
        assert 0 < engine.exploration_rate < 1

    def test_set_learning_mode(self):
        """Test setting learning mode"""
        engine = PersonalizationEngine("user_123")

        engine.set_learning_mode(LearningMode.EXPLORATION)
        assert engine.learning_mode == LearningMode.EXPLORATION
        assert engine.exploration_rate > 0.3  # Higher for exploration

        engine.set_learning_mode(LearningMode.EXPLOITATION)
        assert engine.exploration_rate < 0.1  # Lower for exploitation

    def test_should_explore(self):
        """Test exploration decision"""
        engine = PersonalizationEngine("user_123")
        engine.exploration_rate = 0.5

        # Run multiple times to test randomness
        explore_count = sum(1 for _ in range(100) if engine.should_explore())

        # Should be roughly 50% (with some tolerance)
        assert 30 < explore_count < 70

    def test_learn_from_interaction(self):
        """Test learning from interactions"""
        engine = PersonalizationEngine("user_123")

        engine.learn_from_interaction(
            category="communication",
            option_chosen="concise",
            outcome_score=0.9
        )

        assert len(engine.adaptation_history) > 0

        score = engine.preference_tracker.get_preference("communication", "concise")
        assert score > 0.5

    def test_recommend_option(self):
        """Test option recommendation"""
        engine = PersonalizationEngine("user_123")

        # Train preferences
        for _ in range(5):
            engine.learn_from_interaction(
                category="test",
                option_chosen="option_a",
                outcome_score=0.9
            )

        # Get recommendation
        option, confidence, strategy = engine.recommend_option(
            category="test",
            available_options=["option_a", "option_b", "option_c"]
        )

        assert option in ["option_a", "option_b", "option_c"]
        assert 0 <= confidence <= 1
        assert strategy in ["exploration", "exploitation"]

    def test_personalization_insights(self):
        """Test insights generation"""
        engine = PersonalizationEngine("user_123")

        # Add some interactions
        for i in range(10):
            engine.learn_from_interaction(
                category="communication",
                option_chosen="style_a",
                outcome_score=0.8
            )

        insights = engine.get_personalization_insights()

        assert "user_id" in insights
        assert "learning_mode" in insights
        assert "categories" in insights
        assert insights["total_adaptations"] == 10

    def test_adapt_to_feedback(self):
        """Test adapting to user feedback"""
        engine = PersonalizationEngine("user_123")
        initial_rate = engine.preference_tracker.learning_rate

        engine.adapt_to_feedback({
            "learning_rate_adjustment": 0.05
        })

        assert engine.preference_tracker.learning_rate != initial_rate

    def test_export_import(self):
        """Test export and import"""
        engine = PersonalizationEngine("user_123")

        # Add data
        engine.learn_from_interaction(
            category="test",
            option_chosen="option_a",
            outcome_score=0.8
        )

        # Export
        exported = engine.export_personalization()
        assert exported["user_id"] == "user_123"

        # Import to new engine
        new_engine = PersonalizationEngine("user_123")
        new_engine.import_personalization(exported)

        assert new_engine.learning_mode == engine.learning_mode


@pytest.fixture
def trained_engine():
    """Fixture providing a trained personalization engine"""
    engine = PersonalizationEngine("test_user")

    # Train with various interactions
    categories = ["communication", "scheduling", "information"]
    options = ["option_a", "option_b", "option_c"]

    for i in range(30):
        engine.learn_from_interaction(
            category=categories[i % 3],
            option_chosen=options[i % 3],
            outcome_score=0.7 + (i % 3) * 0.1
        )

    return engine


def test_with_trained_engine(trained_engine):
    """Test using trained engine fixture"""
    assert len(trained_engine.adaptation_history) == 30
    insights = trained_engine.get_personalization_insights()
    assert insights["total_adaptations"] == 30
