"""
Unit tests for Predictive Analytics Engine
"""

import pytest
from datetime import datetime, timedelta
from src.core.predictive_analytics import (
    UserBehaviorModel,
    RoutineDetector,
    AnomalyDetector,
    PredictiveAnalytics
)


class TestUserBehaviorModel:
    """Test UserBehaviorModel class"""

    def test_initialization(self):
        """Test model initialization"""
        model = UserBehaviorModel("user_123")
        assert model.user_id == "user_123"
        assert len(model.interactions) == 0
        assert len(model.preferences) == 0

    def test_record_interaction(self):
        """Test recording interactions"""
        model = UserBehaviorModel("user_123")

        model.record_interaction(
            action="check_email",
            context={"location": "office"},
            success=True,
            feedback_score=0.8
        )

        assert len(model.interactions) == 1
        assert model.get_action_frequency("check_email") == 1
        assert model.get_preference_score("check_email") > 0

    def test_preference_updates(self):
        """Test preference score updates"""
        model = UserBehaviorModel("user_123")

        # Record multiple successful interactions
        for _ in range(5):
            model.record_interaction(
                action="check_email",
                context={"location": "office"},
                success=True,
                feedback_score=0.9
            )

        score = model.get_preference_score("check_email")
        assert score > 1.0  # Should accumulate

    def test_active_hours(self):
        """Test active hours detection"""
        model = UserBehaviorModel("user_123")

        # Record interactions at specific hours
        for hour in [9, 10, 11, 14, 15, 16]:
            for _ in range(3):
                timestamp = datetime.now().replace(hour=hour)
                model.record_interaction(
                    action="work",
                    context={},
                    timestamp=timestamp
                )

        active_hours = model.get_active_hours()
        assert len(active_hours) > 0
        assert all(isinstance(h, int) for h in active_hours)


class TestRoutineDetector:
    """Test RoutineDetector class"""

    def test_initialization(self):
        """Test detector initialization"""
        detector = RoutineDetector(min_occurrences=3)
        assert detector.min_occurrences == 3
        assert len(detector.detected_routines) == 0

    def test_detect_time_patterns(self):
        """Test time-based pattern detection"""
        model = UserBehaviorModel("user_123")
        detector = RoutineDetector(min_occurrences=3)

        # Create routine: check email at 9 AM on weekdays
        for day in range(5):  # Monday to Friday
            for _ in range(3):
                timestamp = datetime.now().replace(
                    hour=9,
                    minute=0
                ) + timedelta(days=day)
                model.record_interaction(
                    action="check_email",
                    context={},
                    timestamp=timestamp
                )

        routines = detector.detect_routines(model)
        assert len(routines) > 0

        # Find the check_email routine
        email_routine = next(
            (r for r in routines if r["action"] == "check_email"),
            None
        )
        assert email_routine is not None
        assert email_routine["type"] == "time_based"

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        model = UserBehaviorModel("user_123")
        detector = RoutineDetector(min_occurrences=2)

        # Consistent pattern
        for _ in range(5):
            timestamp = datetime.now().replace(hour=9)
            model.record_interaction(
                action="task",
                context={},
                timestamp=timestamp
            )

        routines = detector.detect_routines(model)
        if routines:
            assert routines[0]["confidence"] > 0.5


class TestAnomalyDetector:
    """Test AnomalyDetector class"""

    def test_initialization(self):
        """Test detector initialization"""
        detector = AnomalyDetector(sensitivity=0.5)
        assert detector.sensitivity == 0.5
        assert not detector.baseline_established

    def test_establish_baseline(self):
        """Test baseline establishment"""
        model = UserBehaviorModel("user_123")
        detector = AnomalyDetector()

        # Add sufficient interactions
        for i in range(15):
            model.record_interaction(
                action="normal_task",
                context={}
            )

        detector.establish_baseline(model)
        assert detector.baseline_established
        assert len(detector.normal_patterns) > 0

    def test_detect_unusual_time(self):
        """Test detection of unusual activity time"""
        model = UserBehaviorModel("user_123")
        detector = AnomalyDetector()

        # Establish baseline with daytime activity
        for i in range(15):
            timestamp = datetime.now().replace(hour=10)
            model.record_interaction(
                action="work",
                context={},
                timestamp=timestamp
            )

        detector.establish_baseline(model)

        # Add nighttime activity
        late_night = datetime.now().replace(hour=2)
        model.record_interaction(
            action="work",
            context={},
            timestamp=late_night
        )

        anomalies = detector.detect_anomalies([model.interactions[-1]])
        assert len(anomalies) > 0
        assert anomalies[0]["type"] == "unusual_time"


class TestPredictiveAnalytics:
    """Test PredictiveAnalytics main class"""

    def test_initialization(self):
        """Test analytics initialization"""
        analytics = PredictiveAnalytics("user_123")
        assert analytics.user_id == "user_123"
        assert analytics.behavior_model is not None
        assert analytics.routine_detector is not None
        assert analytics.anomaly_detector is not None

    def test_record_and_analyze(self):
        """Test recording and analyzing patterns"""
        analytics = PredictiveAnalytics("user_123")

        # Record multiple interactions
        for _ in range(10):
            analytics.record_interaction(
                action="task",
                context={"location": "office"},
                success=True,
                feedback_score=0.8
            )

        # Analyze patterns
        analysis = analytics.analyze_patterns()
        assert "routines" in analysis
        assert "anomalies" in analysis
        assert "active_hours" in analysis

    def test_predict_next_action(self):
        """Test action prediction"""
        analytics = PredictiveAnalytics("user_123")

        # Record interactions
        for _ in range(5):
            analytics.record_interaction(
                action="check_email",
                context={"location": "office"},
                success=True,
                feedback_score=0.9
            )

        # Predict
        predictions = analytics.predict_next_action(
            current_context={"location": "office"}
        )

        assert isinstance(predictions, list)
        if predictions:
            assert "action" in predictions[0]
            assert "confidence" in predictions[0]

    def test_get_insights(self):
        """Test insights generation"""
        analytics = PredictiveAnalytics("user_123")

        # Add some data
        for i in range(10):
            analytics.record_interaction(
                action=f"task_{i % 3}",
                context={},
                success=True
            )

        insights = analytics.get_insights()
        assert "productivity_peak_hours" in insights
        assert "established_routines" in insights
        assert "recommendations" in insights

    def test_export_import(self):
        """Test model export and import"""
        analytics = PredictiveAnalytics("user_123")

        # Add data
        analytics.record_interaction(
            action="task",
            context={},
            success=True,
            feedback_score=0.8
        )

        # Export
        exported = analytics.export_model()
        assert exported["user_id"] == "user_123"
        assert "preferences" in exported

        # Import to new instance
        new_analytics = PredictiveAnalytics("user_123")
        new_analytics.import_model(exported)

        assert new_analytics.behavior_model.get_preference_score("task") > 0


@pytest.fixture
def sample_behavior_model():
    """Fixture providing a sample behavior model"""
    model = UserBehaviorModel("test_user")

    for i in range(20):
        model.record_interaction(
            action=f"action_{i % 3}",
            context={"location": "office"},
            success=True,
            feedback_score=0.7
        )

    return model


def test_with_fixture(sample_behavior_model):
    """Test using fixture"""
    assert len(sample_behavior_model.interactions) == 20
    assert sample_behavior_model.user_id == "test_user"
