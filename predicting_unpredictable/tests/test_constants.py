"""Tests for predicting_unpredictable.constants module."""

from predicting_unpredictable.constants import (
    EVENT_TYPES,
    NORM_STATS,
    TASK2_NORM_PARAMS,
    __all__ as constants_all,
)


class TestNormStats:
    """Validate NORM_STATS structure/values"""

    EXPECTED_CHANNELS = ["vis", "ir069", "ir107", "vil"]
    ZSCORE_CHANNELS = ["vis", "ir069", "ir107"]
    LOG_CHANNELS = ["vil"]
    COMMON_KEYS = ["target_min", "target_max"]

    def test_norm_stats_keys(self):
        """Test NORM_STATS contains all expected channels."""
        assert set(NORM_STATS.keys()) == set(self.EXPECTED_CHANNELS)

    def test_norm_stats_common_keys(self):
        """Test all channels have valid target min/max (0.0/1.0, float)."""
        for channel in self.EXPECTED_CHANNELS:
            stats = NORM_STATS[channel]
            for key in self.COMMON_KEYS:
                assert key in stats, f"{channel} missing {key}"
                assert isinstance(stats[key], float), \
                    f"{channel}.{key} not float"
                assert stats[key] in (0.0, 1.0), \
                    f"{channel}.{key} not 0.0/1.0"

    def test_norm_stats_vil_log(self):
        """Test VIL channel has valid log normalization stats."""
        vil_stats = NORM_STATS["vil"]
        log_keys = ["data_min", "data_max", "eps", "log_scale"]
        for key in log_keys:
            assert key in vil_stats, f"vil missing {key}"

        # EDA update: Use actual max (211.0) instead of legacy 255.0
        assert vil_stats["data_min"] == 0.0
        assert vil_stats["data_max"] == 211.0
        assert vil_stats["eps"] == 1e-6
        assert isinstance(vil_stats["log_scale"], bool)
        assert vil_stats["log_scale"] is True

        assert "mean" not in vil_stats
        assert "std" not in vil_stats


class TestTask2NormParams:
    """Validate TASK2_NORM_PARAMS (Task 2 baseline normalization)."""

    EXPECTED_CHANNELS = ["ir069", "ir107", "vil"]

    def test_task2_norm_params_keys(self):
        """Test TASK2_NORM_PARAMS contains all expected channels."""
        assert set(TASK2_NORM_PARAMS.keys()) == set(self.EXPECTED_CHANNELS)

    def test_task2_norm_params_ir_channels(self):
        """Test IR channels have valid offset/scale (numeric)."""
        ir_channels = ["ir069", "ir107"]
        for channel in ir_channels:
            params = TASK2_NORM_PARAMS[channel]
            assert "offset" in params, f"{channel} missing offset"
            assert "scale" in params, f"{channel} missing scale"
            assert isinstance(params["offset"], (int, float)), \
                f"{channel}.offset not numeric"
            assert isinstance(params["scale"], (int, float)), \
                f"{channel}.scale not numeric"

        # EDA update: Use EDA-derived offset/scale values
        assert float(TASK2_NORM_PARAMS["ir069"]["offset"]) == 7500.0
        assert float(TASK2_NORM_PARAMS["ir069"]["scale"]) == 6243.0
        assert float(TASK2_NORM_PARAMS["ir107"]["offset"]) == 7422.0
        assert float(TASK2_NORM_PARAMS["ir107"]["scale"]) == 9853.0

    def test_task2_norm_params_vil(self):
        """Test VIL channel has valid scale (no offset, numeric)."""
        vil_params = TASK2_NORM_PARAMS["vil"]
        assert "scale" in vil_params
        assert "offset" not in vil_params
        assert isinstance(vil_params["scale"], (int, float)), \
            "vil.scale not numeric"
        # EDA update: Use actual max (211.0) instead of legacy 255.0
        assert float(vil_params["scale"]) == 211.0


class TestEventTypes:
    """Validate EVENT_TYPES (Task 3 event classification classes)."""

    EXPECTED_EVENT_TYPES = [
        "Flash Flood",
        "Flood",
        "Funnel Cloud",
        "Hail",
        "Heavy Rain",
        "Lightning",
        "Thunderstorm Wind",
        "Tornado",
    ]

    def test_event_types_list(self):
        """Test EVENT_TYPES is a list of strings with expected values."""
        assert isinstance(EVENT_TYPES, list)
        assert len(EVENT_TYPES) == len(self.EXPECTED_EVENT_TYPES)

        for event in EVENT_TYPES:
            assert isinstance(event, str)
            assert len(event) > 0, "Empty event type string"

        assert EVENT_TYPES == self.EXPECTED_EVENT_TYPES
        assert len(set(EVENT_TYPES)) == len(EVENT_TYPES)

    def test_event_types_specific_values(self):
        """Test critical event types exist in EVENT_TYPES."""
        critical_events = ["Tornado", "Hail", "Lightning"]
        for event in critical_events:
            assert event in EVENT_TYPES, f"{event} missing from EVENT_TYPES"


class TestExports:
    """Validate __all__ export list."""

    def test_all_export(self):
        """Test __all__ contains expected public constants."""
        expected_exports = ["EVENT_TYPES", "NORM_STATS", "TASK2_NORM_PARAMS"]
        assert set(constants_all) == set(expected_exports)
        assert len(constants_all) == len(expected_exports)
