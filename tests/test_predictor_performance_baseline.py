"""
Integration tests for QualLM predictor infrastructure.

This module contains end-to-end tests, performance benchmarks, backward compatibility
validation, and regression detection for the Predictor and Prediction classes.
These tests establish baselines before metadata implementation and ensure future
changes don't break existing functionality.
"""

import time
import re
import sys
import inspect
import pandas as pd
from unittest.mock import MagicMock

from quallm.predictor import Predictor
from quallm.prediction import Prediction
from quallm.client import LLMClient
from quallm.tasks import LabelSet, SingleLabelCategorizationTask
from quallm.dataset import Dataset


# ==================== Constants ====================

DEFAULT_COLUMNS = ['timestamp', 'level', 'logger', 'func', 'line', 'message', 'raw']
DURATION_PATTERN = r'Duration: (\d+\.\d{3})s'
LOG_PATTERNS = {
    'begin_prediction': r'Index: .*\. Beginning prediction\.',
    'end_prediction': r'Index: .*\. Returning prediction\. Duration: \d+\.\d{3}s\.',
    'duration_format': DURATION_PATTERN
}


# ==================== Helper Functions ====================

def create_mock_llm(model="test-model", temperature=0.7, mode="json"):
    """Create a mock LLM with all required attributes."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.language_model = model
    mock_llm.temperature = temperature
    mock_llm.mode = mode
    mock_llm.role_args = {}
    return mock_llm


def create_mock_response(code='test', reasoning='test reasoning', confidence=90):
    """Create a standardized mock response object."""
    mock_response = MagicMock()
    mock_response.code = code
    mock_response.reasoning = reasoning
    mock_response.confidence = confidence
    return mock_response


def create_task(name='Test', values=None):
    """Create a standard SingleLabelCategorizationTask for testing."""
    values = values or ['positive', 'negative', 'neutral']
    labels = LabelSet.create(name=name, values=values)
    return SingleLabelCategorizationTask(category_class=labels)


def setup_predictor(task, dataset_items, response_code='test', n_raters=1, side_effect=None):
    """Setup predictor with dataset and mock LLM(s)."""
    dataset = Dataset(dataset_items, 'input_text')
    mock_llms = []
    
    for _ in range(n_raters):
        mock_llm = create_mock_llm()
        if side_effect:
            mock_llm.request.side_effect = side_effect
        else:
            mock_llm.request.return_value = create_mock_response(code=response_code)
        mock_llms.append(mock_llm)
    
    raters = mock_llms if n_raters > 1 else mock_llms[0]
    predictor = Predictor(task=task, raters=raters)
    return predictor, dataset, mock_llms


def verify_predictions(predictions, expected_shape):
    """Verify prediction structure and completeness."""
    assert isinstance(predictions, Prediction)
    assert predictions.shape == expected_shape
    assert predictions.n_obs == expected_shape[0]
    assert predictions.n_raters == expected_shape[1]
    return predictions


def verify_all_filled(predictions):
    """Verify all predictions are non-None."""
    n_obs, n_raters = predictions.shape
    for i in range(n_obs):
        for j in range(n_raters):
            assert predictions[i, j] is not None, f"Missing at [{i}, {j}]"


def extract_durations(predictor):
    """Extract duration values from predictor logs."""
    debug_logs = [log for log in predictor.logs 
                  if log['level'] == 'DEBUG' and 'Duration:' in log['message']]
    return [float(re.search(DURATION_PATTERN, log['message']).group(1)) 
            for log in debug_logs if re.search(DURATION_PATTERN, log['message'])]


def get_logs_df(predictor):
    """Get and verify logs DataFrame."""
    logs_df = predictor.logs_df()
    assert isinstance(logs_df, pd.DataFrame)
    for col in DEFAULT_COLUMNS:
        assert col in logs_df.columns, f"Missing column '{col}'"
    return logs_df


def create_delayed_effect(delay_ms=100, code='test', fail_indices=None):
    """Create a delayed/failing side effect for timing tests."""
    call_count = 0
    
    def effect(*args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        
        if fail_indices and current in fail_indices:
            raise ValueError(f"Simulated failure on call {current}")
        
        time.sleep(delay_ms / 1000.0)
        return create_mock_response(code=code)
    
    return effect


def assert_api_surface(obj, methods, attributes):
    """Verify object has expected methods and attributes."""
    for method in methods:
        assert hasattr(obj, method), f"Missing method: {method}"
    for attr in attributes:
        assert hasattr(obj, attr), f"Missing attribute: {attr}"


def assert_signature_params(method, expected_params):
    """Verify method signature has expected parameters."""
    sig = inspect.signature(method)
    for param in expected_params:
        assert param in sig.parameters, f"Missing parameter: {param}"


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


# ==================== Test Classes ====================

class TestEndToEndIntegration:
    """End-to-end workflow tests with various Task types."""
    
    def test_single_label_categorization_pipeline(self):
        """Test complete pipeline with SingleLabelCategorizationTask."""
        predictor, dataset, mocks = setup_predictor(
            create_task('Sentiment'),
            ['I love this!', 'It\'s okay.', 'I hate this.'],
            response_code='positive'
        )
        
        predictions = predictor.predict(dataset, echo=False)
        verify_predictions(predictions, (3, 1))
        
        assert mocks[0].request.call_count == 3
        for i in range(3):
            assert predictions[i, 0]['response'][0].code == 'positive'
    
    def test_multi_rater_pipeline(self):
        """Test complete pipeline with multiple raters."""
        predictor, dataset, mocks = setup_predictor(
            create_task('Category', values=['A', 'B', 'C']),
            ['Item 1', 'Item 2'],
            n_raters=3
        )
        
        # Set different responses per rater
        for mock, code in zip(mocks, ['A', 'B', 'C']):
            mock.request.return_value = create_mock_response(code=code)
        
        predictions = predictor.predict(dataset, echo=False)
        verify_predictions(predictions, (2, 3))
        
        # Verify each rater called correctly
        for mock in mocks:
            assert mock.request.call_count == 2
        
        # Verify responses captured
        for i in range(2):
            for j, code in enumerate(['A', 'B', 'C']):
                assert predictions[i, j]['response'][0].code == code
    
    def test_dataframe_export(self):
        """Test DataFrame export functionality."""
        predictor, dataset, _ = setup_predictor(
            create_task('DataFrameTest', values=['TypeA', 'TypeB', 'TypeC']),
            ['Text 1', 'Text 2', 'Text 3'],
            response_code='TypeA'
        )
        
        predictions = predictor.predict(dataset, echo=False)
        
        # Basic export
        df = predictions.expand()
        assert len(df) == 3 and 'code' in df.columns
        assert all(df['code'] == 'TypeA')
        
        # With external data
        external = pd.DataFrame({'id': [1, 2, 3], 'source': ['A', 'B', 'C']})
        df_with_data = predictions.expand(data=external)
        assert all(col in df_with_data.columns for col in ['id', 'source'])
    
    def test_error_handling(self):
        """Test behavior with prediction failures."""
        side_effect = create_delayed_effect(0, 'pass', fail_indices=[1])
        predictor, dataset, _ = setup_predictor(
            create_task('Test', values=['pass', 'fail']),
            ['Good', 'Bad', 'Good'],
            side_effect=side_effect
        )
        
        predictions = predictor.predict(dataset, echo=False)
        
        # Verify partial success
        assert predictions[0, 0] is not None
        # Error responses are now dicts with response=None
        pred_1 = predictions[1, 0]
        assert isinstance(pred_1, dict)
        assert pred_1.get('response') is None
        assert pred_1.get('metadata', {}).get('success') is False
        assert predictions[2, 0] is not None
        
        # Verify error logged
        errors = [log for log in predictor.logs if log['level'] == 'ERROR']
        assert any('Simulated failure' in log['message'] for log in errors)


class TestPerformanceBenchmarks:
    """Performance benchmark tests to establish timing baselines."""
    
    def test_single_prediction_timing(self):
        """Establish timing baseline for single prediction."""
        predictor, dataset, _ = setup_predictor(
            create_task('Speed', values=['fast', 'slow']),
            ['Test input'],
            side_effect=create_delayed_effect(100, 'fast')
        )
        
        _, elapsed = measure_time(predictor.predict, dataset, echo=False)
        
        # Verify timing (100ms delay + overhead)
        assert 0.1 <= elapsed <= 0.5, f"Expected 0.1-0.5s, got {elapsed:.3f}s"
        
        # Verify duration logged correctly
        durations = extract_durations(predictor)
        assert durations and 0.1 <= durations[0] <= 0.2
    
    def test_parallel_speedup(self):
        """Verify parallel processing performance gain."""
        task = create_task('Parallel', values=['yes', 'no'])
        items = [f'Item {i}' for i in range(10)]
        side_effect = create_delayed_effect(50, 'yes')
        
        # Sequential
        pred_seq, data_seq, _ = setup_predictor(task, items, side_effect=side_effect)
        _, seq_time = measure_time(pred_seq.predict, data_seq, max_workers=1, echo=False)
        
        # Parallel
        pred_par, data_par, _ = setup_predictor(task, items, side_effect=side_effect)
        preds_par, par_time = measure_time(pred_par.predict, data_par, max_workers=4, echo=False)
        
        # Verify speedup >= 2x
        speedup = seq_time / par_time
        assert speedup >= 2.0, f"Only {speedup:.2f}x speedup"
        verify_all_filled(preds_par)
    
    def test_memory_baseline(self):
        """Establish memory usage baseline."""
        predictor, dataset, _ = setup_predictor(
            create_task('Memory', values=['low', 'high']),
            [f'Text {i}' for i in range(100)],
            response_code='low'
        )
        
        initial = sys.getsizeof(predictor) + sys.getsizeof(dataset)
        predictions = predictor.predict(dataset, echo=False)
        final = initial + sys.getsizeof(predictions)
        
        increase = final - initial
        assert increase <= 100 * 1024, f"Memory increase {increase} > 100KB"
        verify_predictions(predictions, (100, 1))


class TestBackwardCompatibility:
    """Tests ensuring existing code patterns continue working."""
    
    def test_api_compatibility(self):
        """Verify existing API patterns work unchanged."""
        task = create_task('Compat', values=['old', 'new'])
        dataset = Dataset(['Test 1', 'Test 2'], 'input_text')
        
        # Single LLM (not list)
        mock = create_mock_llm()
        mock.request.return_value = create_mock_response(code='old')
        predictor = Predictor(task=task, raters=mock)
        predictions = predictor.predict(dataset, echo=False)
        
        # Test access patterns
        assert predictions[0, 0]['response'][0].code == 'old'
        assert all(predictions.get('code', flatten=True) == 'old')
        
        # DataFrame export
        df = predictions.expand()
        assert isinstance(df, pd.DataFrame) and 'code' in df.columns
        
        # Multi-rater
        pred_multi = Predictor(task=task, raters=[mock, mock])
        assert pred_multi.predict(dataset, echo=False).shape == (2, 2)
    
    def test_logging_compatibility(self):
        """Verify logging functionality unchanged."""
        predictor, dataset, _ = setup_predictor(
            create_task('Log', values=['debug', 'info']),
            ['Log test'],
            response_code='debug'
        )
        
        predictor.predict(dataset, echo=False)
        
        # Test log access
        assert isinstance(predictor.logs, list)
        get_logs_df(predictor)
        
        # Test clearing
        predictor.clear_logs()
        assert len(predictor.logs) == 0
    
    def test_resumption(self):
        """Verify prediction resumption works."""
        side_effect = create_delayed_effect(0, 'complete', fail_indices=[1])
        predictor, dataset, mocks = setup_predictor(
            create_task('Resume', values=['complete', 'partial']),
            ['Item 1', 'Item 2', 'Item 3'],
            side_effect=side_effect
        )
        
        # First run with failure
        predictions = predictor.predict(dataset, echo=False)
        # Error responses are now dicts with response=None
        pred_1 = predictions[1, 0]
        assert isinstance(pred_1, dict)
        assert pred_1.get('response') is None  # Failed item
        assert pred_1.get('metadata', {}).get('success') is False
        
        # Resume
        mocks[0].request.side_effect = None
        mocks[0].request.return_value = create_mock_response(code='complete')
        resumed = predictor.predict(dataset, predictions=predictions, echo=False)
        verify_all_filled(resumed)


class TestLogFormatValidation:
    """Tests verifying log parsing tools remain functional."""
    
    def test_duration_extraction(self):
        """Verify duration extraction from logs."""
        predictor, dataset, _ = setup_predictor(
            create_task('Duration', values=['fast', 'slow']),
            ['Speed test'],
            side_effect=create_delayed_effect(50, 'fast')
        )
        
        predictor.predict(dataset, echo=False)
        durations = extract_durations(predictor)
        
        assert durations and 0.04 <= durations[0] <= 0.1
        
        # Verify 3 decimal precision
        debug_logs = [log for log in predictor.logs if 'Duration:' in log['message']]
        for log in debug_logs:
            match = re.search(DURATION_PATTERN, log['message'])
            if match:
                assert len(match.group(1).split('.')[1]) == 3
    
    def test_logs_df_format(self):
        """Verify logs_df() column structure."""
        predictor, dataset, _ = setup_predictor(
            create_task('LogTest'),
            ['Happy', 'Sad'],
            response_code='positive'
        )
        
        predictor.predict(dataset, echo=False)
        logs_df = get_logs_df(predictor)
        
        # Verify filtering
        debug = logs_df[logs_df['level'] == 'DEBUG']
        assert len(debug) >= 4
        
        # Verify message format
        duration_logs = debug[debug['message'].str.contains('Duration:', na=False)]
        for _, row in duration_logs.iterrows():
            assert all(x in row['message'] for x in ['Index:', 'Returning prediction'])
    
    def test_error_summary(self):
        """Verify get_error_summary() functionality."""
        def conditional(*args, **kwargs):
            if 'Bad' in kwargs.get('user_prompt', ''):
                raise ConnectionError("Simulated connection error")
            return create_mock_response(code='success')
        
        predictor, dataset, _ = setup_predictor(
            create_task('Error', values=['success', 'failure']),
            ['Good', 'Bad', 'Good', 'Bad'],
            side_effect=conditional
        )
        
        predictions = predictor.predict(dataset, echo=False)
        summary = predictor.get_error_summary()
        
        assert summary['tasks_started'] == 4
        assert summary['error_categories']['connection'] == 2
        
        # Verify pattern
        assert predictions[0, 0] is not None
        # Error responses are now dicts with response=None
        pred_1 = predictions[1, 0]
        assert isinstance(pred_1, dict)
        assert pred_1.get('response') is None
        assert pred_1.get('metadata', {}).get('success') is False
        assert predictions[2, 0] is not None
        # Error response for index 3
        pred_3 = predictions[3, 0]
        assert isinstance(pred_3, dict)
        assert pred_3.get('response') is None
        assert pred_3.get('metadata', {}).get('success') is False


class TestRegressionDetection:
    """Framework for detecting performance and functionality regressions."""
    
    def test_timing_regression(self):
        """Detect if metadata additions slow predictions."""
        predictor, dataset, _ = setup_predictor(
            create_task('Perf', values=['baseline']),
            [f'Test {i}' for i in range(20)],
            side_effect=create_delayed_effect(10, 'baseline')
        )
        
        predictions, elapsed = measure_time(predictor.predict, dataset, max_workers=4, echo=False)
        per_pred = elapsed / 20
        
        assert per_pred <= 0.03, f"Performance {per_pred:.4f}s > 0.03s threshold"
        verify_all_filled(predictions)
    
    def test_memory_regression(self):
        """Detect excessive memory growth."""
        n_obs, n_raters = 50, 2
        predictor, dataset, _ = setup_predictor(
            create_task('Memory', values=['test']),
            [f'Test {i}' for i in range(n_obs)],
            response_code='test',
            n_raters=n_raters
        )
        
        predictions = predictor.predict(dataset, echo=False)
        
        total = (sys.getsizeof(predictions) + sys.getsizeof(predictor) + 
                sum(sys.getsizeof(log) for log in predictor.logs))
        per_pred = total / (n_obs * n_raters)
        
        assert per_pred <= 2048, f"Memory {per_pred:.0f} bytes > 2KB threshold"
        verify_predictions(predictions, (n_obs, n_raters))
    
    def test_api_surface(self):
        """Detect API surface changes."""
        predictor, dataset, _ = setup_predictor(
            create_task('API', values=['stable']),
            ['API test'],
            response_code='stable'
        )
        
        predictions = predictor.predict(dataset, echo=False)
        
        # Verify Predictor API
        assert_api_surface(
            predictor,
            ['predict', 'logs_df', 'clear_logs', 'get_error_summary', 'predict_single', 'set_echo_level'],
            ['raters', 'n_raters', 'task', 'logs', 'run_timestamps']
        )
        
        # Verify Prediction API
        assert_api_surface(predictions, ['get', 'expand'], ['n_obs', 'n_raters'])
        
        # Verify signatures
        assert_signature_params(predictor.predict, ['data', 'predictions', 'max_workers', 'echo'])
        assert_signature_params(predictions.expand, ['rater_labels', 'data', 'format', 'explode', 'sort_by', 'embedding_client'])
    
    def test_log_format_stability(self):
        """Detect log format changes."""
        predictor, dataset, _ = setup_predictor(
            create_task('LogStability', values=['stable']),
            ['Log test'],
            side_effect=create_delayed_effect(10, 'stable')
        )
        
        predictor.predict(dataset, echo=False)
        debug_logs = [log for log in predictor.logs if log['level'] == 'DEBUG']
        
        for log in debug_logs:
            msg = log['message']
            if 'Beginning prediction' in msg:
                assert re.match(LOG_PATTERNS['begin_prediction'], msg)
            elif 'Returning prediction' in msg:
                assert re.match(LOG_PATTERNS['end_prediction'], msg)
                assert re.search(LOG_PATTERNS['duration_format'], msg)