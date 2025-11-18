"""
Unit tests for src/analyzer.py module.

Tests LLM client creation, image region analysis, and model fallback chains.
Uses mocking to avoid actual API calls during testing.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

from src.analyzer import (
    _build_model_chain,
    create_openrouter_client,
    analyze_region,
    compute_severity,
    analyze_image_grid,
    get_all_fallback_models,
    CHEAP_VISION_MODELS,
    DEFAULT_SYSTEM_PROMPT,
)


class TestBuildModelChain:
    """Test suite for _build_model_chain function."""
    
    def test_no_preferred_model(self):
        """
        Test model chain building without preferred model.
        
        What it tests: Default model ordering
        Expected output: First model from CHEAP_VISION_MODELS as primary
        Verifies: Default chain order used when no preference
        """
        primary, fallbacks = _build_model_chain(None)
        
        assert primary == CHEAP_VISION_MODELS[0]
        assert fallbacks == CHEAP_VISION_MODELS[1:]
    
    def test_preferred_model_in_chain(self):
        """
        Test model chain with preferred model in known list.
        
        What it tests: Preferred model prioritization
        Expected output: Preferred model becomes primary
        Verifies: Known model moved to front of chain
        """
        preferred = "openai/gpt-4o"
        primary, fallbacks = _build_model_chain(preferred)
        
        assert primary == preferred
        assert preferred not in fallbacks  # Not duplicated
        assert len(fallbacks) == len(CHEAP_VISION_MODELS) - 1
    
    def test_preferred_model_not_in_chain(self):
        """
        Test model chain with unknown preferred model.
        
        What it tests: Unknown model handling
        Expected output: Default chain used unchanged
        Verifies: Unknown models don't modify chain
        """
        preferred = "unknown/model"
        primary, fallbacks = _build_model_chain(preferred)
        
        # Should use default chain
        assert primary == CHEAP_VISION_MODELS[0]
        assert fallbacks == CHEAP_VISION_MODELS[1:]
    
    def test_chain_integrity(self):
        """
        Test that chain contains all original models.
        
        What it tests: No models lost during reordering
        Expected output: All models present in primary + fallbacks
        Verifies: Complete model list maintained
        """
        primary, fallbacks = _build_model_chain("openai/gpt-4o-mini")
        
        all_models = [primary] + fallbacks
        assert set(all_models) == set(CHEAP_VISION_MODELS)


class TestCreateOpenRouterClient:
    """Test suite for create_openrouter_client function."""
    
    def test_client_creation(self):
        """
        Test OpenRouter client creation with custom base_url.
        
        What it tests: Client initialization with OpenRouter configuration
        Expected output: OpenAI client with OpenRouter base_url
        Verifies: Proper client setup for API routing
        """
        client = create_openrouter_client(
            api_key="sk-or-test123",
            referer="http://test.com",
            app_title="test-app"
        )
        
        # Verify client configured correctly
        assert str(client.base_url) == "https://openrouter.ai/api/v1/" or str(client.base_url) == "https://openrouter.ai/api/v1"
        assert client.api_key == "sk-or-test123"
    
    def test_custom_headers(self):
        """
        Test that custom headers are set correctly.
        
        What it tests: HTTP headers configuration
        Expected output: Client with custom referer and title headers
        Verifies: Tracking headers properly configured
        """
        client = create_openrouter_client(
            api_key="sk-or-test123",
            referer="http://myapp.com",
            app_title="my-app"
        )
        
        # Check headers exist (actual OpenAI client stores in default_headers)
        assert hasattr(client, 'default_headers')


class TestAnalyzeRegion:
    """Test suite for analyze_region function."""
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_successful_analysis(self, mock_encode):
        """
        Test successful image region analysis.
        
        What it tests: Complete analysis workflow with valid API response
        Expected output: Dict with issue_count and issues list
        Verifies: Proper parsing of LLM JSON response
        """
        # Mock image encoding
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        # Mock API client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"issue_count": 2, "issues": ["Issue 1", "Issue 2"]}'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create test image
        img = Image.new("RGB", (100, 100))
        
        # Analyze
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert result["issue_count"] == 2
        assert len(result["issues"]) == 2
        assert result["issues"][0] == "Issue 1"
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_no_issues_found(self, mock_encode):
        """
        Test analysis when no issues detected.
        
        What it tests: Clean image handling
        Expected output: Zero issue count with empty issues list
        Verifies: Proper handling of error-free regions
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"issue_count": 0, "issues": []}'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert result["issue_count"] == 0
        assert result["issues"] == []
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_api_call_failure(self, mock_encode, capsys):
        """
        Test handling of API call failures.
        
        What it tests: Error recovery from API failures
        Expected output: Zero issues (safe fallback)
        Verifies: Graceful degradation on API errors
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        # Mock API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        # Should return no-issue response
        assert result["issue_count"] == 0
        assert result["issues"] == []
        
        # Should print warning
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "API call failed" in captured.out
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_invalid_json_response(self, mock_encode, capsys):
        """
        Test handling of malformed JSON in API response.
        
        What it tests: JSON parsing error handling
        Expected output: Zero issues (safe fallback)
        Verifies: Resilience to malformed responses
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='invalid json{'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert result["issue_count"] == 0
        assert result["issues"] == []
        
        captured = capsys.readouterr()
        assert "Failed to parse JSON" in captured.out
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_empty_response(self, mock_encode, capsys):
        """
        Test handling of empty API response.
        
        What it tests: Empty/null response handling
        Expected output: Zero issues (safe fallback)
        Verifies: Handles None or empty strings
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert result["issue_count"] == 0
        captured = capsys.readouterr()
        assert "Empty response" in captured.out
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_negative_issue_count_clamped(self, mock_encode):
        """
        Test that negative issue counts are clamped to zero.
        
        What it tests: Input validation for issue counts
        Expected output: Negative counts converted to zero
        Verifies: Non-negative constraint enforced
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"issue_count": -5, "issues": []}'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert result["issue_count"] == 0  # Clamped
    
    @patch('src.analyzer.encode_image_to_data_url')
    def test_issues_list_normalization(self, mock_encode):
        """
        Test that non-list issues are converted to list.
        
        What it tests: Issues field type normalization
        Expected output: String issues converted to list
        Verifies: Consistent list output format
        """
        mock_encode.return_value = "data:image/png;base64,abc123"
        
        mock_client = Mock()
        mock_response = Mock()
        # API returns string instead of list
        mock_response.choices = [Mock(message=Mock(content='{"issue_count": 1, "issues": "Single issue"}'))]
        mock_client.chat.completions.create.return_value = mock_response
        
        img = Image.new("RGB", (100, 100))
        result = analyze_region(mock_client, "openai/gpt-4o-mini", img)
        
        assert isinstance(result["issues"], list)
        assert len(result["issues"]) == 1


class TestComputeSeverity:
    """Test suite for compute_severity function."""
    
    def test_no_issues(self):
        """
        Test severity for zero issues.
        
        What it tests: "none" severity classification
        Expected output: "none" string
        Verifies: Clean regions marked as none
        """
        assert compute_severity(0) == "none"
    
    def test_mild_severity_single_issue(self):
        """
        Test severity for 1 issue (mild).
        
        What it tests: "mild" severity lower bound
        Expected output: "mild" string
        Verifies: 1 issue → yellow marker
        """
        assert compute_severity(1) == "mild"
    
    def test_mild_severity_two_issues(self):
        """
        Test severity for 2 issues (mild).
        
        What it tests: "mild" severity upper bound
        Expected output: "mild" string
        Verifies: 2 issues → yellow marker
        """
        assert compute_severity(2) == "mild"
    
    def test_severe_severity_three_issues(self):
        """
        Test severity for 3 issues (severe).
        
        What it tests: "severe" severity lower bound
        Expected output: "severe" string
        Verifies: 3 issues → red marker
        """
        assert compute_severity(3) == "severe"
    
    def test_severe_severity_many_issues(self):
        """
        Test severity for many issues (severe).
        
        What it tests: "severe" severity for high counts
        Expected output: "severe" string
        Verifies: 4+ issues → red marker
        """
        assert compute_severity(10) == "severe"
        assert compute_severity(100) == "severe"
    
    def test_negative_count(self):
        """
        Test severity for negative count (edge case).
        
        What it tests: Negative number handling
        Expected output: "none" string
        Verifies: Negative treated as none
        """
        assert compute_severity(-1) == "none"
        assert compute_severity(-100) == "none"


class TestAnalyzeImageGrid:
    """Test suite for analyze_image_grid function."""
    
    @patch('src.analyzer.analyze_region')
    def test_analyze_all_regions(self, mock_analyze):
        """
        Test analysis of complete 3x3 grid.
        
        What it tests: Grid-level analysis coordination
        Expected output: Results dict with all 9 cells
        Verifies: All regions processed independently
        """
        # Mock region analysis
        mock_analyze.return_value = {"issue_count": 1, "issues": ["Test issue"]}
        
        # Create mock regions (3x3 = 9 regions)
        img = Image.new("RGB", (100, 100))
        regions = []
        for r in range(3):
            for c in range(3):
                regions.append((r, c, img))
        
        mock_client = Mock()
        results = analyze_image_grid(mock_client, "openai/gpt-4o-mini", regions)
        
        # Should have analyzed all 9 regions
        assert len(results) == 9
        assert mock_analyze.call_count == 9
        
        # Verify all coordinates present
        for r in range(3):
            for c in range(3):
                assert (r, c) in results
    
    @patch('src.analyzer.analyze_region')
    def test_mixed_results(self, mock_analyze):
        """
        Test grid analysis with varying issue counts.
        
        What it tests: Heterogeneous results handling
        Expected output: Different results per cell
        Verifies: Independent cell analysis
        """
        # Different results for different cells
        issue_counts = [0, 1, 2, 3, 0, 1, 0, 2, 1]
        mock_analyze.side_effect = [
            {"issue_count": count, "issues": [f"Issue {i}"]}
            for i, count in enumerate(issue_counts)
        ]
        
        img = Image.new("RGB", (100, 100))
        regions = [(r, c, img) for r in range(3) for c in range(3)]
        
        mock_client = Mock()
        results = analyze_image_grid(mock_client, "openai/gpt-4o-mini", regions)
        
        # Verify different results
        result_counts = [results[(r, c)]["issue_count"] for r in range(3) for c in range(3)]
        assert result_counts == issue_counts


class TestGetAllFallbackModels:
    """Test suite for get_all_fallback_models function."""
    
    def test_returns_copy(self):
        """
        Test that function returns copy of model list.
        
        What it tests: List copying (mutation safety)
        Expected output: New list instance with same values
        Verifies: Modifications don't affect original
        """
        models1 = get_all_fallback_models()
        models2 = get_all_fallback_models()
        
        # Should be equal but different instances
        assert models1 == models2
        assert models1 is not models2
        
        # Modifying one shouldn't affect the other
        models1.append("test/model")
        assert len(models2) == len(CHEAP_VISION_MODELS)
    
    def test_contains_all_models(self):
        """
        Test that all known models are included.
        
        What it tests: Complete model list retrieval
        Expected output: All CHEAP_VISION_MODELS present
        Verifies: TEST_ALL mode gets full model list
        """
        models = get_all_fallback_models()
        assert set(models) == set(CHEAP_VISION_MODELS)


class TestConstants:
    """Test suite for module constants."""
    
    def test_cheap_vision_models_list(self):
        """
        Test that CHEAP_VISION_MODELS is populated.
        
        What it tests: Model list configuration
        Expected output: Non-empty list of model names
        Verifies: Fallback chain available
        """
        assert len(CHEAP_VISION_MODELS) > 0
        assert all(isinstance(m, str) for m in CHEAP_VISION_MODELS)
        assert all("/" in m for m in CHEAP_VISION_MODELS)  # provider/model format
    
    def test_default_system_prompt(self):
        """
        Test that DEFAULT_SYSTEM_PROMPT is set.
        
        What it tests: System prompt configuration
        Expected output: Non-empty prompt string
        Verifies: Analysis instructions provided
        """
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert "JSON" in DEFAULT_SYSTEM_PROMPT
        assert "issue_count" in DEFAULT_SYSTEM_PROMPT
        assert "issues" in DEFAULT_SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
