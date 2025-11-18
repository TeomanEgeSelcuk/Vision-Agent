"""
Unit tests for src/config.py module.

Tests configuration loading, validation, and API key verification.
Uses pytest fixtures for test isolation and temporary file handling.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.config import (
    load_config,
    validate_model_name,
    validate_openrouter_api_key,
)


class TestLoadConfig:
    """Test suite for load_config function."""
    
    def test_load_valid_config(self, tmp_path):
        """
        Test loading a valid .env file with all required fields.
        
        What it tests: Configuration file parsing and validation
        Expected output: Dictionary with all config keys
        Verifies: Complete config loading with defaults applied
        """
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test123\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
            "OPENROUTER_HTTP_REFERER=http://test.com\n"
            "OPENROUTER_APP_TITLE=test-app\n"
            "TEST_ALL=true\n"
            "DATABASE_PATH=test.db\n"
        )
        
        # Load config
        config = load_config(str(env_file))
        
        # Verify all fields loaded correctly
        assert config["api_key"] == "sk-or-test123"
        assert config["model"] == "openai/gpt-4o-mini"
        assert config["referer"] == "http://test.com"
        assert config["app_title"] == "test-app"
        assert config["test_all"] is True
        assert config["db_path"] == "test.db"
    
    def test_load_config_with_defaults(self, tmp_path):
        """
        Test loading config with minimal required fields (using defaults for optional).
        
        What it tests: Default value application for optional fields
        Expected output: Config dict with defaults for missing fields
        Verifies: Optional fields get proper default values
        """
        # Create .env with only required fields
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test123\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
        )
        
        config = load_config(str(env_file))
        
        # Verify defaults applied
        assert config["api_key"] == "sk-or-test123"
        assert config["model"] == "openai/gpt-4o-mini"
        assert config["referer"] == "http://localhost"  # Default
        assert config["app_title"] == "hygo-vision-agent"  # Default
        assert config["test_all"] is False  # Default
        assert config["db_path"] == "hygo_results.db"  # Default
    
    def test_missing_env_file(self):
        """
        Test error handling when .env file doesn't exist.
        
        What it tests: FileNotFoundError when config file missing
        Expected output: FileNotFoundError exception
        Verifies: Proper error message guiding user to create .env
        """
        with pytest.raises(FileNotFoundError) as excinfo:
            load_config("nonexistent.env")
        
        assert "not found" in str(excinfo.value)
        assert ".env.example" in str(excinfo.value)
    
    def test_missing_api_key(self, tmp_path):
        """
        Test error handling when API key is missing.
        
        What it tests: RuntimeError when required OPENROUTER_API_KEY missing
        Expected output: RuntimeError exception
        Verifies: Validation catches missing API key
        """
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_MODEL=openai/gpt-4o-mini\n")
        
        with pytest.raises(RuntimeError) as excinfo:
            load_config(str(env_file))
        
        assert "OPENROUTER_API_KEY is missing" in str(excinfo.value)
    
    def test_missing_model(self, tmp_path):
        """
        Test error handling when model name is missing.
        
        What it tests: RuntimeError when required OPENROUTER_MODEL missing
        Expected output: RuntimeError exception
        Verifies: Validation catches missing model configuration
        """
        env_file = tmp_path / ".env"
        env_file.write_text("OPENROUTER_API_KEY=sk-or-test123\n")
        
        with pytest.raises(RuntimeError) as excinfo:
            load_config(str(env_file))
        
        assert "OPENROUTER_MODEL is missing" in str(excinfo.value)
    
    def test_test_all_flag_variations(self, tmp_path):
        """
        Test TEST_ALL flag parsing with various true/false values.
        
        What it tests: Boolean flag parsing from string values
        Expected output: Correct boolean conversion for various inputs
        Verifies: "true", "1", "yes" → True; others → False
        """
        env_file = tmp_path / ".env"
        
        # Test "true"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
            "TEST_ALL=true\n"
        )
        config = load_config(str(env_file))
        assert config["test_all"] is True
        
        # Test "1"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
            "TEST_ALL=1\n"
        )
        config = load_config(str(env_file))
        assert config["test_all"] is True
        
        # Test "yes"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
            "TEST_ALL=yes\n"
        )
        config = load_config(str(env_file))
        assert config["test_all"] is True
        
        # Test "false"
        env_file.write_text(
            "OPENROUTER_API_KEY=sk-or-test\n"
            "OPENROUTER_MODEL=openai/gpt-4o-mini\n"
            "TEST_ALL=false\n"
        )
        config = load_config(str(env_file))
        assert config["test_all"] is False


class TestValidateModelName:
    """Test suite for validate_model_name function."""
    
    def test_valid_model_names(self):
        """
        Test validation of correctly formatted model names.
        
        What it tests: Model name format validation (provider/model-name)
        Expected output: True for valid formats
        Verifies: Standard model name patterns are accepted
        """
        assert validate_model_name("openai/gpt-4o-mini") is True
        assert validate_model_name("google/gemini-flash-1.5") is True
        assert validate_model_name("anthropic/claude-3-haiku") is True
        assert validate_model_name("provider/model") is True
    
    def test_invalid_model_names(self):
        """
        Test rejection of incorrectly formatted model names.
        
        What it tests: Invalid model name detection
        Expected output: False for invalid formats
        Verifies: Names without "/" or too short are rejected
        """
        assert validate_model_name("gpt-4") is False  # No slash
        assert validate_model_name("a/b") is False  # Too short
        assert validate_model_name("model-name") is False  # No provider
        assert validate_model_name("") is False  # Empty
        assert validate_model_name("/") is False  # Just slash


class TestValidateOpenRouterApiKey:
    """Test suite for validate_openrouter_api_key function."""
    
    def test_invalid_key_format(self, capsys):
        """
        Test rejection of incorrectly formatted API keys.
        
        What it tests: API key format validation (must start with sk-or-)
        Expected output: False for invalid format
        Verifies: Keys not starting with "sk-or-" are rejected
        """
        result = validate_openrouter_api_key("invalid-key")
        captured = capsys.readouterr()
        
        assert result is False
        assert "API key format appears invalid" in captured.out
    
    def test_empty_key(self, capsys):
        """
        Test rejection of empty API key.
        
        What it tests: Empty string handling
        Expected output: False for empty key
        Verifies: Empty keys are caught before network call
        """
        result = validate_openrouter_api_key("")
        captured = capsys.readouterr()
        
        assert result is False
        assert "API key format appears invalid" in captured.out
    
    @patch('requests.get')
    def test_valid_key_response(self, mock_get, capsys):
        """
        Test successful API key validation with 200 response.
        
        What it tests: Valid key confirmed by OpenRouter API
        Expected output: True with success message
        Verifies: 200 response with key metadata is accepted
        """
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "name": "Test Key",
                "limit": 100
            }
        }
        mock_get.return_value = mock_response
        
        result = validate_openrouter_api_key("sk-or-test123")
        captured = capsys.readouterr()
        
        assert result is True
        assert "OpenRouter API key is valid" in captured.out
        assert "Test Key" in captured.out
    
    @patch('requests.get')
    def test_unauthorized_key(self, mock_get, capsys):
        """
        Test rejection of unauthorized API key (401 response).
        
        What it tests: Invalid key detected by OpenRouter API
        Expected output: False with error message
        Verifies: 401 response triggers rejection
        """
        # Mock unauthorized response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = validate_openrouter_api_key("sk-or-invalid")
        captured = capsys.readouterr()
        
        assert result is False
        assert "API key is invalid or unauthorized" in captured.out
    
    @patch('requests.get')
    def test_network_error(self, mock_get, capsys):
        """
        Test handling of network errors during validation.
        
        What it tests: Network failure handling (soft fail)
        Expected output: True (soft fail) with warning message
        Verifies: Network errors don't block offline development
        """
        # Mock network error
        import requests
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = validate_openrouter_api_key("sk-or-test123")
        captured = capsys.readouterr()
        
        assert result is True  # Soft fail
        assert "Network error" in captured.out
        assert "Proceeding anyway" in captured.out
    
    @patch('requests.get')
    def test_unexpected_status_code(self, mock_get, capsys):
        """
        Test handling of unexpected HTTP status codes.
        
        What it tests: Unexpected API response handling (soft fail)
        Expected output: True (soft fail) with warning
        Verifies: Unknown status codes don't block usage
        """
        # Mock unexpected response
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        mock_get.return_value = mock_response
        
        result = validate_openrouter_api_key("sk-or-test123")
        captured = capsys.readouterr()
        
        assert result is True  # Soft fail
        assert "Unexpected response" in captured.out
        assert "503" in captured.out
    
    @patch('requests.get')
    def test_json_parse_error(self, mock_get, capsys):
        """
        Test handling of malformed JSON response.
        
        What it tests: JSON parsing error handling
        Expected output: True (key still valid) with warning
        Verifies: Parse errors don't invalidate valid keys
        """
        # Mock response with JSON parse error
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        result = validate_openrouter_api_key("sk-or-test123")
        captured = capsys.readouterr()
        
        assert result is True  # Key is still valid
        assert "Could not parse key metadata" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
