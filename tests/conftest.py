"""Pytest configuration file."""

import os
from dotenv import load_dotenv

def pytest_configure(config):
    """Load environment variables before running tests."""
    # Load .env file from the project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)