"""Compatibility exports for the canonical application Agent API."""

from src.application.agent import Agent as Agent
from src.application.agent import build_agent_profile as build_agent_profile

__all__ = ["Agent", "build_agent_profile"]
