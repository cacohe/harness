from src.application import Agent, AgentSession, MemoryFailure, MemoryRuntime, TaskRun
from src.application.agent import Agent as CanonicalAgent
from src.application.session import AgentSession as CanonicalAgentSession
from src.demo.agent import Agent as LegacyAgent
from src.demo.session import (
    AgentSession as LegacyAgentSession,
)
from src.demo.session import (
    MemoryFailure as LegacyMemoryFailure,
)
from src.demo.session import (
    MemoryRuntime as LegacyMemoryRuntime,
)
from src.demo.session import (
    TaskRun as LegacyTaskRun,
)


def test_application_package_exports_the_canonical_agent_api():
    assert Agent is CanonicalAgent
    assert AgentSession is CanonicalAgentSession


def test_demo_agent_and_session_are_exact_compatibility_exports():
    assert LegacyAgent is Agent
    assert LegacyAgentSession is AgentSession
    assert LegacyMemoryFailure is MemoryFailure
    assert LegacyMemoryRuntime is MemoryRuntime
    assert LegacyTaskRun is TaskRun
