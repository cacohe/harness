from src.application import (
    Agent,
    AgentSession,
    HarnessRunner,
    MemoryFailure,
    MemoryRuntime,
    TaskRun,
)
from src.application.agent import Agent as CanonicalAgent
from src.application.session import AgentSession as CanonicalAgentSession
from src.application.session import HarnessRunner as CanonicalHarnessRunner
from src.demo.agent import Agent as LegacyAgent
from src.demo.session import (
    AgentSession as LegacyAgentSession,
)
from src.demo.session import (
    HarnessRunner as LegacyHarnessRunner,
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
    assert LegacyHarnessRunner is HarnessRunner
    assert HarnessRunner is CanonicalHarnessRunner
    assert LegacyMemoryFailure is MemoryFailure
    assert LegacyMemoryRuntime is MemoryRuntime
    assert LegacyTaskRun is TaskRun
