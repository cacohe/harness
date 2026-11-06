from enum import Enum


class HarnessState(Enum):
    HARNESS_STATE_STARTING = "starting"
    HARNESS_STATE_GUARDING = "guarding"
    HARNESS_STATE_PLANNING = "planning"
    HARNESS_STATE_INFERING = "inferring"
    HARNESS_STATE_TOOL_CALLING = "tool_calling"
    HARNESS_STATE_RUNNING = "running"
    HARNESS_STATE_FINISHED = "finished"
