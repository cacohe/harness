from pydantic import BaseModel, Field
from src.schema.llm import LLMOutput
from src.schema.tool import ToolCallResult


class StepResult(BaseModel):
    """The result of a step."""
    llm_output: LLMOutput = Field(default_factory=LLMOutput)
    tool_call_result: ToolCallResult = Field(default_factory=ToolCallResult)
