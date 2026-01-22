"""Safe tool execution with loop prevention."""
from typing import List, Dict, Any, Optional, Callable
from langchain_core.messages import BaseMessage

from core.services.tools.tool_result import ToolExecutionResult
from core.services.agents.agent_tools import create_user_info_tool
from core.models.user import UserMetadata
from core.utils.logger import logger


class ToolExecutor:
    """Execute tools safely with loop prevention and validation."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    def execute(
        self,
        llm_response: BaseMessage,
        user_metadata: Optional[UserMetadata],
        llm: Any  # AzureChatOpenAI
    ) -> ToolExecutionResult:
        """
        Execute tool calls from LLM response with safety limits.
        
        Args:
            llm_response: LLM response that may contain tool calls
            user_metadata: User metadata for tool context
            llm: LLM instance for follow-up calls
        
        Returns:
            ToolExecutionResult with execution results
        """
        if not (hasattr(llm_response, 'tool_calls') and llm_response.tool_calls): # type: ignore
            return ToolExecutionResult(
                results=[],
                tools_called=[],
                iterations=0,
                success=True
            )
        
        # Setup tools
        tools = []
        if user_metadata:
            user_info_tool = create_user_info_tool(user_metadata)
            tools.append(user_info_tool)
        
        if not tools:
            logger.warning("No tools available for execution")
            return ToolExecutionResult(
                results=[],
                tools_called=[],
                iterations=0,
                success=False,
                error="No tools available"
            )
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Execute tool calls with iteration limit
        results = []
        tools_called = []
        current_response = llm_response
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                
                # Check for tool calls
                if not (hasattr(current_response, 'tool_calls') and current_response.tool_calls): # type: ignore
                    break
                
                # Execute each tool call
                for tool_call in current_response.tool_calls: # type: ignore
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    
                    logger.info(f"[TOOL] Executing {tool_name} (iteration {iteration})")
                    
                    # Find tool
                    tool_func = None
                    for tool in tools:
                        if hasattr(tool, 'name') and tool.name == tool_name:
                            tool_func = tool
                            break
                    
                    if not tool_func:
                        logger.warning(f"Tool {tool_name} not found")
                        results.append(f"Tool {tool_name} not available")
                        continue
                    
                    # Execute tool
                    try:
                        tool_result = tool_func.invoke(tool_args)
                        results.append(f"{tool_name}: {tool_result}")
                        tools_called.append(tool_name)
                        logger.info(f"[TOOL] {tool_name} executed successfully")
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        logger.error(f"[TOOL] {error_msg}")
                        results.append(error_msg)
                
                # If we have results, get LLM response with tool results
                if results and iteration < self.max_iterations:
                    # Create follow-up message with tool results
                    from langchain_core.messages import HumanMessage
                    tool_results_text = "\n".join(results)
                    follow_up = HumanMessage(
                        content=f"Tool execution results:\n{tool_results_text}\n\nPlease provide a final answer incorporating this information."
                    )
                    
                    # Get LLM response
                    current_response = llm_with_tools.invoke([follow_up])
                else:
                    break
            
            if iteration >= self.max_iterations:
                logger.warning(f"Tool execution reached max iterations ({self.max_iterations})")
            
            return ToolExecutionResult(
                results=results,
                tools_called=tools_called,
                iterations=iteration,
                success=True
            )
        
        except Exception as e:
            logger.error(f"Error in tool execution: {str(e)}")
            return ToolExecutionResult(
                results=results,
                tools_called=tools_called,
                iterations=iteration,
                success=False,
                error=str(e)
            )
