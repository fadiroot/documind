"""Agent tools for LangChain."""
from langchain_core.tools import tool

from core.models.user import UserMetadata


def create_user_info_tool(user_metadata: UserMetadata):
    """
    Create a LangChain tool for retrieving user information.
    
    Args:
        user_metadata: User metadata to access when tool is called
    
    Returns:
        LangChain tool function
    """
    @tool
    def get_user_info() -> str:
        """
        Retrieve the current user's personal information.
        
        Use this tool when questions ask about personal information such as:
        - Salary, rank, position, cadre, benefits
        - Questions requiring personal context for accurate answers
        
        Returns:
            Formatted string with user information or error message
        """
        if not user_metadata:
            return "No user information available."
        
        user = user_metadata
        info_parts = [
            f"User Information:",
            f"- Full Name: {user.full_name}",
            f"- Cadre: {user.cadre}",
            f"- Current Rank: {user.current_rank or 'N/A'}",
            f"- Years in Rank: {user.years_in_rank or 'N/A'}",
            f"- Administration: {user.administration}"
        ]
        
        if user.job_title:
            info_parts.append(f"- Job Title: {user.job_title}")
        
        if user.expected_filter:
            info_parts.append(f"- Expected Filter: {user.expected_filter}")
        
        return "\n".join(info_parts)
    
    return get_user_info
