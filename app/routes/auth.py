"""Authentication endpoints."""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.models.user import LoginRequest, LoginResponse, UserMetadata
from core.models.response import APIResponse
from core.services.auth.auth_service import auth_service
from core.utils.logger import logger

router = APIRouter()
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserMetadata:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = auth_service.decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = auth_service.get_user_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth_service.get_user_metadata(user)


@router.post("/login", response_model=APIResponse)
async def login(request: LoginRequest):
    """
    Login endpoint to authenticate user and get JWT token.
    
    Args:
        request: Login credentials (username and password)
    
    Returns:
        API response with access token and user metadata
    """
    try:
        user = auth_service.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": user.username, "user_id": user.user_id}
        )
        
        user_metadata = auth_service.get_user_metadata(user)
        
        login_response = LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_metadata
        )
        
        return APIResponse(
            success=True,
            message="Login successful",
            data=login_response.dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login: {str(e)}"
        )


@router.get("/me", response_model=APIResponse)
async def get_current_user_info(current_user: UserMetadata = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user (from token)
    
    Returns:
        API response with user metadata
    """
    return APIResponse(
        success=True,
        message="User information retrieved successfully",
        data=current_user.dict()
    )


@router.get("/health")
async def auth_health():
    """Health check for auth service."""
    return {"status": "healthy", "service": "auth"}
