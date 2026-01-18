"""Authentication service for user management and JWT token handling."""
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    JWTError = Exception
    jwt = None

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    pwd_context = None

from core.models.user import User, UserMetadata
from core.utils.logger import logger

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Mock user database (in production, use a real database)
MOCK_USERS: Dict[str, User] = {
    "ahmed": User(
        user_id="U001",
        username="ahmed",
        password="password123",  # In production, hash this
        full_name="Eng. Ahmed Al-Zahrani",
        cadre="Engineering Cadre",
        current_rank="Associate Engineer (مهندس مشارك)",
        years_in_rank=4,
        administration="Project Management Office",
        expected_filter="cadre eq 'Engineering' and rank eq 'Associate'"
    ),
    "sami": User(
        user_id="U002",
        username="sami",
        password="password123",
        full_name="Sami bin Khalid",
        cadre="Wage Band (بند الأجور)",
        current_rank="Skilled Group - Category B",
        job_title="Heavy Equipment Driver (سائق شاحنة)",
        administration="General Services",
        expected_filter="cadre eq 'Wage Band' and group eq 'Skilled'"
    ),
    "hessa": User(
        user_id="U003",
        username="hessa",
        password="password123",
        full_name="Hessa Al-Mubarak",
        cadre="Civil Servant (الخدمة المدنية)",
        current_rank="Rank 11 (المرتبة الحادية عشرة)",
        administration="General Administration for Performance and Development",
        expected_filter="cadre eq 'Civil Service' and process eq 'Performance'"
    ),
    "rayan": User(
        user_id="U004",
        username="rayan",
        password="password123",
        full_name="Dr. Rayan Al-Fahad",
        cadre="Talent & Contractors (الكفاءات والمتعاقدين)",
        current_rank="Senior Specialist",
        administration="Urban Planning",
        expected_filter="cadre eq 'Talent' and contract_type eq 'Full-Time'"
    ),
    "ali": User(
        user_id="U005",
        username="ali",
        password="password123",
        full_name="Ali Al-Ghamdi",
        cadre="Users (لائحة المستخدمين)",
        current_rank="Rank 31 (مرتبة 31)",
        job_title="Office Messenger (مراسل)",
        administration="Administrative Communications",
        expected_filter="cadre eq 'Users' and process eq 'Discipline'"
    )
}


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        # For mock users, simple comparison (in production, use hashed passwords)
        return plain_password == hashed_password
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return MOCK_USERS.get(username.lower())
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by user ID."""
        for user in MOCK_USERS.values():
            if user.user_id == user_id:
                return user
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.password):
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        # Try to import at runtime in case package was installed after app start
        jwt_module = jwt
        if not JOSE_AVAILABLE or jwt_module is None:
            try:
                from jose import jwt as jwt_module_import
                jwt_module = jwt_module_import
            except ImportError:
                raise ImportError(
                    "python-jose is required for authentication. "
                    "Install it with: pip install 'python-jose[cryptography]>=3.3.0' "
                    "Then restart the application."
                )
        
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt_module.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[dict]:
        """Decode and verify a JWT token."""
        # Try to import at runtime in case package was installed after app start
        jwt_module = jwt
        jwt_error_class = JWTError
        
        if not JOSE_AVAILABLE or jwt_module is None:
            try:
                from jose import jwt as jwt_module_import, JWTError as JWTError_module
                jwt_module = jwt_module_import
                jwt_error_class = JWTError_module
            except ImportError:
                logger.error("python-jose is not installed. Authentication will not work.")
                return None
        
        try:
            payload = jwt_module.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt_error_class:
            return None
    
    def get_user_metadata(self, user: User) -> UserMetadata:
        """Get user metadata without sensitive information."""
        return UserMetadata(
            user_id=user.user_id,
            full_name=user.full_name,
            cadre=user.cadre,
            current_rank=user.current_rank,
            years_in_rank=user.years_in_rank,
            job_title=user.job_title,
            administration=user.administration,
            expected_filter=user.expected_filter
        )


# Global auth service instance
auth_service = AuthService()
