# Python Application Conventions

## Introduction

This document outlines the coding conventions and best practices for Python applications in our organization, with special consideration for AI-assisted development. Following these guidelines will ensure consistency, maintainability, and optimal collaboration between human developers and AI assistants.

## Table of Contents

- [Conventions](#conventions)
- [Project Structure](#project-structure)
- [Type Hints](#type-hints)
- [API Development with FastAPI](#api-development-with-fastapi)
- [HTTP Client: HTTPX](#http-client-httpx)
- [Environment Variables Management](#environment-variables-management)
- [Logging](#logging)
- [Scalability Considerations](#scalability-considerations)
- [Testing](#testing)

## Conventions

1. If you want to run a tool, such as test - always prefix the tool with `uv run` as project is managed by uv package manager. For example, instead of `pytest` run `uv run pytest`.

2. If you want to install package using `pip`, use `uv pip`. For example, instead of `pip install foo`, run `uv pip install foo`.

## Project Structure

1. When importing from other packages, use explicit module names:

```python
from src.auth import constants as auth_constants
from src.notifications import service as notification_service
```

2. Never add imports inside of functions or methods - only add them to the top of the file

3. Never write any comments unless requested by the user. If requested by the user, try to be as concise as possible.

## Type Hints

### General Guidelines

Always use type hints to improve code clarity, enable better IDE support, and facilitate AI code understanding and generation.

```python
# Variables
age: int = 1
names: list[str] = ["Alice", "Bob"]
user_data: dict[str, Any] = {"name": "Alice", "age": 30}

# Functions
def calculate_area(length: float, width: float) -> float:
    return length * width
```

When fixing typing errors, avoid adding type ignore hints. 

### Best Practices

1. Use `TypeAlias` for type aliases:
```python
from typing import TypeAlias

IntList: TypeAlias = list[int]
```

2. Use the appropriate collection type hints:
```python
# For Python 3.9+
values: list[int] = [1, 2, 3]
mappings: dict[str, float] = {"field": 2.0}
fixed_tuple: tuple[int, str, float] = (3, "yes", 7.5)
variable_tuple: tuple[int, ...] = (1, 2, 3)
```

3. Use `Any` when a type cannot be expressed appropriately with the current type system:
```python
from typing import Any

def process_unknown_data(data: Any) -> str:
    return str(data)
```

4. Use `object` instead of `Any` when a function accepts any possible object but doesn't need specific operations:
```python
def log_value(value: object) -> None:
    print(f"Value: {value}")
```

5. Prefer protocols and abstract types for arguments, concrete types for return values:
```python
from typing import Sequence, Iterable, Mapping

def process_items(items: Sequence[int]) -> list[str]:
    return [str(item) for item in items]
```

6. Use mypy for static type checking:
```bash
python -m pip install mypy
mypy src/

7. Never use raw dicts as parameters or return values. Always define either pydantic BaseModel for externally-facing or shared models or plain python dataclasses for internal models, such as method or function parameters.
```

### Structured Error Handling

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str
    code: str

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": "HTTP_ERROR"}
    )
```

## HTTP Client: HTTPX

Always prefer HTTPX over requests for making HTTP requests, especially for modern Python applications.

### Synchronous Usage

```python
import httpx

def fetch_data(url: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()
```


### Asynchronous Usage

```python
import httpx
import asyncio

async def fetch_data_async(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
```


### Best Practices

1. Use a single client instance for connection pooling:
```python
# Global client for reuse
http_client = httpx.Client()

# Close on application shutdown
@app.on_event("shutdown")
def shutdown_event():
    http_client.close()
```

2. Always set appropriate timeouts:
```python
client = httpx.Client(
    timeout=httpx.Timeout(5.0, connect=3.0)
)
```

3. Use structured error handling:
```python
try:
    response = client.get(url)
    response.raise_for_status()
except httpx.RequestError as exc:
    logger.error(f"Request failed: {exc}")
except httpx.HTTPStatusError as exc:
    logger.error(f"HTTP error: {exc}")
```

## Environment Variables Management

Use python-dotenv for environment variable management.

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```


### Best Practices

1. Keep `.env` out of version control:

```
# .gitignore
.env
```

2. Provide `.env.example` with dummy values:

```
# .env.example
DATABASE_URL=postgresql://user:password@localhost/dbname
DEBUG=False
SECRET_KEY=replace_with_secure_key
```

3. Validate required environment variables on startup:

```python
def validate_env_vars():
    required_vars = ["SECRET_KEY", "DATABASE_URL"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
```


## Logging

### Basic Setup

```python
import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        },
    },
    "loggers": {
        "app": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        },
    },
}

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")
```


### FastAPI Integration

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import uuid
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"Response {request_id}: {response.status_code}")
    return response
```


### Best Practices

1. Use appropriate log levels:

```python
logger.debug("Detailed information for debugging")
logger.info("Confirmation of expected events")
logger.warning("Something unexpected but the application still works")
logger.error("An error that prevents a function from working")
logger.critical("An error that prevents the application from working")
```

2. Mask sensitive information:

```python
def mask_email(email: str) -> str:
    username, domain = email.split('@')
    return f"{username[:2]}{'*' * (len(username) - 2)}@{domain}"

logger.info(f"Processing request for user: {mask_email(user.email)}")
```

3. Log structured data for easier parsing:

```python
import json

def log_event(event_type: str, data: dict) -> None:
    logger.info(f"{event_type}: {json.dumps(data)}")
```


## Scalability Considerations

### Statelessness

1. Avoid storing session state in the application memory:

```python
# BAD - In-memory state
user_sessions = {}

# GOOD - Use external session store
from fastapi_sessions.backends.redis import RedisBackend
```

2. Use external storage for shared state:

```python
import redis

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

def increment_counter(key: str) -> int:
    return redis_client.incr(key)
```

3. Design for horizontal scaling:

```python
# Configure connection pooling appropriately
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
)
```

## Testing

### Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_main.py          # Application-level tests
└── domain1/
    ├── test_api.py       # API tests
    ├── test_models.py    # Model tests
    └── test_services.py  # Service tests
```


### Example Test with Type Hints

```python
import pytest
from httpx import AsyncClient
from typing import AsyncGenerator

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    from src.main import app
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_read_main(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```


## Documentation

### Docstring Format

Use Google-style docstrings for better AI understanding:

```python
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle.
        width: The width of the rectangle.
        
    Returns:
        The area of the rectangle.
        
    Raises:
        ValueError: If length or width is negative.
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be positive")
    return length * width
```

---

By following these conventions, you'll create Python applications that are maintainable, scalable, and optimally suited for AI-assisted development.
