import httpx


def describe(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error.response.text[:200]}"
    if isinstance(error, httpx.ConnectError):
        return "az Ollama nem érhető el"
    return str(error) or error.__class__.__name__
