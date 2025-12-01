import os
from urllib.parse import urlparse

from openai import OpenAI

try:
    from openai import AzureOpenAI
except ImportError:  # pragma: no cover - fallback for older SDKs
    AzureOpenAI = None


def _looks_like_azure(url: str | None) -> bool:
    return bool(url and "/openai/deployments/" in url)


def _azure_endpoint(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def create_chat_client(prefix: str) -> OpenAI:
    """Return an OpenAI-compatible client for the configured deployment."""
    api_key = os.environ.get(f"{prefix}_KEY")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_version = os.environ.get(f"{prefix}_API_VERSION")

    if _looks_like_azure(base_url):
        if not api_version:
            raise RuntimeError(
                f"{prefix}_BASE_URL points to an Azure deployment but no API "
                f"version was provided. Set {prefix}_API_VERSION in chatcfd_config.json."
            )
        if AzureOpenAI is None:
            raise RuntimeError(
                "Azure endpoint detected but the installed openai package does not "
                "support AzureOpenAI. Upgrade the dependency to >=1.0.0."
            )
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=_azure_endpoint(base_url),
            api_version=api_version,
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

