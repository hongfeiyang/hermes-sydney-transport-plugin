"""Shared bounded retry helpers for TfNSW HTTP operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ....models.errors import DomainError
from .contracts import RetryPolicy

_AUTH_FAILURES = frozenset({401, 403})
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(slots=True)
class RetryableStatus(Exception):
    status: int
    retry_after: str | None


class HeaderLookup(Protocol):
    def get(self, key: str, default: str | None = None) -> str | None: ...


class ResponseStatus(Protocol):
    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> HeaderLookup: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def is_redirect(self) -> bool: ...


def retry_delay(retry_state: object, policy: RetryPolicy) -> float:
    attempt = int(getattr(retry_state, "attempt_number", 1))
    outcome = getattr(retry_state, "outcome", None)
    exception = outcome.exception() if outcome is not None else None
    if isinstance(exception, RetryableStatus) and exception.retry_after:
        retry_after = exception.retry_after
        if retry_after.replace(".", "", 1).isdecimal():
            return min(float(retry_after), policy.max_delay_seconds)
    exponential = policy.base_delay_seconds * float(2 ** (attempt - 1))
    return float(min(exponential, policy.max_delay_seconds))


def validate_status(response: ResponseStatus) -> None:
    if response.status_code in _RETRYABLE_STATUS:
        raise RetryableStatus(response.status_code, response.headers.get("Retry-After"))
    if response.status_code in _AUTH_FAILURES:
        raise DomainError(
            "authentication_failed",
            "TfNSW rejected the configured API credential.",
            http_status=response.status_code,
        )
    if response.is_error or response.is_redirect:
        raise DomainError(
            "upstream_http_error",
            f"TfNSW request failed with HTTP {response.status_code}.",
            http_status=response.status_code,
        )
