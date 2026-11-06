"""Strip secrets from text before it is printed or put in an exception message.

Exception text from HTTP clients can carry request headers, URLs with tokens,
or the API key itself (e.g. an auth error echoing the key it was given). Every
place sembaker prints or re-raises exception text routes it through here.

Two layers:
  1. exact values of secret-looking environment variables (OPENAI_API_KEY and
     any *KEY*, *TOKEN*, *SECRET*, *PASSWORD* variable), so a key is caught
     whatever its format;
  2. patterns for common credential shapes, for secrets that never went
     through the environment.
"""

from __future__ import annotations

import os
import re

REDACTED = "[REDACTED]"

_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)
_MIN_LEN = 8          # shorter values are too likely to be ordinary text

_PATTERNS = [
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-*]{16,}"),           # OpenAI / Anthropic (incl. masked)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                              # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),                    # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),                 # Slack tokens
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),                           # Hugging Face
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-+/=]{16,}"),          # Authorization header
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)"
               r"(\s*[=:]\s*)['\"]?[^\s'\",;&]{8,}"),                 # key=value
]


def _env_secrets() -> list[str]:
    vals = {v for k, v in os.environ.items()
            if _SECRET_ENV.search(k) and v and len(v) >= _MIN_LEN}
    return sorted(vals, key=len, reverse=True)      # longest first


def redact(text) -> str:
    """Return `text` as a string with known secret values and patterns masked."""
    s = str(text)
    for v in _env_secrets():
        s = s.replace(v, REDACTED)
    for p in _PATTERNS:
        if p.groups == 2:
            s = p.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", s)
        elif p.groups == 1:
            s = p.sub(lambda m: f"{m.group(1)} {REDACTED}", s)
        else:
            s = p.sub(REDACTED, s)
    return s


def describe(exc: BaseException) -> str:
    """`ExcType: message`, redacted -- the form to print or embed in a new error."""
    return f"{type(exc).__name__}: {redact(exc)}"
