"""fetch_page tool: download a web page and return its readable text.

Used for job postings and company pages. Safety limits:
- http/https only, 15 s timeout, 2 MB download cap, 6,000 characters returned
  (a small model's context window can't hold a whole page anyway).
- Refuses private, loopback, and link-local addresses. This matters once the
  API is deployed: without it, a prompt could make the server fetch internal
  URLs such as the cloud metadata endpoint (an SSRF attack).
- GET only. No tool in AgentLite can submit forms, so the agents cannot
  submit a job application even if a model tried to.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

MAX_CHARS = 6000
MAX_BYTES = 2_000_000
TIMEOUT_S = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AgentLite/0.1; +https://github.com/)"}


def is_public_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Only public http(s) hosts are allowed."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False, "only http:// and https:// URLs are allowed"
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        return False, f"could not resolve host '{parsed.hostname}'"
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False, f"'{parsed.hostname}' points to a private or local address"
    return True, ""


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line))
    return f"{title}\n\n{body}" if title else body


@tool(parse_docstring=True)
def fetch_page(url: str) -> str:
    """Download a web page (for example a job posting or a company's About page) and return its text.

    Args:
        url: The full URL starting with http:// or https://.
    """
    url = url.strip()
    ok, reason = is_public_url(url)
    if not ok:
        return f"ERROR: cannot fetch {url}: {reason}."

    try:
        with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True, headers=HEADERS) as client:
            with client.stream("GET", url) as response:
                # Redirects could lead to a private address; check the final URL too.
                ok, reason = is_public_url(str(response.url))
                if not ok:
                    return f"ERROR: redirected to a blocked address: {reason}."
                if response.status_code >= 400:
                    return f"ERROR: {url} returned HTTP {response.status_code}."
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_BYTES:
                        break
                content_type = response.headers.get("content-type", "")
                encoding = response.encoding or "utf-8"
    except httpx.HTTPError as exc:
        return f"ERROR: failed to fetch {url}: {exc}"

    raw = b"".join(chunks).decode(encoding, errors="replace")
    if "html" in content_type or raw.lstrip().startswith("<"):
        text = html_to_text(raw)
    elif content_type.startswith("text/") or "json" in content_type:
        text = raw
    else:
        return f"ERROR: {url} is not a text page (content-type: {content_type})."

    if len(text.strip()) < 50:
        return f"ERROR: {url} returned almost no text. The page may need JavaScript or a login; ask the user to paste the text."
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n...[truncated, page has {len(text)} characters]"
    return text
