"""
Paid vectorization API backends (stubs with real HTTP wiring when keys are set).

Set TRACE_BACKEND=vectorizer_ai or recraft and the corresponding API key.
"""
import logging
from typing import Any, Dict, Tuple

import requests

import config

logger = logging.getLogger(__name__)

SUPPORTED = frozenset({'vectorizer_ai', 'recraft'})


def supported_backends() -> Tuple[str, ...]:
    return tuple(sorted(SUPPORTED))


def vectorize_via_paid_backend(
    image_bytes: bytes,
    backend: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call a paid vectorization API. Returns (svg_str, meta).
    Raises ValueError if backend unknown or API key missing / request fails.
    """
    name = (backend or '').strip().lower()
    if name not in SUPPORTED:
        raise ValueError(
            f"Unknown TRACE_BACKEND '{backend}'. Supported paid backends: {', '.join(SUPPORTED)}"
        )
    if name == 'vectorizer_ai':
        return _vectorizer_ai(image_bytes)
    return _recraft(image_bytes)


def _vectorizer_ai(image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    if not config.VECTORIZER_AI_API_KEY:
        raise ValueError(
            'VECTORIZER_AI_API_KEY is not set. Add it to .env or use TRACE_BACKEND=vtracer.'
        )
    headers = {'Authorization': f'Bearer {config.VECTORIZER_AI_API_KEY}'}
    files = {'image': ('image.png', image_bytes, 'image/png')}
    try:
        response = requests.post(
            config.VECTORIZER_AI_API_URL,
            headers=headers,
            files=files,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise ValueError(f'vectorizer.ai request failed: {exc}') from exc

    if response.status_code >= 400:
        raise ValueError(
            f'vectorizer.ai returned {response.status_code}: {response.text[:500]}'
        )

    content_type = (response.headers.get('Content-Type') or '').lower()
    if 'svg' in content_type or response.text.lstrip().startswith('<'):
        svg_str = response.text.strip()
    else:
        data = response.json()
        svg_str = (data.get('svg') or data.get('data') or '').strip()
        if not svg_str:
            raise ValueError('vectorizer.ai response did not contain SVG data')

    if not svg_str.startswith('<'):
        raise ValueError('vectorizer.ai returned invalid SVG')

    return svg_str, {'backend': 'vectorizer_ai', 'api_url': config.VECTORIZER_AI_API_URL}


def _recraft(image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    if not config.RECRAFT_API_KEY:
        raise ValueError(
            'RECRAFT_API_KEY is not set. Add it to .env or use TRACE_BACKEND=vtracer.'
        )
    headers = {'Authorization': f'Bearer {config.RECRAFT_API_KEY}'}
    files = {'file': ('image.png', image_bytes, 'image/png')}
    try:
        response = requests.post(
            config.RECRAFT_VECTORIZE_URL,
            headers=headers,
            files=files,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise ValueError(f'Recraft vectorization request failed: {exc}') from exc

    if response.status_code >= 400:
        raise ValueError(
            f'Recraft API returned {response.status_code}: {response.text[:500]}'
        )

    content_type = (response.headers.get('Content-Type') or '').lower()
    if 'svg' in content_type or response.text.lstrip().startswith('<'):
        svg_str = response.text.strip()
    else:
        data = response.json()
        svg_str = (
            data.get('svg')
            or (data.get('image') or {}).get('svg')
            or ''
        ).strip()
        if not svg_str:
            raise ValueError('Recraft API response did not contain SVG data')

    if not svg_str.startswith('<'):
        raise ValueError('Recraft API returned invalid SVG')

    return svg_str, {'backend': 'recraft', 'api_url': config.RECRAFT_VECTORIZE_URL}
