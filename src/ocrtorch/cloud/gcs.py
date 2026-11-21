"""Google Cloud Storage URI utilities."""

from __future__ import annotations


def parse_gcs(uri: str) -> tuple[str, str]:
    """Parse a GCS URI into (bucket, prefix)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    return bucket, prefix.strip("/")


def require_gcs_prefix(uri: str, option: str) -> tuple[str, str]:
    """Validate that a GCS URI contains a bucket and a non-empty prefix."""
    bucket, prefix = parse_gcs(uri)
    if not prefix:
        raise ValueError(f"{option} must include a GCS object prefix, for example gs://{bucket}/path")
    return bucket, prefix
