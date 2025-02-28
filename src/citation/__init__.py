"""
Citation parsing and resolution package.

This package centralizes all citation parsing and resolution logic to avoid duplication.
"""

from .parser import (
    find_cluster_id,
    find_cluster_id_fuzzy,
    clean_citation_text,
    extract_citation_components,
    extract_parallel_citations,
    normalize_reporter,
    get_reporter_mappings,
    create_reporter_lookup,
    find_best_match,
    try_exact_match,
    try_fuzzy_reporter_match,
    try_volume_page_match,
    try_simplified_reporter_match,
)

__all__ = [
    "find_cluster_id",
    "find_cluster_id_fuzzy",
    "clean_citation_text",
    "extract_citation_components",
    "extract_parallel_citations",
    "normalize_reporter",
    "get_reporter_mappings",
    "create_reporter_lookup",
    "find_best_match",
    "try_exact_match",
    "try_fuzzy_reporter_match",
    "try_volume_page_match",
    "try_simplified_reporter_match",
]
