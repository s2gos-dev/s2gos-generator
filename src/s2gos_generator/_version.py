def get_version() -> str:
    """Get the current version of the s2gos-generator package.

    Returns:
        Version string from package metadata
    """
    from importlib.metadata import version

    return version("s2gos-generator")
