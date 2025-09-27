class Sam2AvailabilityChecker:
    """
    Checks whether the SAM2 Python package is importable in the current environment.
    """

    def try_import(self) -> bool:
        """
        Attempts to import the `sam2` package and reports availability.

        Returns:
            bool: True if `sam2` can be imported, otherwise False.
        """
        try:
            import sam2  # noqa: F401
            return True
        except Exception:
            return False
