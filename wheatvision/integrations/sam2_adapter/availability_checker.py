class Sam2AvailabilityChecker:
    def try_import(self) -> bool:
        try:
            import sam2  # noqa: F401
            return True
        except Exception:
            return False
