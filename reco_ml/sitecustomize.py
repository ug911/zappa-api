# sitecustomize.py
# Loaded automatically by Python if present on sys.path.
# Patches old code that imports ABCs from `collections` instead of `collections.abc`.
try:
    import collections
    import collections.abc as _abc

    for _name in ("Mapping", "MutableMapping", "Sequence", "Set", "MutableSet"):
        if not hasattr(collections, _name) and hasattr(_abc, _name):
            setattr(collections, _name, getattr(_abc, _name))
except Exception:
    # Never block startup if this patch fails
    pass
