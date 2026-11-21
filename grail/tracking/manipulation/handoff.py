"""Compatibility entry point for the manipulation tracking handoff."""
from grail.retargeting.manipulation.handoff import (
    SCHEMA,
    check_contact_map,
    check_native,
    main,
    ordered_obj,
)
from grail.tracking.manipulation.paths import sha256

__all__ = ["SCHEMA", "check_contact_map", "check_native", "main", "ordered_obj", "sha256"]


if __name__ == "__main__":
    main()
