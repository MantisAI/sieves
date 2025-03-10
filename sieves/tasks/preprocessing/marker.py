"""Marker task."""

from sieves.tasks.core import Task


class Marker(Task):
    """Marker task."""

    def __init__(self, task_id: str | None = None, show_progress: bool = True, include_meta: bool = False):
        super().__init__(task_id=task_id, show_progress=show_progress, include_meta=include_meta)
