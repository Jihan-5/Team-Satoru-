class TaskTrackerError(Exception):
    """Base class for task tracker errors."""


class ValidationError(TaskTrackerError):
    """Raised when user input fails validation (e.g. empty title)."""


class TaskNotFoundError(TaskTrackerError):
    """Raised when a task ID does not exist in the repository."""
