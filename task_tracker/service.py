from .exceptions import ValidationError
from .repository import TaskRepository
from .task import Task, TaskStatus


class TaskService:
    """Business logic and validation for tasks."""

    def __init__(self, repository: TaskRepository | None = None) -> None:
        self._repo = repository if repository is not None else TaskRepository()

    def add_task(self, title: str, description: str = "") -> Task:
        if title is None or not title.strip():
            raise ValidationError("Task title cannot be empty")
        task = Task(
            id=self._repo.next_id(),
            title=title.strip(),
            description=(description or "").strip(),
            status=TaskStatus.PENDING,
        )
        self._repo.add(task)
        return task

    def remove_task(self, task_id: int) -> None:
        self._repo.remove(task_id)

    def list_tasks(self, completed: bool | None = None) -> list[Task]:
        return self._repo.list_all(completed=completed)

    def mark_complete(self, task_id: int) -> Task:
        task = self._repo.get(task_id)
        task.mark_complete()
        return task

    def mark_incomplete(self, task_id: int) -> Task:
        task = self._repo.get(task_id)
        task.mark_incomplete()
        return task
