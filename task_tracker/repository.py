from .exceptions import TaskNotFoundError
from .task import Task, TaskStatus


class TaskRepository:
    """In-memory storage for tasks, keyed by integer ID."""

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1

    def next_id(self) -> int:
        task_id = self._next_id
        self._next_id += 1
        return task_id

    def add(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: int) -> Task:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task #{task_id} not found")
        return self._tasks[task_id]

    def remove(self, task_id: int) -> None:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task #{task_id} not found")
        del self._tasks[task_id]

    def list_all(self, completed: bool | None = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if completed is None:
            return tasks
        target = TaskStatus.COMPLETED if completed else TaskStatus.PENDING
        return [t for t in tasks if t.status == target]
