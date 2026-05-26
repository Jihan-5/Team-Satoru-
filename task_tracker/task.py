from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"

    def __str__(self) -> str:
        return self.value


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: TaskStatus = field(default=TaskStatus.PENDING)

    def mark_complete(self) -> None:
        self.status = TaskStatus.COMPLETED

    def mark_incomplete(self) -> None:
        self.status = TaskStatus.PENDING

    def __repr__(self) -> str:
        return f"[#{self.id}] {self.title} ({self.status}) - {self.description}"
