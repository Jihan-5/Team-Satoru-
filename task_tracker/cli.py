from __future__ import annotations

import sys
from typing import Callable, TextIO

from .exceptions import TaskNotFoundError, ValidationError
from .service import TaskService


HELP_TEXT = """\
Commands:
  add                  Add a new task (prompts for title and description)
  list                 List all tasks
  complete <id>        Mark a task as completed
  incomplete <id>      Mark a task as pending
  remove <id>          Remove a task
  help                 Show this help message
  quit                 Exit the program
"""


class TaskCLI:
    """Command-line interface for the task tracker."""

    def __init__(
        self,
        service: TaskService | None = None,
        input_fn: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> None:
        self._service = service if service is not None else TaskService()
        self._input = input_fn
        self._out: TextIO = output if output is not None else sys.stdout

    def _print(self, msg: str = "") -> None:
        print(msg, file=self._out)

    def run(self) -> None:
        self._print("Task Tracker — type 'help' for commands.")
        while True:
            try:
                raw = self._input("> ")
            except (EOFError, KeyboardInterrupt):
                self._print()
                return
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            if cmd in ("quit", "exit"):
                return
            try:
                self._dispatch(cmd, args)
            except ValidationError as e:
                self._print(f"Validation error: {e}")
            except TaskNotFoundError as e:
                self._print(f"Not found: {e}")

    def _dispatch(self, cmd: str, args: str) -> None:
        handlers = {
            "add": self._handle_add,
            "list": self._handle_list,
            "complete": self._handle_complete,
            "incomplete": self._handle_incomplete,
            "remove": self._handle_remove,
            "help": self._handle_help,
        }
        handler = handlers.get(cmd)
        if handler is None:
            self._print(f"Unknown command: {cmd!r}. Type 'help'.")
            return
        handler(args)

    def _parse_id(self, args: str) -> int:
        if not args.strip():
            raise ValidationError("Task ID is required")
        try:
            return int(args.strip())
        except ValueError as e:
            raise ValidationError(f"Invalid task ID: {args!r}") from e

    def _handle_add(self, args: str) -> None:
        title = args.strip() if args.strip() else self._input("Title: ").strip()
        description = self._input("Description: ").strip()
        task = self._service.add_task(title, description)
        self._print(f"Added task #{task.id}: {task.title}")

    def _handle_list(self, args: str) -> None:
        tasks = self._service.list_tasks()
        if not tasks:
            self._print("(no tasks)")
            return
        for t in tasks:
            self._print(repr(t))

    def _handle_complete(self, args: str) -> None:
        task = self._service.mark_complete(self._parse_id(args))
        self._print(f"Completed task #{task.id}")

    def _handle_incomplete(self, args: str) -> None:
        task = self._service.mark_incomplete(self._parse_id(args))
        self._print(f"Reopened task #{task.id}")

    def _handle_remove(self, args: str) -> None:
        task_id = self._parse_id(args)
        self._service.remove_task(task_id)
        self._print(f"Removed task #{task_id}")

    def _handle_help(self, args: str) -> None:
        self._print(HELP_TEXT)
