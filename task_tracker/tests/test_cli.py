import io

import pytest

from task_tracker.cli import TaskCLI
from task_tracker.service import TaskService


def make_cli(inputs, service=None):
    it = iter(inputs)

    def fake_input(_prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    out = io.StringIO()
    cli = TaskCLI(
        service=service if service is not None else TaskService(),
        input_fn=fake_input,
        output=out,
    )
    return cli, out


class TestRunLoop:
    def test_quit_exits(self):
        cli, out = make_cli(["quit"])
        cli.run()
        assert "Task Tracker" in out.getvalue()

    def test_exit_alias(self):
        cli, out = make_cli(["exit"])
        cli.run()

    def test_eof_exits(self):
        cli, out = make_cli([])
        cli.run()

    def test_keyboard_interrupt_exits(self):
        def fake_input(_):
            raise KeyboardInterrupt

        out = io.StringIO()
        cli = TaskCLI(input_fn=fake_input, output=out)
        cli.run()

    def test_blank_line_skipped(self):
        cli, out = make_cli(["", "   ", "quit"])
        cli.run()

    def test_unknown_command(self):
        cli, out = make_cli(["nope", "quit"])
        cli.run()
        assert "Unknown command" in out.getvalue()


class TestAdd:
    def test_add_with_title_arg(self):
        cli, out = make_cli(["add buy milk", "2%", "list", "quit"])
        cli.run()
        v = out.getvalue()
        assert "Added task #1: buy milk" in v
        assert "[#1] buy milk (pending) - 2%" in v

    def test_add_prompts_for_title(self):
        cli, out = make_cli(["add", "my title", "my desc", "quit"])
        cli.run()
        assert "Added task #1: my title" in out.getvalue()

    def test_add_empty_title_validation(self):
        cli, out = make_cli(["add", "  ", "  ", "quit"])
        cli.run()
        assert "Validation error" in out.getvalue()


class TestList:
    def test_list_empty(self):
        cli, out = make_cli(["list", "quit"])
        cli.run()
        assert "(no tasks)" in out.getvalue()

    def test_list_with_tasks(self):
        svc = TaskService()
        svc.add_task("a", "d")
        cli, out = make_cli(["list", "quit"], service=svc)
        cli.run()
        assert "[#1] a" in out.getvalue()


class TestComplete:
    def test_complete(self):
        svc = TaskService()
        svc.add_task("a")
        cli, out = make_cli(["complete 1", "quit"], service=svc)
        cli.run()
        assert "Completed task #1" in out.getvalue()

    def test_complete_missing_id(self):
        cli, out = make_cli(["complete", "quit"])
        cli.run()
        assert "Validation error" in out.getvalue()

    def test_complete_bad_id(self):
        cli, out = make_cli(["complete abc", "quit"])
        cli.run()
        assert "Invalid task ID" in out.getvalue()

    def test_complete_not_found(self):
        cli, out = make_cli(["complete 99", "quit"])
        cli.run()
        assert "Not found" in out.getvalue()


class TestIncomplete:
    def test_incomplete(self):
        svc = TaskService()
        t = svc.add_task("a")
        svc.mark_complete(t.id)
        cli, out = make_cli(["incomplete 1", "quit"], service=svc)
        cli.run()
        assert "Reopened task #1" in out.getvalue()

    def test_incomplete_not_found(self):
        cli, out = make_cli(["incomplete 5", "quit"])
        cli.run()
        assert "Not found" in out.getvalue()


class TestRemove:
    def test_remove(self):
        svc = TaskService()
        svc.add_task("a")
        cli, out = make_cli(["remove 1", "quit"], service=svc)
        cli.run()
        assert "Removed task #1" in out.getvalue()

    def test_remove_not_found(self):
        cli, out = make_cli(["remove 5", "quit"])
        cli.run()
        assert "Not found" in out.getvalue()


class TestHelp:
    def test_help(self):
        cli, out = make_cli(["help", "quit"])
        cli.run()
        assert "Commands:" in out.getvalue()


class TestDefaults:
    def test_default_construction(self):
        cli = TaskCLI()
        assert cli._service is not None
        assert cli._out is not None
