from task_tracker.task import Task, TaskStatus


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"

    def test_str(self):
        assert str(TaskStatus.PENDING) == "pending"
        assert str(TaskStatus.COMPLETED) == "completed"


class TestTask:
    def test_defaults(self):
        t = Task(id=1, title="x")
        assert t.id == 1
        assert t.title == "x"
        assert t.description == ""
        assert t.status == TaskStatus.PENDING

    def test_explicit_fields(self):
        t = Task(id=2, title="a", description="b", status=TaskStatus.COMPLETED)
        assert t.description == "b"
        assert t.status == TaskStatus.COMPLETED

    def test_mark_complete(self):
        t = Task(id=1, title="x")
        t.mark_complete()
        assert t.status == TaskStatus.COMPLETED

    def test_mark_incomplete(self):
        t = Task(id=1, title="x", status=TaskStatus.COMPLETED)
        t.mark_incomplete()
        assert t.status == TaskStatus.PENDING

    def test_mark_complete_idempotent(self):
        t = Task(id=1, title="x", status=TaskStatus.COMPLETED)
        t.mark_complete()
        assert t.status == TaskStatus.COMPLETED

    def test_mark_incomplete_idempotent(self):
        t = Task(id=1, title="x")
        t.mark_incomplete()
        assert t.status == TaskStatus.PENDING

    def test_repr(self):
        t = Task(id=7, title="buy milk", description="2%")
        assert repr(t) == "[#7] buy milk (pending) - 2%"
        t.mark_complete()
        assert repr(t) == "[#7] buy milk (completed) - 2%"
