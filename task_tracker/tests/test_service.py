import pytest

from task_tracker.exceptions import TaskNotFoundError, ValidationError
from task_tracker.service import TaskService
from task_tracker.task import TaskStatus


@pytest.fixture
def service():
    return TaskService()


class TestAddTask:
    def test_add_basic(self, service):
        task = service.add_task("buy milk", "2%")
        assert task.id == 1
        assert task.title == "buy milk"
        assert task.description == "2%"
        assert task.status == TaskStatus.PENDING

    def test_add_strips_whitespace(self, service):
        task = service.add_task("  hello  ", "  desc  ")
        assert task.title == "hello"
        assert task.description == "desc"

    def test_add_default_description(self, service):
        task = service.add_task("title")
        assert task.description == ""

    def test_add_none_description_is_empty(self, service):
        task = service.add_task("title", None)
        assert task.description == ""

    def test_add_ids_increment(self, service):
        a = service.add_task("a")
        b = service.add_task("b")
        assert (a.id, b.id) == (1, 2)

    def test_empty_title_raises(self, service):
        with pytest.raises(ValidationError):
            service.add_task("")

    def test_whitespace_title_raises(self, service):
        with pytest.raises(ValidationError):
            service.add_task("   ")

    def test_none_title_raises(self, service):
        with pytest.raises(ValidationError):
            service.add_task(None)


class TestListTasks:
    def test_empty(self, service):
        assert service.list_tasks() == []

    def test_multiple(self, service):
        service.add_task("a")
        service.add_task("b")
        assert [t.title for t in service.list_tasks()] == ["a", "b"]

    def test_filter_completed(self, service):
        a = service.add_task("a")
        service.add_task("b")
        service.mark_complete(a.id)
        assert [t.id for t in service.list_tasks(completed=True)] == [a.id]
        assert [t.title for t in service.list_tasks(completed=False)] == ["b"]


class TestRemoveTask:
    def test_remove(self, service):
        t = service.add_task("a")
        service.remove_task(t.id)
        assert service.list_tasks() == []

    def test_remove_missing(self, service):
        with pytest.raises(TaskNotFoundError):
            service.remove_task(1)


class TestMarkComplete:
    def test_mark_complete(self, service):
        t = service.add_task("a")
        result = service.mark_complete(t.id)
        assert result.status == TaskStatus.COMPLETED
        assert result is t

    def test_mark_complete_missing(self, service):
        with pytest.raises(TaskNotFoundError):
            service.mark_complete(99)


class TestMarkIncomplete:
    def test_mark_incomplete(self, service):
        t = service.add_task("a")
        service.mark_complete(t.id)
        result = service.mark_incomplete(t.id)
        assert result.status == TaskStatus.PENDING

    def test_mark_incomplete_missing(self, service):
        with pytest.raises(TaskNotFoundError):
            service.mark_incomplete(99)
