import pytest

from task_tracker.exceptions import TaskNotFoundError
from task_tracker.repository import TaskRepository
from task_tracker.task import Task


@pytest.fixture
def repo():
    return TaskRepository()


class TestTaskRepository:
    def test_empty_list(self, repo):
        assert repo.list_all() == []

    def test_next_id_increments(self, repo):
        assert repo.next_id() == 1
        assert repo.next_id() == 2
        assert repo.next_id() == 3

    def test_add_and_get(self, repo):
        task = Task(id=1, title="x")
        repo.add(task)
        assert repo.get(1) is task

    def test_get_missing_raises(self, repo):
        with pytest.raises(TaskNotFoundError, match="#42"):
            repo.get(42)

    def test_remove(self, repo):
        repo.add(Task(id=1, title="x"))
        repo.remove(1)
        assert repo.list_all() == []

    def test_remove_missing_raises(self, repo):
        with pytest.raises(TaskNotFoundError, match="#99"):
            repo.remove(99)

    def test_list_all_returns_copy(self, repo):
        repo.add(Task(id=1, title="x"))
        snapshot = repo.list_all()
        snapshot.clear()
        assert len(repo.list_all()) == 1

    def test_add_overwrites_same_id(self, repo):
        repo.add(Task(id=1, title="a"))
        repo.add(Task(id=1, title="b"))
        assert repo.get(1).title == "b"

    def test_list_all_filter_completed(self, repo):
        a = Task(id=1, title="a")
        b = Task(id=2, title="b")
        b.mark_complete()
        repo.add(a)
        repo.add(b)
        assert [t.id for t in repo.list_all(completed=True)] == [2]
        assert [t.id for t in repo.list_all(completed=False)] == [1]
        assert len(repo.list_all(completed=None)) == 2
        assert len(repo.list_all()) == 2

    def test_list_all_filter_empty_results(self, repo):
        repo.add(Task(id=1, title="a"))
        assert repo.list_all(completed=True) == []

    def test_next_id_independent_of_add(self, repo):
        id1 = repo.next_id()
        repo.add(Task(id=id1, title="a"))
        id2 = repo.next_id()
        assert id2 == 2
