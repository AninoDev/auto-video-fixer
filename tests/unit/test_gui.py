"""Smoke tests for the Qt GUI - catches fabricated/broken Qt API usage.

Runs headless via the offscreen QPA platform so it works in CI without a display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication

from autovideofixer.config import Config
from autovideofixer.core.pipeline import JobResult


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def config(tmp_path):
    return Config(path=tmp_path / "config.yaml")


@pytest.fixture
def window(qapp, config):
    from autovideofixer.gui.main_window import MainWindow

    win = MainWindow(config=config)
    yield win
    win.close()


class TestMainWindowConstruction:
    def test_constructs_without_error(self, window):
        assert window is not None

    def test_toolbar_actions_have_valid_shortcuts(self, window):
        # Regression test: QtGui.QKeyCombination(...) does not exist on the
        # installed PySide6 and previously crashed _setup_toolbar() on every launch.
        actions = window.menuBar().findChildren(QAction)
        matching = [a for a in actions if "Add &Files" in (a.text() or "")]
        assert matching, "expected an 'Add &Files...' action on the menu bar"
        assert not matching[0].shortcut().isEmpty()


class TestJobQueue:
    def test_add_job_stores_retrievable_job_reference(self, window, tmp_path):
        # Regression test: QTableWidget.setRowData()/.rowData() do not exist on
        # PySide6 and previously crashed on every "add file" action.
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"0" * 1024)

        window._add_job_to_table(str(video_path))

        assert window.job_table.rowCount() == 1
        item = window.job_table.item(0, 0)
        job = item.data(Qt.ItemDataRole.UserRole)
        assert job is not None
        assert job.input_path == str(video_path)

    def test_job_complete_updates_matching_row(self, window, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"0" * 1024)
        window._add_job_to_table(str(video_path))

        item = window.job_table.item(0, 0)
        job = item.data(Qt.ItemDataRole.UserRole)
        result = JobResult(
            input_path=job.input_path,
            success=True,
            output_path="/tmp/out.mp4",
            total_duration=1.5,
        )

        window._on_job_complete(job, result)

        assert window.job_table.item(0, 1).text() == "Done"

    def test_queue_controls_disabled_while_processing(self, window, tmp_path, monkeypatch):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"0" * 1024)
        window._add_job_to_table(str(video_path))

        # Avoid actually spinning up ffmpeg/probing a fake file in a background
        # thread; only the synchronous button-state transition is under test.
        monkeypatch.setattr(window.pipeline, "execute_all", lambda callback=None: [])

        window._on_start()
        try:
            assert not window.btn_add_files.isEnabled()
            assert not window.btn_add_dir.isEnabled()
            assert not window.btn_clear.isEnabled()
        finally:
            if window._thread is not None:
                window._thread.wait(5000)

        window._on_processing_finished()
        assert window.btn_add_files.isEnabled()
        assert window.btn_add_dir.isEnabled()
        assert window.btn_clear.isEnabled()


class TestPresetSelection:
    def test_preset_change_mutates_config_in_place(self, window):
        # Regression test: _on_preset_changed used to reassign self.config via
        # copy.deepcopy(), desyncing it from self.pipeline.config (which holds
        # the original object by reference) - preset selection was a silent no-op.
        original_config = window.config
        assert window.pipeline.config is original_config

        window._on_preset_changed("1080p60")

        assert window.config is original_config
        assert window.pipeline.config is window.config

    def test_preset_change_merges_rather_than_replaces_stages(self, window):
        # Regression test: preset application used shallow config.set() calls,
        # which replace an entire top-level section instead of merging into it.
        window.config.set({"custom_untouched_flag": True}, "stages", "encode")

        window._on_preset_changed("1080p60")

        stages = window.config.get("stages", default={})
        assert "upscale" in stages
        assert window.config.get("stages", "encode", "custom_untouched_flag", default=None) is True

    def test_preset_change_with_unknown_name_is_a_noop(self, window):
        original_config = window.config
        window._on_preset_changed("not-a-real-preset")
        assert window.config is original_config
