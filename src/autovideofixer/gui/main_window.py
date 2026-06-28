"""Auto Video Fixer - Qt GUI application.

Provides the main window, job queue, progress display, and settings.
"""

from __future__ import annotations

import copy
import os
import sys

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autovideofixer import __version__
from autovideofixer.config import Config
from autovideofixer.core.analysis import is_video_file, scan_directory
from autovideofixer.core.pipeline import Job, JobResult, Pipeline
from autovideofixer.core.presets import get_preset, list_presets


class ProcessingThread(QThread):
    """Background thread for pipeline execution."""

    progress = Signal(str, float)  # message, progress
    job_complete = Signal(object, object)  # job, result
    finished = Signal()
    error = Signal(str)

    def __init__(self, pipeline: Pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline

    def run(self):
        try:

            def on_job(job, result):
                self.job_complete.emit(job, result)

            self.pipeline.execute_all(callback=on_job)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()
        self.pipeline = Pipeline(self.config)
        self._thread: ProcessingThread | None = None

        self.setWindowTitle(f"Auto Video Fixer v{__version__}")
        self.setMinimumSize(1000, 700)
        self._setup_ui()
        self._setup_toolbar()

    def _setup_ui(self) -> None:
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top: File selection and controls
        top = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_add_dir = QPushButton("Add Directory...")
        self.btn_add_dir.clicked.connect(self._on_add_directory)
        self.btn_clear = QPushButton("Clear Queue")
        self.btn_clear.clicked.connect(self._on_clear_queue)

        top.addWidget(self.btn_add_files)
        top.addWidget(self.btn_add_dir)
        top.addWidget(self.btn_clear)
        top.addStretch()

        # Preset selection
        top.addWidget(QLabel("Preset:"))
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(list(list_presets().keys()))
        self.combo_preset.currentTextChanged.connect(self._on_preset_changed)
        top.addWidget(self.combo_preset)

        # Output directory
        top.addWidget(QLabel("Output:"))
        self.input_output = QLineEdit()
        self.input_output.setPlaceholderText("Same as input")
        top.addWidget(self.input_output)
        top.addWidget(QPushButton("Browse...", clicked=self._on_browse_output))

        layout.addLayout(top)

        # Middle: Job list and processing
        splitter = QSplitter(Qt.Vertical)

        # Job list
        self.job_table = QtWidgets.QTableWidget()
        self.job_table.setColumnCount(6)
        self.job_table.setHorizontalHeaderLabels(
            ["File", "Status", "Progress", "Stage", "Duration", "Output"]
        )
        self.job_table.horizontalHeader().setStretchLastSection(True)
        self.job_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.job_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        splitter.addWidget(self.job_table)

        # Processing controls
        proc_widget = QWidget()
        proc_layout = QVBoxLayout(proc_widget)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        proc_layout.addLayout(btn_row)

        # Status bar in GUI
        self.gui_status = QLabel("Ready")
        proc_layout.addWidget(self.gui_status)

        splitter.addWidget(proc_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        # Bottom status bar
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")

    def _setup_toolbar(self) -> None:
        """Add menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(
            "Add &Files...",
            self._on_add_files,
            QtGui.QKeyCombination(QtGui.QKeySequence.Key_CtrlModifier, QtGui.QKeySequence.Key_O),
        )
        file_menu.addAction("Add &Directory...", self._on_add_directory)
        file_menu.addSeparator()
        file_menu.addAction("&Settings...", self._on_settings)
        file_menu.addSeparator()
        file_menu.addAction(
            "E&xit",
            self.close,
            QtGui.QKeyCombination(QtGui.QKeySequence.Key_CtrlModifier, QtGui.QKeySequence.Key_Q),
        )

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About", self._on_about)

    def _on_add_files(self) -> None:
        """Open file dialog and add selected files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv "
            "*.flv *.webm *.m4v *.mpg *.mpeg);;All Files (*)",
        )
        if files:
            for f in files:
                if is_video_file(f):
                    self._add_job_to_table(f)
            self.gui_status.setText(f"Added {len(files)} file(s)")

    def _on_add_directory(self) -> None:
        """Open directory dialog and scan for videos."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            videos = scan_directory(directory, recursive=True)
            for v in videos:
                self._add_job_to_table(v)
            self.gui_status.setText(f"Added {len(videos)} video(s) from directory")

    def _add_job_to_table(self, filepath: str) -> None:
        """Add a file as a job in the table."""
        row = self.job_table.rowCount()
        self.job_table.insertRow(row)

        job = self.pipeline.add_job(filepath)

        self.job_table.setItem(row, 0, QTableWidgetItem(os.path.basename(filepath)))
        self.job_table.setItem(row, 1, QTableWidgetItem("Queued"))
        self.job_table.setItem(row, 2, QTableWidgetItem("0%"))
        self.job_table.setItem(row, 3, QTableWidgetItem("-"))
        self.job_table.setItem(row, 4, QTableWidgetItem("-"))
        self.job_table.setItem(row, 5, QTableWidgetItem("-"))

        # Store job reference
        self.job_table.setRowData(row, job)

    def _on_start(self) -> None:
        """Start processing all queued jobs."""
        if self.pipeline.jobs:
            self._thread = ProcessingThread(self.pipeline)
            self._thread.progress.connect(self._on_progress)
            self._thread.job_complete.connect(self._on_job_complete)
            self._thread.finished.connect(self._on_processing_finished)
            self._thread.error.connect(self._on_error)
            self._thread.start()

            self.btn_start.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.gui_status.setText("Processing...")

    def _on_cancel(self) -> None:
        """Cancel processing."""
        if self._thread:
            self.pipeline.cancel()
            self.gui_status.setText("Cancelling...")

    def _on_clear_queue(self) -> None:
        """Clear all jobs from the queue."""
        self.job_table.setRowCount(0)
        self.pipeline.clear_queue()
        self.gui_status.setText("Queue cleared")

    def _on_progress(self, message: str, progress: float) -> None:
        """Update progress display."""
        self.gui_status.setText(message)

    def _on_job_complete(self, job: Job, result: JobResult) -> None:
        """Handle completed job."""
        # Update table
        for row in range(self.job_table.rowCount()):
            data = self.job_table.rowData(row)
            if data and data.get("job") is job:
                status = "Done" if result.success else "Failed"
                self.job_table.setItem(row, 1, QTableWidgetItem(status))
                self.job_table.setItem(row, 2, QTableWidgetItem("100%"))
                self.job_table.setItem(row, 4, QTableWidgetItem(f"{result.total_duration:.1f}s"))
                if result.output_path:
                    self.job_table.setItem(row, 5, QTableWidgetItem(result.output_path))
                break

    def _on_processing_finished(self) -> None:
        """Handle all processing finished."""
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.gui_status.setText("Processing complete")

    def _on_error(self, error_msg: str) -> None:
        """Handle processing error."""
        QMessageBox.critical(self, "Error", f"Processing error:\n{error_msg}")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_preset_changed(self, preset_name: str) -> None:
        """Apply preset configuration (in memory only, not persisted)."""
        preset = get_preset(preset_name)
        if preset:
            config_data = preset.to_config()
            # Work on a deep copy to avoid persisting preset values to disk
            self.config = copy.deepcopy(self.config)
            for key, value in config_data.items():
                if isinstance(value, dict):
                    self.config.set(value, key)
                else:
                    self.config.set(value, key)

    def _on_browse_output(self) -> None:
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.input_output.setText(directory)

    def _on_settings(self) -> None:
        """Open settings dialog."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            dialog.apply_settings()

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Auto Video Fixer",
            f"Auto Video Fixer v{__version__}\n\n"
            "Automated video enhancement and processing suite.\n"
            "Features AI-powered upscaling, frame interpolation,\n"
            "denoising, and intelligent pipeline optimization.\n\n"
            "Built with Python, Qt, and FFmpeg.",
        )


class SettingsDialog(QtWidgets.QDialog):
    """Settings dialog for configuration."""

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # General tab
        general = QWidget()
        gen_layout = QVBoxLayout(general)
        gen_layout.addWidget(QLabel("Output directory:"))
        self.output_dir_edit = QLineEdit(self.config.get("general", "output_dir", default=""))
        gen_layout.addWidget(self.output_dir_edit)
        gen_layout.addWidget(QLabel("Max concurrent jobs:"))
        self.threads_spin = QtWidgets.QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(self.config.get("general", "max_concurrent_jobs", default=1))
        gen_layout.addWidget(self.threads_spin)
        tabs.addTab(general, "General")

        # GPU tab
        gpu = QWidget()
        gpu_layout = QVBoxLayout(gpu)
        gpu_layout.addWidget(QLabel("Preferred GPU device:"))
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.addItems(["auto", "cuda", "metal", "cpu"])
        current = self.config.get("gpu", "preferred_device", default="auto")
        idx = self.gpu_combo.findText(current)
        if idx >= 0:
            self.gpu_combo.setCurrentIndex(idx)
        gpu_layout.addWidget(self.gpu_combo)
        tabs.addTab(gpu, "GPU")

        # Encoding tab
        encoding = QWidget()
        enc_layout = QVBoxLayout(encoding)
        enc_layout.addWidget(QLabel("Video codec:"))
        self.codec_combo = QtWidgets.QComboBox()
        self.codec_combo.addItems(["libx264", "libx265", "libvpx-vp9", "copy"])
        enc_layout.addWidget(self.codec_combo)
        enc_layout.addWidget(QLabel("CRF (lower = better quality):"))
        self.crf_spin = QtWidgets.QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        enc_layout.addWidget(self.crf_spin)
        tabs.addTab(encoding, "Encoding")

        layout.addWidget(tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def apply_settings(self) -> None:
        """Apply settings to configuration."""
        self.config.set(self.output_dir_edit.text(), "general", "output_dir")
        self.config.set(self.threads_spin.value(), "general", "max_concurrent_jobs")
        self.config.set(self.gpu_combo.currentText(), "gpu", "preferred_device")

    def exec(self) -> int:
        result = super().exec()
        if result == QtWidgets.QDialog.Accepted:
            self.apply_settings()
        return result


def run() -> None:
    """Entry point for the GUI application."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
