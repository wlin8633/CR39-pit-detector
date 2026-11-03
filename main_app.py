"""Application entry point kept for existing users and scripts."""

from cr39_detector.gui.application import PitDetectionNotebookApp, main

__all__ = ["PitDetectionNotebookApp", "main"]


if __name__ == "__main__":
    main()
