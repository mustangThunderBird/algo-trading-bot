import sys
import platform
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import qt_main_window as mw

VERSION = "0.0.1"

if __name__ == "__main__":
    # Prevent multiple instances
    if os.getenv("APP_INSTANCE_RUNNING") == "1":
        print("Application is already running.")
        sys.exit(0)
    
    # Set environment variable
    os.environ["APP_INSTANCE_RUNNING"] = "1"

    if os.name == "posix":
        sys_name = platform.system()
        if sys_name == "Linux":
            os.system("export QT_QPA_PLATFORM=xcb")
    app = QApplication(sys.argv)
    main_window = mw.MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    # Reset the environment variable on exit (optional)
    os.environ["APP_INSTANCE_RUNNING"] = "0"