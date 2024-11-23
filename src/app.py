import sys
import platform
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import qt_main_window as mw

if __name__ == "__main__":
    if os.name == "posix":
        sys_name = platform.system()
        if sys_name == "Linux":
            os.system("export QT_QPA_PLATFORM=xcb")
    app = QApplication(sys.argv)
    main_window = mw.MainWindow()
    main_window.show()
    sys.exit(app.exec_())