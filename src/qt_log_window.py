from PyQt5.QtWidgets import QVBoxLayout, QPlainTextEdit, QDialog

class LogWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Logs")
        self.setGeometry(300, 300, 800, 600)
        
        self.layout = QVBoxLayout()
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)
        
        self.setLayout(self.layout)
        self.process = None