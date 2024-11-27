from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QHeaderView, QGroupBox, QGridLayout, QProgressBar, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QProcess
import webbrowser
import os
import pandas as pd
from qt_log_window import LogWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quant/Qual Bot Dashboard")
        self.setGeometry(100,100,1200,800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.welcome_tab = WelcomeTab()
        self.manual_train_tab = ManualTrainTab()
        self.schedule_tab = ScheduleTab()
        self.buy_sell_tab = BuySellTab()
        self.decision_tab = DecisionTab()
        self.performance_tab = PerformanceTab()
        self.report_tab = ReportTab()

        self.tabs.addTab(self.welcome_tab, "Welcome")
        self.tabs.addTab(self.manual_train_tab, "Manual Training")
        self.tabs.addTab(self.schedule_tab, "Scheduled Training")
        self.tabs.addTab(self.buy_sell_tab, "Update Buy/Sell Decisions")
        self.tabs.addTab(self.decision_tab, "View Decisions")
        self.tabs.addTab(self.performance_tab, "Performance Graph")
        self.tabs.addTab(self.report_tab, "Reports")

        # Connect tab change signal to a method
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # Check if the "View Decisions" tab is selected
        if self.tabs.tabText(index) == "View Decisions":
            self.decision_tab.load_decisions_from_csv()  # Load data only when this tab is clicked

class WelcomeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title Message
        title = QLabel("Welcome to the Qual/Quant Trading Bot Dashboard!")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Instructional Message
        instructions = QLabel(
            "Explore the tabs above to get started with the bot.\n"
            "Click on the Help tab to learn more about how to use the dashboard."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(instructions)
        
        # Button to Open GitHub
        github_button = QPushButton("Visit GitHub Repository")
        github_button.setStyleSheet("font-size: 16px; padding: 10px;")
        github_button.setFixedWidth(400)
        github_button.clicked.connect(self.open_github)
        # Center the button
        button_layout = QVBoxLayout()
        button_layout.addWidget(github_button)
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)
        
        # Set the layout
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
    
    def open_github(self):
        webbrowser.open("https://github.com/mustangThunderBird/algo-trading-bot")
        
class ManualTrainTab(QWidget):
    def __init__(self):
        super().__init__()

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Custom Title
        self.title_label = QLabel("Manual Training")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignCenter)  # Center the title label
        self.main_layout.addWidget(self.title_label)

        # Group Box for Training Buttons
        self.group_box = QGroupBox()
        self.group_box.setStyleSheet("padding: 20px;")
        self.group_box.setMinimumSize(600, 400)
        self.group_box_layout = QGridLayout()

        # Train Quantitative Model Button
        self.quant_button = QPushButton("Train Quantitative Model")
        self.quant_button.setMinimumSize(250, 60)
        self.quant_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.quant_button.clicked.connect(self.train_quant_model)
        self.group_box_layout.addWidget(self.quant_button, 0, 0)

        # Train Qualitative Model Button
        self.qual_button = QPushButton("Train Qualitative Model")
        self.qual_button.setMinimumSize(250, 60)
        self.qual_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.qual_button.clicked.connect(self.train_qual_model)
        self.group_box_layout.addWidget(self.qual_button, 0, 1)

        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 18px; color: green; padding: 10px;")
        self.group_box_layout.addWidget(self.status_label, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(25)
        self.group_box_layout.addWidget(self.progress_bar, 2, 0, 1, 2)

        self.group_box.setLayout(self.group_box_layout)
        self.main_layout.addWidget(self.group_box)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set the main layout
        self.setLayout(self.main_layout)

    def train_quant_model(self):
        self.run_training_script("python3", os.path.join(os.path.dirname(__file__), 'model', 'quantitative', 'batch_train.py'), "Quantitative Model")
    
    def train_qual_model(self):
        self.run_training_script("python3", os.path.join(os.path.dirname(__file__), 'model', 'qualitative', 'qual_model.py'), "Qualitative Model")

    def run_training_script(self, interpreter, script_path, model_name):
        self.quant_button.setEnabled(False)
        self.qual_button.setEnabled(False)

        self.status_label.setText(f"Status: Training {model_name} in progress...")
        self.status_label.setStyleSheet("font-size: 18px; color: orange;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.log_window = LogWindow()
        self.log_window.show()

        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.update_logs)
        self.process.finished.connect(lambda: self.training_complete(model_name))
        self.process.start(interpreter, [script_path])
        self.log_window.process = self.process

    def update_logs(self):
        output = self.process.readAllStandardOutput().data().decode()
        self.log_window.log_area.appendPlainText(output)
        self.progress_bar.setValue(min(self.progress_bar.value() + 10, 100))

    def training_complete(self, model_name):
        self.status_label.setText(f"Status: {model_name} training completed.")
        self.status_label.setStyleSheet("font-size: 18px; color: green;")
        self.progress_bar.setVisible(False)
        self.quant_button.setEnabled(True)
        self.qual_button.setEnabled(True)
        self.log_window.log_area.appendPlainText(f"{model_name} training completed.")


class ScheduleTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Scheduler Controls
        layout.addWidget(QLabel("Schedule Automated Training"))
        self.start_button = QPushButton("Start Scheduler")
        self.stop_button = QPushButton("Stop Scheduler")
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        self.start_button.clicked.connect(self.start_scheduler)
        self.stop_button.clicked.connect(self.stop_scheduler)
        
        self.setLayout(layout)
    
    def start_scheduler(self):
        # Placeholder: Integrate scheduler logic
        QMessageBox.information(self, "Scheduler", "Scheduler started!")
    
    def stop_scheduler(self):
        # Placeholder: Integrate scheduler stop logic
        QMessageBox.information(self, "Scheduler", "Scheduler stopped!")

class DecisionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Header Label
        layout.addWidget(QLabel("Model Buy/Sell Decisions"))

        #Label for last modified date
        self.last_modified_label = QLabel("")
        self.last_modified_label.setAlignment(Qt.AlignLeft)
        self.last_modified_label.setStyleSheet("font-size: 14px; color: gray; margin: 5px 0;")
        layout.addWidget(self.last_modified_label)

        #Table to display decisions
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.setLayout(layout)
    
    def load_decisions_from_csv(self):
        csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'buy_sell_decisions.csv')

        if os.path.exists(csv_file):
            try:
                # Update last modified date
                last_modified_time = os.path.getmtime(csv_file)
                self.last_modified_label.setText(
                    f"Last Modified: {pd.to_datetime(last_modified_time, unit='s')}"
                )
                # Read CSV
                df = pd.read_csv(csv_file, index_col=0)  # Read with index set
                # Reset the index to make 'ticker' a regular column
                df.reset_index(inplace=True)
                # Ensure 'ticker' is treated as a string and sort, then reset index again
                df["ticker"] = df["ticker"].astype(str)
                df = df.sort_values(by="ticker").reset_index(drop=True)
                # Populate table
                self.populate_table(df)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"An error occurred while reading the CSV file: {e}"
                )
        else:
            QMessageBox.warning(
                self,
                "File Not Found",
                "The buy/sell decisions CSV file was not found.\n"
                "Please generate predictions from the trained model to see the table",
            )
            self.table.setRowCount(0)
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(
                ["Ticker", "Predicted Return", "Sentiment Score", "Decision Score", "Action"]
            )
            self.last_modified_label.setText("Last Modified: File not found")
    
    def populate_table(self, df):
        """Populate the QTableWidget with data from a pandas DataFrame."""
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        # Add data to the table
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        
        # Set horizontal header to stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

class BuySellTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Make Buy Sell Decisions"))
        self.make_decisions_button = QPushButton("Update Decisions")

        layout.addWidget(self.make_decisions_button)

        self.make_decisions_button.clicked.connect(self.update_decisions)

        self.setLayout(layout)

    def update_decisions(self):
        #Placeholder: Update decisions
        QMessageBox.information(self, "Buy/Sell", "Buy/Sell Decisions Made!")

class PerformanceTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Placeholder for performance graph
        layout.addWidget(QLabel("Performance Graph"))
        self.graph_button = QPushButton("Show Graph")
        self.graph_button.clicked.connect(self.show_graph)
        layout.addWidget(self.graph_button)
        
        self.setLayout(layout)
    
    def show_graph(self):
        # Placeholder: Call graph generation logic
        QMessageBox.information(self, "Graph", "Graph generation not implemented yet!")

class ReportTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Report Download
        layout.addWidget(QLabel("Download Performance Report"))
        self.download_button = QPushButton("Download Report")
        self.download_button.clicked.connect(self.download_report)
        layout.addWidget(self.download_button)
        
        self.setLayout(layout)
    
    def download_report(self):
        # Placeholder: Implement report generation logic
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Report", "", "CSV Files (*.csv);;All Files (*)")
        if save_path:
            # Simulate saving a report
            QMessageBox.information(self, "Report", f"Report saved to {save_path}!")
        else:
            QMessageBox.warning(self, "Report", "Report saving canceled.")