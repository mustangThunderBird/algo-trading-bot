from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QHeaderView, QGroupBox, QGridLayout, QProgressBar, QSpacerItem, QSizePolicy,
    QLineEdit, QFormLayout, QApplication, QComboBox
)
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import webbrowser
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scheduler import Scheduler
import trade_execution 
from model.model_manager import ModelManager
from model.quantitative import batch_train
from model.qualitative import qual_model
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
import traceback
from fpdf import FPDF
from cryptography.fernet import Fernet
from openvino.runtime import Core
from app import VERSION


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quant/Qual Bot Dashboard")
        self.setGeometry(100,100,1200,800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.scheduler = Scheduler()

        # Create tabs
        self.welcome_tab = WelcomeTab()
        self.manual_train_tab = ManualTrainTab()
        self.schedule_tab = ScheduleTab(self.scheduler)
        self.buy_sell_tab = BuySellTab()
        self.decision_tab = DecisionTab()
        self.trade_execution_tab = TradeExecutionTab()
        self.performance_tab = PerformanceTab()
        self.report_tab = ReportTab()
        self.settings_tab = SettingsTab()

        self.tabs.addTab(self.welcome_tab, "Welcome")
        self.tabs.addTab(self.manual_train_tab, "Manual Training")
        self.tabs.addTab(self.schedule_tab, "Scheduled Training")
        self.tabs.addTab(self.buy_sell_tab, "Update Buy/Sell Decisions")
        self.tabs.addTab(self.decision_tab, "View Decisions")
        self.tabs.addTab(self.trade_execution_tab, "Execute Trades")
        self.tabs.addTab(self.performance_tab, "Performance Graph")
        self.tabs.addTab(self.report_tab, "Reports")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Connect tab change signal to a method
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # Check if the "View Decisions" tab is selected
        if self.tabs.tabText(index) == "View Decisions":
            self.decision_tab.load_decisions_from_csv()  # Load data only when this tab is clicked
        elif self.tabs.tabText(index) == "Performance Graph":
            self.performance_tab.plot_graph()

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
            "Click on the GitHub Wiki button to learn more about how to use the dashboard.\n"
            "This app requires an alpaca trade API account in order to work. If you do not have an alpaca\n"
            "account, you can visit the sign up page by clicking on the Alpaca button below"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(instructions)
        
        # Button to Open GitHub
        github_button = QPushButton("Visit GitHub Wiki")
        github_button.setStyleSheet("font-size: 16px; padding: 10px;")
        github_button.setFixedWidth(400)
        github_button.clicked.connect(self.open_github)
        # Center the button
        button_layout = QVBoxLayout()
        button_layout.addWidget(github_button)
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)

        # Button to Open GitHub
        alpaca_button = QPushButton("Visit Alpaca")
        alpaca_button.setStyleSheet("font-size: 16px; padding: 10px;")
        alpaca_button.setFixedWidth(400)
        alpaca_button.clicked.connect(self.open_alpaca)
        # Center the button
        button1_layout = QVBoxLayout()
        button1_layout.addWidget(alpaca_button)
        button1_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button1_layout)

        verison = QLabel(f"\n\nApp Version: {VERSION}")
        verison.setAlignment(Qt.AlignCenter)
        layout.addWidget(verison)
        
        # Set the layout
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)
    
    def open_github(self):
        webbrowser.open("https://github.com/mustangThunderBird/algo-trading-bot/wiki")
    
    def open_alpaca(self):
        webbrowser.open("https://alpaca.markets/")

class TrainingWorker(QThread):
    progress = pyqtSignal(int)  # Signal for progress updates
    finished = pyqtSignal(str)  # Signal when training is finished
    error = pyqtSignal(str)  # Signal for errors

    def __init__(self, train_func, model_name, parent=None):
        super().__init__(parent)
        self.train_func = train_func
        self.model_name = model_name
        self._stop_requested = False

    def run(self):
        try:
            self.train_func(progress_callback=lambda progress: self.progress.emit(int(progress)))
            self.finished.emit(f"{self.model_name} Training Completed.")
        except Exception as e:
            self.error.emit(f"Error in {self.model_name} Training: {e}")
    
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True

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

        self.worker = None

    def train_quant_model(self):
        self._start_training(batch_train.train_models, "Quantitative Model")

    def train_qual_model(self):
        self._start_training(qual_model.determine_sentiments, "Qualitative Model")

    def _start_training(self, train_func, model_name):
        # Disable buttons and show progress
        self.quant_button.setEnabled(False)
        self.qual_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Status: {model_name} Training in Progress...")
        self.status_label.setStyleSheet("font-size: 18px; color: orange;")

        # Create and start the worker thread
        self.worker = TrainingWorker(train_func, model_name)
        self.worker.progress.connect(self._update_progress) 
        self.worker.finished.connect(self._training_complete)
        self.worker.error.connect(self._training_error)
        self.worker.start()

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _training_complete(self, message):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("font-size: 18px; color: green;")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.quant_button.setEnabled(True)
        self.qual_button.setEnabled(True)

    def _training_error(self, error_message):
        self.status_label.setText(error_message)
        self.status_label.setStyleSheet("font-size: 18px; color: red;")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.quant_button.setEnabled(True)
        self.qual_button.setEnabled(True)

class SchedulerThread(QThread):
    scheduler_started = pyqtSignal()
    scheduler_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self._stop_requested = False  # Use internal flag for stopping

    def run(self):
        try:
            self._stop_requested = False
            self.scheduler_started.emit()
            self.scheduler.start()
            while not self._stop_requested:
                # Ensure scheduler loop handles periodic tasks
                QThread.msleep(100)  # Prevent high CPU usage
        except Exception as e:
            error_message = f"Error in scheduler: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_message)
        finally:
            self.scheduler.stop()  # Ensure proper cleanup
            self.scheduler_stopped.emit()

    def stop(self):
        self._stop_requested = True  # Signal the thread to stop

class ScheduleTab(QWidget):
    def __init__(self, scheduler):
        super().__init__()

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("Schedule Automated Tasks")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        self.group_box = QGroupBox()
        self.group_box.setStyleSheet("padding: 20px;")
        self.group_box.setMinimumSize(600, 400)
        self.group_box_layout = QGridLayout()

        self.start_button = QPushButton("Start Scheduler")
        self.start_button.setMinimumSize(250, 60)
        self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.start_button.clicked.connect(self.start_scheduler)
        self.group_box_layout.addWidget(self.start_button, 0, 0)

        self.stop_button = QPushButton("Stop Scheduler")
        self.stop_button.setMinimumSize(250, 60)
        self.stop_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.stop_button.clicked.connect(self.stop_scheduler)
        self.stop_button.setEnabled(False)
        self.group_box_layout.addWidget(self.stop_button, 0, 1)

        self.status_label = QLabel("Status: Scheduler is stopped")
        self.status_label.setStyleSheet("font-size: 18px; color: red; padding: 10px;")
        self.group_box_layout.addWidget(self.status_label, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        self.group_box.setLayout(self.group_box_layout)
        self.main_layout.addWidget(self.group_box)

        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.setLayout(self.main_layout)

        self.scheduler = scheduler 
        self.scheduler_thread = None

    def start_scheduler(self):
        if self.scheduler_thread and self.scheduler_thread.isRunning():
            return  # Prevent multiple threads

        self.scheduler_thread = SchedulerThread(self.scheduler)
        self.scheduler_thread.scheduler_started.connect(self.on_scheduler_started)
        self.scheduler_thread.scheduler_stopped.connect(self.on_scheduler_stopped)
        self.scheduler_thread.error_occurred.connect(self.on_scheduler_error)

        self.scheduler_thread.start()

    def stop_scheduler(self):
        if self.scheduler_thread and self.scheduler_thread.isRunning():
            self.scheduler_thread.stop()
            self.scheduler_thread.wait(5000)  # Wait with timeout (5s)
            if self.scheduler_thread.isRunning():
                QMessageBox.warning(self, "Warning", "Failed to stop scheduler in time.")
        else:
            QMessageBox.warning(self, "Warning", "Scheduler is not running.")

    def on_scheduler_started(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Scheduler is running...")
        self.status_label.setStyleSheet("font-size: 18px; color: orange;")

    def on_scheduler_stopped(self):
        QApplication.processEvents() 
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Scheduler is stopped")
        self.status_label.setStyleSheet("font-size: 18px; color: red;")

    def on_scheduler_error(self, error_message):
        QMessageBox.critical(self, "Scheduler Error", error_message)
        self.on_scheduler_stopped()

    def update_status_indicator(self, color):
        """
        Updates the LED-like status indicator.
        """
        pixmap = QPixmap(30, 30)
        pixmap.fill(QColor(color))
        self.status_indicator.setPixmap(pixmap)

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
    def __init__(self, ):
        super().__init__()

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Custom Title
        self.title_label = QLabel("Update Buy/Sell Decisions")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Group Box for Update Decisions Button
        self.group_box = QGroupBox()
        self.group_box.setStyleSheet("padding: 20px;")
        self.group_box.setMinimumSize(600, 300)
        self.group_box_layout = QVBoxLayout()

        # Update Decisions Button
        self.update_button = QPushButton("Update Decisions")
        self.update_button.setMinimumSize(250, 60)
        self.update_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.update_button.clicked.connect(self.update_decisions)
        self.group_box_layout.addWidget(self.update_button, alignment=Qt.AlignCenter)

        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 18px; color: green; padding: 10px;")
        self.group_box_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        self.group_box.setLayout(self.group_box_layout)
        self.main_layout.addWidget(self.group_box)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set the main layout
        self.setLayout(self.main_layout)

    def update_decisions(self):
        # Paths for required files and directories
        sentiment_file = os.path.join(os.path.dirname(__file__), 'model', 'qualitative', 'sentiment_scores.csv')
        model_dir = os.path.join(os.path.dirname(__file__), 'model', 'quantitative', 'models')

        # Check if sentiment scores file exists
        if not os.path.exists(sentiment_file):
            QMessageBox.critical(self, "Error", "Sentiment scores CSV file is missing. Please train the qualitative model first.")
            return

        # Check if the models directory exists and is not empty
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            QMessageBox.critical(self, "Error", "Quantitative models directory is missing or empty. Please train the quantitative model first.")
            return

        self.status_label.setText("Updating Buy/Sell Decisions...")
        try:
            # Ensure logs directory exists
            LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            os.makedirs(LOG_DIR, exist_ok=True)

            # Initialize the model manager and generate decisions
            model_manager = ModelManager(sentiment_file, model_dir)
            decisions_file = os.path.join(LOG_DIR, 'buy_sell_decisions.csv')
            model_manager.make_decisions(decisions_file)

            self.status_label.setText("Decisions Updated Successfully.")
        except Exception as e:
            print(f"Error in updating decisions: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred while updating decisions: {e}")
            self.status_label.setText("Error in Updating Decisions.")

    def decisions_complete(self):
        # Re-enable the button and update status
        self.status_label.setText("Status: Decisions updated successfully!")
        self.status_label.setStyleSheet("font-size: 18px; color: green;")
        self.progress_bar.setVisible(False)
        self.update_button.setEnabled(True)

class TradeExecutionTab(QWidget):
    def __init__(self):
        super().__init__()

        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Custom Title
        self.title_label = QLabel("Execute Trades")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Group Box for Execute Trades Button
        self.group_box = QGroupBox()
        self.group_box.setStyleSheet("padding: 20px;")
        self.group_box.setMinimumSize(600, 300)
        self.group_box_layout = QVBoxLayout()

        # Execute Trades Button
        self.execute_button = QPushButton("Execute Trades")
        self.execute_button.setMinimumSize(250, 60)
        self.execute_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.execute_button.clicked.connect(self.execute_trades)
        self.group_box_layout.addWidget(self.execute_button, alignment=Qt.AlignCenter)

        # Status Label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 18px; color: green; padding: 10px;")
        self.group_box_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(25)
        self.group_box_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        self.group_box.setLayout(self.group_box_layout)
        self.main_layout.addWidget(self.group_box)

        # Add vertical spacers to balance layout
        self.main_layout.addSpacerItem(QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set the main layout
        self.setLayout(self.main_layout)

    def execute_trades(self):
        self.status_label.setText("Executing Trades...")
        try:
            status = trade_execution.execute_trades()
            if status == 0: 
                QMessageBox.information(self, "Success", "Trades Executed Successfully.")
                self.status_label.setStyleSheet("font-size: 18px; color: green;")
                self.status_label.setText("Trades Executed Successfully.")
            elif status == -1:
                QMessageBox.critical(self, "Error", "Failed to load API credentials.")
                self.status_label.setStyleSheet("font-size: 18px; color: red;")
                self.status_label.setText("Failed to load API credentials.")
            elif status == -2:
                QMessageBox.critical(self, "Error", "API Key or Secret missing.")
                self.status_label.setStyleSheet("font-size: 18px; color: red;")
                self.status_label.setText("API Key or Secret missing.")
            elif status == -3:
                QMessageBox.critical(self, "Error", "Buy/Sell decisions file not found.")
                self.status_label.setStyleSheet("font-size: 18px; color: red;")
                self.status_label.setText("Buy/Sell decisions file not found.")
        except Exception as e:
            print(f"Error in trade execution: {e}")
            self.status_label.setText("Error in Trade Execution.")

class DataFetchThread(QThread):
    data_fetched = pyqtSignal(pd.DataFrame)

    def __init__(self, positions, parent=None):
        super().__init__(parent)
        self.positions = positions

    def run(self):
        try:
            data = {}
            for _, row in self.positions.iterrows():
                ticker = row['Ticker']
                quantity = row['Quantity']
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo", interval="1d")  # Daily data
                if hist.empty:
                    continue
                data[ticker] = hist['Close'] * quantity
            
            portfolio_data = pd.DataFrame(data)
            portfolio_data['Portfolio Value'] = portfolio_data.sum(axis=1)

            # Resample to reduce resolution
            portfolio_data = portfolio_data.resample('W').mean()

            self.data_fetched.emit(portfolio_data)
        except Exception as e:
            self.data_fetched.emit(None)

class PerformanceTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Title
        title = QLabel("Portfolio Performance - Last 1 Month")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(title)

        # Canvas for the graph
        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.ax = self.canvas.figure.add_subplot(111)  # Create an axis for later use
        self.layout.addWidget(self.canvas)

        # Button to refresh graph
        self.refresh_button = QPushButton("Refresh Graph")
        self.refresh_button.clicked.connect(self.plot_graph)
        self.layout.addWidget(self.refresh_button)

        # Set layout
        self.setLayout(self.layout)

    def fetch_positions(self):
        """Fetch current positions from Alpaca API."""
        settings_tab = self.parentWidget().findChild(SettingsTab)
        credentials = settings_tab.load_credentials()
        if not credentials:
            raise ValueError("Failed to load API credentials")
        
        API_KEY = credentials.get("api_key")
        API_SECRET = credentials.get("api_secret")
        if not API_KEY or not API_SECRET:
            raise ValueError("Missing API Key or Secret in credentials.")

        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        positions = trading_client.get_all_positions()

        # Parse positions into a DataFrame
        data = {
            "Ticker": [pos.symbol for pos in positions],
            "Quantity": [float(pos.qty) for pos in positions]
        }
        return pd.DataFrame(data)

    def plot_graph(self):
        """Initiate data fetch and plot graph."""
        try:
            # Display loading message
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'LOADING...', fontsize=24, ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()

            # Fetch positions and start background thread
            positions = self.fetch_positions()
            self.thread = DataFetchThread(positions)
            self.thread.data_fetched.connect(self.on_data_fetched)
            self.thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not fetch portfolio data: {e}")

    def on_data_fetched(self, portfolio_data):
        """Handle data fetched by the thread."""
        if portfolio_data is None:
            QMessageBox.critical(self, "Error", "Failed to fetch portfolio data.")
            return

        self.ax.clear()  # Clear loading text
        self.ax.plot(portfolio_data.index, portfolio_data['Portfolio Value'], label="Portfolio Value", linewidth=2)
        self.ax.set_title("Portfolio Performance - Last 1 Month")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Portfolio Value")
        self.ax.legend(loc="upper left")
        self.ax.grid(True)
        self.canvas.draw()

class ReportTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Download Performance Report")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Button to generate the report
        self.download_button = QPushButton("Download Report")
        self.download_button.clicked.connect(self.generate_report)
        layout.addWidget(self.download_button)

        self.setLayout(layout)

    def fetch_portfolio_data(self):
        """Fetch current positions from Alpaca API."""
        settings_tab = self.parentWidget().findChild(SettingsTab)
        credentials = settings_tab.load_credentials()
        if not credentials:
            raise ValueError("Failed to load API credentials")
        
        API_KEY = credentials.get("api_key")
        API_SECRET = credentials.get("api_secret")
        if not API_KEY or not API_SECRET:
            raise ValueError("Missing API Key or Secret in credentials.")

        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        positions = trading_client.get_all_positions()

        # Parse positions into a DataFrame
        data = {
            "Ticker": [pos.symbol for pos in positions],
            "Quantity": [float(pos.qty) for pos in positions]
        }
        portfolio_df = pd.DataFrame(data)
        portfolio_df["Price"] = portfolio_df["Ticker"].apply(lambda ticker: yf.Ticker(ticker).info.get("regularMarketPrice", 0))
        return portfolio_df

    def fetch_sp500_data(self):
        """Fetch historical S&P 500 data."""
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period="1mo")
        return hist["Close"]

    def generate_report(self):
        # Set up a loading message
        self.loading_label = QLabel("Loading report, please wait...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 18px; font-weight: bold; color: orange;")
        self.layout().addWidget(self.loading_label)

        # Ensure the GUI updates immediately
        QApplication.processEvents()

        try:
            # Fetch portfolio data
            portfolio_df = self.fetch_portfolio_data()

            # Compute the percentage allocation based on quantities
            portfolio_df["Percentage"] = portfolio_df["Quantity"] / portfolio_df["Quantity"].sum() * 100

            # Filter out rows where 'Percentage' is NaN or zero
            valid_portfolio = portfolio_df[portfolio_df["Percentage"] > 0]

            if valid_portfolio.empty:
                QMessageBox.warning(self, "Error", "No valid data available for asset allocation.")
                self.layout().removeWidget(self.loading_label)
                self.loading_label.deleteLater()
                return

            # Pie chart: Asset allocation
            pie_fig, pie_ax = plt.subplots(figsize=(6, 4))
            valid_portfolio.set_index("Ticker")["Percentage"].plot.pie(
                ax=pie_ax, autopct="%1.1f%%", startangle=90
            )
            pie_ax.set_title("Asset Allocation")
            pie_ax.set_ylabel("")

            # Fetch performance data for S&P 500
            sp500_data = self.fetch_sp500_data()
            sp500_data.index = pd.to_datetime(sp500_data.index)

            # Fetch historical data for portfolio stocks
            tickers = portfolio_df["Ticker"].tolist()
            quantities = portfolio_df.set_index("Ticker")["Quantity"].to_dict()

            portfolio_values = pd.DataFrame(index=sp500_data.index)
            for ticker in tickers:
                try:
                    stock_data = yf.Ticker(ticker).history(start=sp500_data.index.min(), end=sp500_data.index.max())
                    stock_data = stock_data.reindex(sp500_data.index, method="ffill")  # Align with S&P 500 index
                    portfolio_values[ticker] = stock_data["Close"] * quantities[ticker]
                except Exception as e:
                    print(f"Failed to fetch data for {ticker}: {e}")

            # Calculate total portfolio value over time
            portfolio_values["Total"] = portfolio_values.sum(axis=1)

            # Calculate returns for portfolio and S&P 500
            portfolio_returns = portfolio_values["Total"].pct_change().dropna() * 100
            sp500_returns = sp500_data.pct_change().dropna() * 100

            # Line chart: Portfolio returns vs. S&P 500 returns
            perf_fig, perf_ax = plt.subplots(figsize=(6, 4))
            sp500_returns.plot(ax=perf_ax, label="S&P 500 Returns", color="blue")
            portfolio_returns.plot(ax=perf_ax, label="Portfolio Returns", color="orange")
            perf_ax.set_title("Returns Comparison")
            perf_ax.set_xlabel("Date")
            perf_ax.set_ylabel("Returns (%)")
            perf_ax.legend()

            # Save charts as temporary files
            pie_path = "pie_chart.png"
            perf_path = "returns_chart.png"
            pie_fig.savefig(pie_path)
            perf_fig.savefig(perf_path)
            plt.close(pie_fig)
            plt.close(perf_fig)

            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Performance Report", ln=True, align="C")

            # Add pie chart
            pdf.cell(200, 10, txt="Asset Allocation", ln=True, align="C")
            pdf.image(pie_path, x=10, y=30, w=190)

            # Add returns graph
            pdf.add_page()
            pdf.cell(200, 10, txt="Returns Comparison", ln=True, align="C")
            pdf.image(perf_path, x=10, y=30, w=190)

            # Save PDF
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf);;All Files (*)")
            if save_path:
                pdf.output(save_path)
                QMessageBox.information(self, "Report Saved", f"Report saved to {save_path}.")
            else:
                QMessageBox.warning(self, "Save Canceled", "Report saving was canceled.")

            # Clean up temporary files
            os.remove(pie_path)
            os.remove(perf_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while generating the report: {e}")
        finally:
            # Remove the loading message
            self.layout().removeWidget(self.loading_label)
            self.loading_label.deleteLater()

class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', "config.json")
        self.encryption_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'credentials', "encryption_key.key")

        # Title
        self.title_label = QLabel("Settings")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Form for API Key and Secret
        self.form_layout = QFormLayout()

        credentials = self.load_credentials()
        if not credentials:
            #if path does not exist leave the fields with placeholder text
            # API Key Input
            self.api_key_input = QLineEdit()
            self.api_key_input.setPlaceholderText("Enter your Alpaca API Key")
            self.form_layout.addRow("API Key:", self.api_key_input)
            # API Secret Input
            self.api_secret_input = QLineEdit()
            self.api_secret_input.setPlaceholderText("Enter your Alpaca API Secret")
            self.api_secret_input.setEchoMode(QLineEdit.Password)
            self.form_layout.addRow("API Secret:", self.api_secret_input)
        else:
            # If path exists, pre-fill fields with the decrypted text
            self.api_key_input = QLineEdit()
            self.api_key_input.setText(credentials["api_key"])
            self.form_layout.addRow("API Key:", self.api_key_input)

            self.api_secret_input = QLineEdit()
            self.api_secret_input.setText(credentials["api_secret"])
            self.api_secret_input.setEchoMode(QLineEdit.Password)
            self.form_layout.addRow("API Secret:", self.api_secret_input)

        self.layout.addLayout(self.form_layout)

        #Dropdown for Device Selection
        self.device_combo = QComboBox()
        self.detect_devices()
        #self.device_combo.addItems(self.detect_devices())
        self.device_combo.setCurrentText(self.load_device_preference())
        self.form_layout.addRow("Training Device:", self.device_combo)

        # Save Button
        self.save_button = QPushButton("Save Settings")
        self.save_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button, alignment=Qt.AlignCenter)

        # Status Label
        self.status_label = QLabel("Status: Waiting for input")
        self.status_label.setStyleSheet("font-size: 16px; color: gray;")
        self.layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)

    def detect_devices(self):
        """return list of available devices"""
        devices = ["CPU"]
        try:
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            for device in local_devices:
                if device.device_type == "GPU":
                    devices.append(f"GPU ({device.name})")
        except Exception as e:
            print(f"Error detecting NVIDIA devices: {e}")
        
        try:
            core = Core()
            available_devices = core.available_devices
            for device in available_devices:
                if "GPU" in device.upper():
                    devices.append(f"GPU ({device})")
        except Exception as e:
            print(f"Error detecting Intel devices: {e}")

        self.device_combo.addItems(devices)
    
    def load_device_preference(self):
        """load device preference from file"""
        if not os.path.exists(self.config_path):
            return "CPU"
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            return config.get("training_device", "CPU")
        except Exception as e:
            print(f"Error loading device preference: {e}")
            return "CPU"
    
    def load_credentials(self):
        if not os.path.exists(self.config_path) or not os.path.exists(self.encryption_key_path):
            return False
        try:
            # Load the encryption key
            with open(self.encryption_key_path, "rb") as key_file:
                encryption_key = key_file.read()

            cipher = Fernet(encryption_key)

            # Load and decrypt credentials
            with open(self.config_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = cipher.decrypt(encrypted_data).decode()
            return json.loads(decrypted_data)
        except Exception as e:
            self.status_label.setText(f"Status: Error loading credentials - {str(e)}")
            self.status_label.setStyleSheet("font-size: 16px; color: red;")
            return False
        
    def save_settings(self):
        api_key = self.api_key_input.text()
        api_secret = self.api_secret_input.text()
        device = self.device_combo.currentText()

        if not api_key or not api_secret:
            self.status_label.setText("Status: Please fill in both fields.")
            self.status_label.setStyleSheet("font-size: 16px; color: red;")
            return

        try:
            # Save device preference
            config = {"training_device": device}
            with open(self.config_path, "w") as f:
                json.dump(config, f)
            # Encrypt and save credentials securely
            encrypted_data = self.encrypt_credentials(api_key, api_secret)
            with open(self.config_path, "wb") as f:
                f.write(encrypted_data)
            self.status_label.setText("Status: Credentials saved successfully!")
            self.status_label.setStyleSheet("font-size: 16px; color: green;")
        except Exception as e:
            self.status_label.setText(f"Status: Error saving credentials - {str(e)}")
            self.status_label.setStyleSheet("font-size: 16px; color: red;")

    def encrypt_credentials(self, api_key, api_secret):
        # Generate a key for encryption
        encryption_key = Fernet.generate_key()
        cipher = Fernet(encryption_key)

        # Save the key securely
        with open(self.encryption_key_path, "wb") as key_file:
            key_file.write(encryption_key)

        # Encrypt the credentials
        credentials = json.dumps({"api_key": api_key, "api_secret": api_secret}).encode()
        return cipher.encrypt(credentials)
