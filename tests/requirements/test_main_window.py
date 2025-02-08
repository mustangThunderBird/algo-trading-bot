import unittest
from PyQt5.QtWidgets import QApplication
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'src'))
from qt_main_window import MainWindow, WelcomeTab, ManualTrainTab, ScheduleTab, BuySellTab, DecisionTab, TradeExecutionTab, PerformanceTab, ReportTab, SettingsTab

class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.main_window = MainWindow()

    def test_initial_tabs(self):
        self.assertEqual(self.main_window.tabs.count(), 9)
        self.assertEqual(self.main_window.tabs.tabText(0), "Welcome")
        self.assertEqual(self.main_window.tabs.tabText(1), "Manual Training")
        self.assertEqual(self.main_window.tabs.tabText(2), "Scheduled Training")
        self.assertEqual(self.main_window.tabs.tabText(3), "Update Buy/Sell Decisions")
        self.assertEqual(self.main_window.tabs.tabText(4), "View Decisions")
        self.assertEqual(self.main_window.tabs.tabText(5), "Execute Trades")
        self.assertEqual(self.main_window.tabs.tabText(6), "Performance Graph")
        self.assertEqual(self.main_window.tabs.tabText(7), "Reports")
        self.assertEqual(self.main_window.tabs.tabText(8), "Settings")

    def test_welcome_tab(self):
        welcome_tab = self.main_window.welcome_tab
        self.assertIsInstance(welcome_tab, WelcomeTab)

    def test_manual_train_tab(self):
        manual_train_tab = self.main_window.manual_train_tab
        self.assertIsInstance(manual_train_tab, ManualTrainTab)

    def test_schedule_tab(self):
        schedule_tab = self.main_window.schedule_tab
        self.assertIsInstance(schedule_tab, ScheduleTab)

    def test_buy_sell_tab(self):
        buy_sell_tab = self.main_window.buy_sell_tab
        self.assertIsInstance(buy_sell_tab, BuySellTab)

    def test_decision_tab(self):
        decision_tab = self.main_window.decision_tab
        self.assertIsInstance(decision_tab, DecisionTab)

    def test_trade_execution_tab(self):
        trade_execution_tab = self.main_window.trade_execution_tab
        self.assertIsInstance(trade_execution_tab, TradeExecutionTab)

    def test_performance_tab(self):
        performance_tab = self.main_window.performance_tab
        self.assertIsInstance(performance_tab, PerformanceTab)

    def test_report_tab(self):
        report_tab = self.main_window.report_tab
        self.assertIsInstance(report_tab, ReportTab)

    def test_settings_tab(self):
        settings_tab = self.main_window.settings_tab
        self.assertIsInstance(settings_tab, SettingsTab)

if __name__ == "__main__":
    unittest.main()