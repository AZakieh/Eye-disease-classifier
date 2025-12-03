import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget


class EyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Disease Classifier")
        self.resize(500, 500)


app = QApplication(sys.argv)
window = EyeApp()
window.show()
sys.exit(app.exec())

