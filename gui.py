import eye_classifier
import sys
from PyQt6.QtWidgets import( 
QApplication, QMainWindow, QWidget, 
QVBoxLayout, QLabel, QPushButton, QFileDialog)


class EyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = eye_classifier.load_model_for_inference()
        self.setWindowTitle("Eye Disease Classifier")
        self.resize(500, 500)
        
        layout = QVBoxLayout()
        button = QPushButton("Upload Image")
        button.clicked.connect(self.upload_image)
        layout.addWidget(button)
        self.label = QLabel("No image selected")
        layout.addWidget(self.label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def upload_image(self):
        self.dialog = QFileDialog(self)
        self.dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.dialog.setNameFilter("Images (*.png *.jpg)")
        self.dialog.setViewMode(QFileDialog.ViewMode.Detail)
        self.dialog.setDirectory('C:/images/')
        if self.dialog.exec():
            fileNames = self.dialog.selectedFiles()
            file_path = fileNames[0]
            result = eye_classifier.predict_image(self.model, file_path)
            self.label.setText(f"Prediction: {result}")
            
    







app = QApplication(sys.argv)
window = EyeApp()
window.show()
sys.exit(app.exec())

