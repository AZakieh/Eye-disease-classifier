
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt


import eye_classifier 


PREDICTION_MAP = {
    0: "Diagnosis: Healthy",
    1: "Diagnosis: Disease Detected"
}

class EyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        #Setup the Window
        self.setWindowTitle("Eye Disease Classifier")
        self.resize(600, 400)
        
        # Load the Brain
        print("Loading AI Model...")
        self.model = eye_classifier.load_model_for_inference()
        print("Model Loaded.")

        # Setup the Layout (The Container)
        layout = QVBoxLayout()
        
        # Create Widgets
        self.label = QLabel("Please upload an eye scan.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center text
        # Make the font bigger and bold using CSS styling
        self.label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        
        button = QPushButton("Upload Image")
        button.setFixedSize(200, 50) # Give the button a consistent size
        button.clicked.connect(self.upload_image)
        
        # Add Widgets to Layout
        layout.addStretch() 
        layout.addWidget(self.label)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        #Finalize Central Widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def upload_image(self):
        """
        Handles the button click. Opens file explorer, predicts, and updates text.
        """
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Eye Scan", 
            os.getcwd(), # Start in current directory
            "Image Files (*.png *.jpg *.jpeg)"
        )

        # Only proceed if the user actually picked a file (didn't hit Cancel)
        if file_path:
            # Update text to show we are working
            self.label.setText("Analyzing...")
            self.label.repaint() # Force the GUI to update immediately
            
            # Get Prediction from your backend
            result_index = eye_classifier.predict_image(self.model, file_path)
            
            # Convert Index (0/1) to Text
            result_text = PREDICTION_MAP.get(result_index, "Unknown Error")
            
            # Update UI
            self.label.setText(result_text)
            
            # Change color based on result
            if result_index == 0:
                self.label.setStyleSheet("font-size: 18px; color: green;")
            else:
                self.label.setStyleSheet("font-size: 18px; color: red;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Sets a global style for the app (Fusion is a clean standard theme)
    app.setStyle("Fusion")
    
    window = EyeApp()
    window.show()
    
    sys.exit(app.exec())

