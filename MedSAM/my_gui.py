import sys
import numpy as np

try:
    from PyQt4 import QtCore
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    
from PIL import Image

try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QGuiApplication
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
import torch
from transformers import SamModel, SamProcessor

import random

from enum import Enum
from qtpy import QtGui


   
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MedSAM Segmentation tool")
        
        self.saved = False
        
        # ============================= MENUS ==================================
        # Exit Action to go into the "File Menu"
        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit Application")
        exitAction.triggered.connect(self.close_application)

        # Open FIle Action to go into the "File Menu"
        openFile = QAction("&Open File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip("Open File")
        openFile.triggered.connect(self.load_image)

        # Save File Action to go into the "File Menu"
        saveFile = QAction("&Save Segmentation", self)
        saveFile.setShortcut("Ctrl+S")
        saveFile.setStatusTip("Save Segmentation")
        saveFile.triggered.connect(self.file_save)
        
        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(exitAction)
        
        # =============================== TOOLBARS =============================
        # Make a toolbar for open/save and load
        toolbarOpenAction = QAction(QIcon("icons/open.png"), 'Open File', self)
        toolbarOpenAction.triggered.connect(self.load_image)

        toolbarSaveAction = QAction(QIcon("icons/save_2.png"), 'Save Annotation', self)
        toolbarSaveAction.triggered.connect(self.file_save)

        toolbarExitAction = QAction(QIcon("icons/exit_1.png"), 'Exit Application', self)
        toolbarExitAction.triggered.connect(self.close_application)
        
        
        self.fileToolbar = self.addToolBar("Horizontal Toolbar / Most common actions")
        self.fileToolbar.setMovable(False)
        self.fileToolbar.addAction(toolbarOpenAction)
        self.fileToolbar.addAction(toolbarSaveAction)
        self.fileToolbar.addAction(toolbarExitAction)
        ################################################# my code
        central_widget = QWidget(self)
        self.main_layout = QVBoxLayout(central_widget)
        
        self.main_layout.addWidget(self.fileToolbar)
        
        self.image_widget = DrawRectangle()
        self.imageArea = DrawingArea(self, main_window=self)
        
        
        self.extract_rect_button = QPushButton("segmentation", self)
        self.extract_rect_button.setMinimumSize(100, 50)
        self.extract_rect_button.setMaximumSize(200, 100)
        self.extract_rect_button.clicked.connect(self.do_segmentation)
        
        self.main_layout.addWidget(self.extract_rect_button)
        
        self.central_layout = QHBoxLayout()
        
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        frame.setFixedSize(512, 512)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid gray; 
                border-radius: 5px; 
                background-color: light gray; 
            }
        """)

        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.image_widget, alignment=Qt.AlignCenter, stretch=1)
        self.central_layout.addWidget(frame)
        
        
        self.frame_segmentation = QFrame()
        self.frame_segmentation.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.frame_segmentation.setLineWidth(2)
        self.frame_segmentation.setFixedSize(512, 512)
        self.frame_segmentation.setStyleSheet("""
            QFrame {
        border: 2px solid gray;
        border-radius: 5px;
    }
""")

        frame_layout2 = QVBoxLayout(self.frame_segmentation)
        frame_layout2.addWidget(self.imageArea)
        self.central_layout.addWidget(self.frame_segmentation)
        
        self.group = QButtonGroup()
        self.group.setExclusive(True) 
        self.group.buttonClicked.connect(self.check_buttons)

        self.eraser_radio_button = QRadioButton("Eraser", self)
        self.group.addButton(self.eraser_radio_button)
        
        
        self.brush_radio_button = QRadioButton("Brush", self)
        self.group.addButton(self.brush_radio_button)
        
        
        self.grid = QGridLayout()
        self.box = ToolBox()
        
        spacer_widget = QWidget()
        spacer_widget.setFixedHeight(5) 
        
        self.info_label = QLabel('Select an organ name from the provided list')
        self.info_label.setStyleSheet("background-color: yellow; padding: 2px; font-size: 15px; font-weight: bold;")
        self.info_label.setWordWrap(True)
        self.info_label.setVisible(False)
        self.box.vbox.addWidget(self.info_label)
        self.box.vbox.addWidget(spacer_widget)
        
        # list for organs
        self.organsList()
        
        # clear mask button
        self.clear_mask_button = QPushButton("Clear Mask", self)
        self.clear_mask_button.setMinimumSize(80, 50)
        self.clear_mask_button.setMaximumSize(200, 100)
        self.clear_mask_button.clicked.connect(self.clear_mask)
        self.box.vbox.addWidget(self.clear_mask_button)
        self.box.vbox.addWidget(spacer_widget)
        
        # undo button
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.setMinimumSize(80, 50)
        self.undo_button.setMaximumSize(200, 100)
        self.undo_button.setShortcut("Ctrl+Z")
        self.undo_button.clicked.connect(self.imageArea.undo)
        self.box.vbox.addWidget(self.undo_button)
        self.box.vbox.addWidget(spacer_widget)
        
        self.setBrushSlider()
        self.setBrushStyle()
        self.setBrushCap()
        self.setColorChanger()
        self.setEraserSlider()
        
        
        self.grid.addWidget(self.box, 0, 0, 1, 1)
        
        win = QWidget()
        win.setLayout(self.grid)
        
        self.central_layout.addWidget(win)

        self.setCentralWidget(central_widget)
        
        # for rectangle
        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None
        self.input_boxes = None
        
        self.image_width = None
        self.image_height = None
        
        # MedSAM
        self.model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")


        self.main_layout.addLayout(self.central_layout)
        

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg)", options=options)
        if file_name:
            self.image_widget.load_image(file_name)
            self.frame_segmentation.setStyleSheet(f"""
            QFrame {{
                border: 2px solid black;
                background-image: url({file_name});
                background-repeat: no-repeat;
            }}
        """)
        self.clear_mask()
        self.imageArea.mask_image_labels = np.zeros((512, 512), dtype=np.uint8)

    def close_application(self):
        if not self.saved:
            exitChoice = QMessageBox.question(self, "Exit", "There is an unsaved segmentation. Are you sure you want to exit?",
                                                    QMessageBox.Yes | QMessageBox.No)
            if exitChoice == QMessageBox.No:
                return

        self.close()
        
        
    def file_save(self):
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            mask_image = Image.fromarray(self.imageArea.mask_image_labels)
            mask_image.save(file_path, "PNG")
        self.saved = True
    
            
    def get_normalized_rect_coordinates(self):
        if self.image_widget.rect_start and self.image_widget.rect_end:
            rect = self.image_widget.get_rect()
            self.image_width = self.image_widget.image.width()
            self.image_height = self.image_widget.image.height()

            # based on image dimensions
            start_x = rect.topLeft().x() / self.image_width
            start_y = rect.topLeft().y() / self.image_height
            end_x = rect.bottomRight().x() / self.image_width
            end_y = rect.bottomRight().y() / self.image_height

            # to ensure [0, 1] range
            start_x = max(0, min(1, start_x))
            start_y = max(0, min(1, start_y))
            end_x = max(0, min(1, end_x))
            end_y = max(0, min(1, end_y))

            self.rect_coordinates = (start_x, start_y, end_x, end_y)
            return self.rect_coordinates
        else:
            return None
        
    def do_segmentation(self):
        
        if self.image_widget.image is not None:
            self.start_x, self.start_y, self.end_x, self.end_y = self.get_normalized_rect_coordinates()
            
            if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
                
                min_x = min(self.start_x, self.end_x)
                max_x = max(self.start_x, self.end_x)
                min_y = min(self.start_y, self.end_y)
                max_y = max(self.start_y, self.end_y)
                
                # update coordinates 
                min_x = min_x*self.image_width
                max_x = max_x*self.image_width
                min_y = min_y*self.image_height
                max_y = max_y*self.image_height
                
                self.input_boxes = [min_x, min_y, max_x, max_y]
    
                image = self.image_widget.get_pil_image()
                
                image = image.resize((512, 512))
                inputs = self.processor(image, input_boxes=[[self.input_boxes]], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                outputs = self.model(**inputs, multimask_output=False)
                probs = self.processor.image_processor.post_process_masks(outputs.pred_masks.sigmoid().cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), binarize=False)
                mask = probs[0] > 0.5
                
                # different colors for different masks except black
                color = np.array([random.randint(1, 255) if i != 3 else 255 for i in range(4)])
                #color = np.array([143, 0, 255, 153])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1).numpy() * color.reshape(1, 1, -1)
                mask_image = mask_image.astype(np.uint8)
                
                height, width, channel = mask_image.shape
                num_bytes = channel*width
                self.qimage = QImage(mask_image.data, width, height, num_bytes, QImage.Format_ARGB32)
                
                self.imageArea.set_background_image(self.qimage)
        else:
            self.warning_box = QMessageBox()
            self.warning_box.setIcon(QMessageBox.Warning)
            self.warning_box.setWindowTitle('Warning')
            self.warning_box.setText('Please upload an image')
            self.warning_box.setStandardButtons(QMessageBox.Ok)
            self.warning_box.setDefaultButton(QMessageBox.Ok)
            self.warning_box.exec_()
                
    def clear_mask(self):
        self.imageArea.clear_all()
        self.info_label.setVisible(False)
        self.imageArea.current_region_named = True
        self.imageArea.mask_image_labels = np.zeros((512, 512), dtype=np.uint8)
        

    def check_buttons(self, radioButton):
        if radioButton == self.brush_radio_button:
            self.set_brush_mode(True)
        elif radioButton == self.eraser_radio_button:
            self.set_eraser_mode(True)
        
        
    def set_brush_mode(self, toggled):
        if toggled:
            self.imageArea.drawMode = DrawMode.Point
            self.imageArea.brushSize = 5  
            self.imageArea.brushColor = Qt.red
            

    def set_eraser_mode(self, toggled):
        if toggled:
            self.imageArea.drawMode = DrawMode.Eraser
            self.imageArea.brushSize_eraser = 20
    
    def organsList(self):
        self.organs = QGroupBox("Organs list")
        
        # list for organs
        self.organs_list = QComboBox()
        self.organs_list.addItem("") 
        self.organs_list.addItem("Bladder")
        self.organs_list.addItem("Bowel")
        self.organs_list.addItem("Gallbladder")
        self.organs_list.addItem("Kidney")
        self.organs_list.addItem("Liver")
        self.organs_list.addItem("Pancreas")
        self.organs_list.addItem("Soft Tissue")
        self.organs_list.addItem("Spleen")
        self.organs_list.addItem("Vascular")
        self.organs_list.addItem("Free abdominal fluid")
        self.organs_list.addItem("Lung")
        self.organs_list.setEditable(False)
        self.organs_list.currentIndexChanged.connect(self.imageArea.set_region_name)
        
        qv = QVBoxLayout()
        qv.addWidget(self.organs_list)
        self.organs.setLayout(qv)
        self.box.vbox.addWidget(self.organs)
    
    def setBrushStyle(self):
        self.brush_line_type = QGroupBox("Brush style")
        self.brush_line_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.styleBtn1 = QRadioButton(" Solid")
        self.styleBtn1.setIconSize(QSize(32, 64))
        self.styleBtn1.clicked.connect(lambda: self.changeBrushStyle(self.styleBtn1))

        self.styleBtn2 = QRadioButton(" Dash")
        self.styleBtn2.setIconSize(QSize(32, 64))
        self.styleBtn2.clicked.connect(lambda: self.changeBrushStyle(self.styleBtn2))

        self.styleBtn3 = QRadioButton(" Dot")
        self.styleBtn3.setIconSize(QSize(32, 64))
        self.styleBtn3.clicked.connect(lambda: self.changeBrushStyle(self.styleBtn3))

        self.styleBtn1.setChecked(True)
        qv = QVBoxLayout()
        qv.addWidget(self.styleBtn1)
        qv.addWidget(self.styleBtn2)
        qv.addWidget(self.styleBtn3)
        self.brush_line_type.setLayout(qv)
        self.box.vbox.addWidget(self.brush_line_type)

    def setBrushCap(self):
        self.brush_cap_type = QGroupBox("Brush cap")
        self.brush_cap_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
\
        self.capBtn1 = QRadioButton("Square")
        self.capBtn1.clicked.connect(lambda: self.changeBrushCap(self.capBtn1))
        self.capBtn2 = QRadioButton("Flat")
        self.capBtn2.clicked.connect(lambda: self.changeBrushCap(self.capBtn2))
        self.capBtn3 = QRadioButton("Round")
        self.capBtn3.clicked.connect(lambda: self.changeBrushCap(self.capBtn3))

        self.capBtn3.setChecked(True)
        qv = QVBoxLayout()
        qv.addWidget(self.capBtn1)
        qv.addWidget(self.capBtn2)
        qv.addWidget(self.capBtn3)
        self.brush_cap_type.setLayout(qv)
        self.box.vbox.addWidget(self.brush_cap_type)

   
    def changeBrushCap(self, btn):
        if btn.text() == "Square":
            if btn.isChecked():
                self.imageArea.brushCap = Qt.SquareCap
        if btn.text() == "Flat":
            if btn.isChecked():
                self.imageArea.brushCap = Qt.FlatCap
        if btn.text() == "Round":
            if btn.isChecked():
                self.imageArea.brushCap = Qt.RoundCap

  
    def changeBrushStyle(self, btn):
        if btn.text() == " Solid":
            if btn.isChecked():
                self.imageArea.brushStyle = Qt.SolidLine
        if btn.text() == " Dash":
            if btn.isChecked():
                self.imageArea.brushStyle = Qt.DashLine
        if btn.text() == " Dot":
            if btn.isChecked():
                self.imageArea.brushStyle = Qt.DotLine

 
    def setBrushSlider(self):
        self.groupBoxSlider = QGroupBox("Brush size")
        self.groupBoxSlider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.brush_thickness = QSlider(Qt.Horizontal)
        self.brush_thickness.setMinimum(1)
        self.brush_thickness.setMaximum(40)
        self.brush_thickness.valueChanged.connect(self.sizeSliderChange)
        
        self.brushSizeLabel = QLabel()
        self.brushSizeLabel.setText("%s px" % self.imageArea.brushSize)

        qv = QVBoxLayout()
        qv.addWidget(self.brush_radio_button)
        qv.addWidget(self.brush_thickness)
        qv.addWidget(self.brushSizeLabel)
        self.groupBoxSlider.setLayout(qv)

        self.box.vbox.addWidget(self.groupBoxSlider)

  
    def sizeSliderChange(self, value):
        self.imageArea.brushSize = value
        self.brushSizeLabel.setText("%s px" % value)

    def setColorChanger(self):
        self.groupBoxColor = QGroupBox("Color")
        self.groupBoxColor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.col = QColor(255, 0, 0)
        self.brush_colour = QPushButton()
        self.brush_colour.setFixedSize(60, 60)
        self.brush_colour.clicked.connect(self.showColorDialog)
        self.brush_colour.setStyleSheet("background-color: %s" % self.col.name())
        self.box.vbox.addWidget(self.brush_colour)

        qv = QVBoxLayout()
        qv.addWidget(self.brush_colour)
        self.groupBoxColor.setLayout(qv)

        self.box.vbox.addWidget(self.groupBoxColor)

    def setEraserSlider(self):
        self.groupBoxSliderEraser = QGroupBox("Eraser size")
        self.groupBoxSliderEraser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.eraser_thickness = QSlider(Qt.Horizontal)
        self.eraser_thickness.setMinimum(1)
        self.eraser_thickness.setMaximum(40)
        self.eraser_thickness.valueChanged.connect(self.sizeSliderChange_eraser)
        
        self.eraserSizeLabel = QLabel()
        self.eraserSizeLabel.setText("%s px" % self.imageArea.brushSize_eraser)

        qv = QVBoxLayout()
        qv.addWidget(self.eraser_radio_button)
        qv.addWidget(self.eraser_thickness)
        qv.addWidget(self.eraserSizeLabel)
        self.groupBoxSliderEraser.setLayout(qv)

        self.box.vbox.addWidget(self.groupBoxSliderEraser)
        
    def sizeSliderChange_eraser(self, value):
        self.imageArea.brushSize_eraser = value
        self.eraserSizeLabel.setText("%s px" % value)
 
    def showColorDialog(self):
        self.col = QColorDialog.getColor()
        if self.col.isValid():
            self.brush_colour.setStyleSheet("background-color: %s" % self.col.name())
            self.imageArea.brushColor = self.col

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        if self.imageArea.resizeSavedImage.width() != 0:
            self.imageArea.image = self.imageArea.resizeSavedImage.scaled(self.imageArea.width(), self.imageArea.height(), QtCore.Qt.IgnoreAspectRatio)
        self.imageArea.update()
        
        

# for drawing rectangle on the uploaded image
class DrawRectangle(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.rect_start = None
        self.rect_end = None

    def load_image(self, image_path):
        self.image = QPixmap(image_path)
        self.setMinimumSize(self.image.size())
        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.image)

            if self.rect_start and self.rect_end:
                rect = self.get_rect()
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                painter.drawRect(rect)

    def get_rect(self):
        start = QPoint(min(self.rect_start.x(), self.rect_end.x()),
                       min(self.rect_start.y(), self.rect_end.y()))
        end = QPoint(max(self.rect_start.x(), self.rect_end.x()),
                     max(self.rect_start.y(), self.rect_end.y()))
        return QRect(start, end)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect_start = event.pos()
            self.rect_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.rect_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect_end = event.pos()
            self.update()
            
    def get_pil_image(self):
        if self.image:
            qimage = self.image.toImage()
            raw_data = qimage.bits().asstring(qimage.byteCount())
            image = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), raw_data, "raw", "BGRA", 0, 1)
            pil_image = image.convert('RGB')
            return pil_image
        return None
    
    
class DrawMode(Enum):
    Point = 1
    Line = 2
    Eraser = 3
    

class ToolBox(QWidget):
    def __init__(self):
        super().__init__()

        screen = QGuiApplication.primaryScreen()
        screen_size = screen.size()
        
        self.setMaximumWidth(int(screen_size.width()*0.2))
        self.setMinimumWidth(int(screen_size.width()*0.1))

        self.vbox = QVBoxLayout()
        self.vbox.setSpacing(10)
        self.setLayout(self.vbox)
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    
class DrawingArea(QWidget):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        
        self.main_window = main_window
        
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.image = QImage(self.size(), QImage.Format_ARGB32)
        self.image.fill(Qt.transparent)
       
        self.resizeSavedImage = QImage(0, 0, QImage.Format_RGB32)
        self.savedImage = QImage(0, 0, QImage.Format_RGB32)
        
        #width_image = self.image.width()
        #height_image = self.image.height()
        self.mask_image_labels = np.zeros((512, 512), dtype=np.uint8)
        self.current_region_named = True
        self.current_region = {}
        
        self.undo_stack = []

        self.drawing = False
        self.brushSize = 1
        self.brushSize_eraser = 5
        self.brushColor = Qt.red
        self.brushStyle = Qt.SolidLine
        self.brushCap = Qt.RoundCap
        self.drawMode = DrawMode.Point


        self.lastPoint = QPoint()
        self.setMinimumWidth(512)
        
        self.highlighted_regions = []
     
        
    def resizeEvent(self, event):
        self.image = self.image.scaled(self.width(), self.height())
 
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.save_undo_state()
            if self.drawMode == DrawMode.Point:
                painter = QPainter(self.image) 
                painter.setPen(QPen(self.brushColor, self.brushSize, self.brushStyle, self.brushCap))
                painter.drawPoint(event.pos())
                self.drawing = True 
                self.lastPoint = event.pos() 
            elif self.drawMode == DrawMode.Line:
                
                if self.lastPoint == QPoint():
                    self.lastPoint = event.pos()
                else:
                    painter = QPainter(self.image)
                    painter.setPen(QPen(self.brushColor, self.brushSize, self.brushStyle, self.brushCap))
                    painter.drawLine(self.lastPoint, event.pos())
                    self.lastPoint = QPoint()
                    
            elif self.drawMode == DrawMode.Eraser:
                self.drawing = True
                self.erase_point(event.pos())
                self.lastPoint = event.pos()
            
            self.update()
            
        elif event.button() == Qt.RightButton:
            found = False
            if not self.highlighted_regions:
                self.highlight_region(event.pos())
            else:
                right_click = event.pos()
                clicked_point = (right_click.x(), right_click.y())
                for region in self.highlighted_regions:
                    if clicked_point in region:
                        self.unhighlight_region(event.pos())
                        found = True
                        break
                if not found:
                    self.highlight_region(event.pos())

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            if self.drawMode == DrawMode.Point:
                painter = QPainter(self.image) 
                painter.setPen(QPen(self.brushColor, self.brushSize, self.brushStyle, self.brushCap))
                painter.drawLine(self.lastPoint, event.pos())
            elif self.drawMode == DrawMode.Eraser:
                self.erase_line(self.lastPoint, event.pos())
                
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.save_undo_state()
            self.savedImage = self.resizeSavedImage
            self.resizeSavedImage = self.image
            self.drawing = False

        
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        
    def erase_point(self, pos):
        painter = QPainter(self.image)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setPen(QPen(Qt.transparent, self.brushSize_eraser, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(pos)

    def erase_line(self, last_point, current_pos):
        painter = QPainter(self.image)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setPen(QPen(Qt.transparent, self.brushSize_eraser, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(last_point, current_pos)
        
    def set_background_image(self, image):
        self.save_undo_state()
        painter = QPainter(self.image)
        painter.drawImage(0, 0, image.scaled(self.image.size()))
        self.update()
    
    def save_undo_state(self):
        self.undo_stack.append(self.image.copy())

    def undo(self):
        if self.undo_stack:
            self.image = self.undo_stack.pop()
            if not self.current_region_named:
                self.main_window.info_label.setVisible(False)
                self.current_region_named = True
            self.update()
        else:
            self.main_window.info_label.setVisible(False)
            self.current_region_named = True
    
    def clear_all(self):
        self.save_undo_state()
        self.image.fill(Qt.transparent)
        self.update()
        
    
    def highlight_region(self, start_point):
        if not self.current_region_named:
            
            self.warning_box_highlighted_region = QMessageBox()
            self.warning_box_highlighted_region.setIcon(QMessageBox.Warning)
            self.warning_box_highlighted_region.setWindowTitle('Warning')
            self.warning_box_highlighted_region.setText('Please name the current highlighted region before highlighting a new one')
            self.warning_box_highlighted_region.setStandardButtons(QMessageBox.Ok)
            self.warning_box_highlighted_region.setDefaultButton(QMessageBox.Ok)
            self.warning_box_highlighted_region.exec_()
        
        else:
            self.main_window.organs_list.setCurrentIndex(0)
            
            target_color = self.image.pixelColor(start_point)
            if target_color.alpha() != 0:
                width, height = self.image.width(), self.image.height()
                visited = np.zeros((height, width), dtype=bool)
                
                stack = [start_point]
                self.current_region = {}
                
                while stack:
                    point = stack.pop()
                    x, y = point.x(), point.y()
                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue
                    if visited[y, x]:
                        continue
                    
                    current_color = self.image.pixelColor(point)
                    if current_color != target_color or current_color.alpha() == 0:
                        continue
                    
                    visited[y, x] = True
                    self.current_region[(x, y)] = current_color
                    
                    self.image.setPixelColor(point, QColor(255, 255, 0, 127))
                    
                    stack.append(QPoint(x+1, y))
                    stack.append(QPoint(x-1, y))
                    stack.append(QPoint(x, y+1))
                    stack.append(QPoint(x, y-1))
                
                self.highlighted_regions.append(self.current_region)
                self.current_region_named = False 
                self.main_window.info_label.setVisible(True)
                
                #self.setEnabled(False)
                self.update()
     
        
    def set_region_name(self):
        selected_organ = self.main_window.organs_list.currentText()
        if selected_organ and not self.current_region_named:
            self.current_region_named = True 
            self.main_window.info_label.setVisible(False)
            
            value = 0  
            
            if selected_organ == "Bladder":
                value = 1
            elif selected_organ == "Bowel":
                value = 2
            elif selected_organ == "Gallbladder":
                value = 3
            elif selected_organ == "Kidney":
                value = 4
            elif selected_organ == "Liver":
                value = 5
            elif selected_organ == "Pancreas":
                value = 6
            elif selected_organ == "Soft Tissue":
                value = 7
            elif selected_organ == "Spleen":
                value = 8
            elif selected_organ == "Vascular":
                value = 9
            elif selected_organ == "Free abdominal fluid":
                value = 10
            elif selected_organ == "Lung":
                value = 11
            
            for x, y in self.current_region:
                self.mask_image_labels[y, x] = value
            
            #self.setEnabled(True)
            
    def unhighlight_region(self, point):
        chosen_point = (point.x(), point.y())
        for region in self.highlighted_regions:
            if chosen_point in region:
                for (x, y), color in region.items():
                    self.image.setPixelColor(QPoint(x, y), color)
                self.highlighted_regions.remove(region)
                break
            
        if chosen_point in self.current_region:
            self.current_region_named = True
            self.main_window.info_label.setVisible(False)
                
        self.update()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = QGuiApplication.primaryScreen()
    screen_size = screen.size()
    width = int(screen_size.width()*0.6) 
    height = int(screen_size.height()*0.8)
    window = Window()
    window.setGeometry(100, 100, width, height)
    window.show()
    sys.exit(app.exec_())
    
