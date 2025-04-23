import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QLineEdit, QTabWidget, QWidget, QVBoxLayout, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox
import pyqtgraph.opengl as gl
from shell_generator import generate_shell
from PyQt5 import QtGui
import numpy as np  # for ontogeny function eval
import qdarkstyle  # apply modern dark style
import septa_fitting as sf  # module for fitting and saving equations
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSplitter, QLabel, QSlider, QGridLayout, 
                           QDoubleSpinBox, QComboBox, QToolBar, QAction, QCheckBox, QDockWidget, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont
from taphonomy import apply_taphonomy, restore_taphonomy  # import taphonomy transforms

class LitGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initializeGL(self):
        super().initializeGL()
        # Set up OpenGL lighting when context is initialized
        from OpenGL.GL import (
            glEnable, glLightfv, glColorMaterial, GL_LIGHT0, GL_LIGHT1,
            GL_LIGHTING, GL_NORMALIZE, GL_COLOR_MATERIAL, GL_POSITION,
            GL_AMBIENT, GL_DIFFUSE, GL_SPECULAR, glShadeModel,
            GL_SMOOTH, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE
        )
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        # Key light
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1.0, 1.0, 1.0, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        # Fill light
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (-1.0, -0.5, 0.5, 0.0))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (0.1, 0.1, 0.1, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.4, 0.4, 0.4, 1.0))
        glLightfv(GL_LIGHT1, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

class ShellViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # apply dark style for modern look
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # add parameters panel dock
        self.param_view = QTextEdit()
        self.param_view.setReadOnly(True)
        dock = QDockWidget("Parameters", self)
        dock.setWidget(self.param_view)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        # debounced update timer
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50)
        self.update_timer.timeout.connect(self._update_mesh)
        self._build_menu()
        self._build_toolbar()
        self.setWindowTitle("Cephalopod Shell 3D Simulator")
        self.resize(1200, 800)
        # store default sampling parameters
        self.theta_steps = 400
        self.phi_steps = 30
        self._build_ui()
        # initialize butterfly flag before first mesh update
        self.butterfly = False
        self._update_mesh()
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        # File menu
        file_menu = self.menuBar().addMenu("File")
        exp_obj = QtWidgets.QAction("Export OBJ...", self)
        exp_obj.triggered.connect(self.export_obj)
        file_menu.addAction(exp_obj)
        file_menu.addSeparator()
        exit_act = QtWidgets.QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)
        # View menu
        view_menu = self.menuBar().addMenu("View")
        toggle_grid = QtWidgets.QAction("Toggle Grid", self, checkable=True)
        toggle_grid.setChecked(True)
        toggle_grid.triggered.connect(self._toggle_grid)
        view_menu.addAction(toggle_grid)
        # Presets menu
        presets_menu = self.menuBar().addMenu("Presets")
        for name in ("Nautilus", "Spirula", "Baculites"):
            act = QtWidgets.QAction(name, self)
            act.triggered.connect(lambda _, n=name: self.load_preset(n))
            presets_menu.addAction(act)

    def _build_toolbar(self):
        tb = self.addToolBar("Tools")
        reset_view = QtWidgets.QAction("Reset View", self)
        reset_view.triggered.connect(lambda: self.view.opts.update({'distance':10}))
        tb.addAction(reset_view)
        animate = QtWidgets.QAction("Animate Growth", self)
        animate.triggered.connect(self._animate_growth)
        tb.addAction(animate)
        # fit view button
        fit_act = QtWidgets.QAction("Fit Shell", self)
        fit_act.triggered.connect(self.fit_view)
        tb.addAction(fit_act)
        # x-ray toggle
        xray_act = QtWidgets.QAction("X-ray View", self, checkable=True)
        xray_act.setToolTip("Toggle X-ray transparency to see internal structures")
        xray_act.triggered.connect(self.toggle_xray)
        tb.addAction(xray_act)
        # butterfly view toggle
        bf_act = QtWidgets.QAction("Butterfly View", self, checkable=True)
        bf_act.setToolTip("Toggle butterfly cut-and-mirror view to inspect internals")
        bf_act.triggered.connect(self.toggle_butterfly)
        tb.addAction(bf_act)
        
        # Add cross-section view toggle
        cs_act = QtWidgets.QAction("Cross-Section View", self, checkable=True)
        cs_act.setToolTip("Toggle cross-section view to analyze internal structures")
        cs_act.triggered.connect(self.toggle_cross_section)
        tb.addAction(cs_act)
        
        # Add butterfly axis selection
        tb.addSeparator()
        tb.addWidget(QLabel("Butterfly Axis:"))
        self.butterfly_axis = QComboBox()
        self.butterfly_axis.addItems(["Auto", "X", "Y", "Z"])
        self.butterfly_axis.setToolTip("Select axis for butterfly view cutting plane")
        self.butterfly_axis.currentTextChanged.connect(self._schedule_update)
        tb.addWidget(self.butterfly_axis)
        
        # Add cross-section position control
        tb.addSeparator()
        tb.addWidget(QLabel("Cross-Section:"))
        self.cross_section_slider = QSlider(Qt.Horizontal)
        self.cross_section_slider.setRange(0, 100)
        self.cross_section_slider.setValue(50)
        self.cross_section_slider.setToolTip("Adjust cross-section position")
        self.cross_section_slider.valueChanged.connect(self._schedule_update)
        self.cross_section_slider.setFixedWidth(100)
        tb.addWidget(self.cross_section_slider)
        
        # Add color mapping controls
        tb.addSeparator()
        tb.addWidget(QLabel("Color Mode:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["None", "Radius", "Height"])
        self.color_mode_combo.setToolTip("Color vertices by a data metric")
        self.color_mode_combo.currentTextChanged.connect(self._update_mesh)
        tb.addWidget(self.color_mode_combo)
        tb.addWidget(QLabel("Color Scale:"))
        self.color_scale_combo = QComboBox()
        self.color_scale_combo.addItems(["Viridis", "Plasma", "Jet", "Hot", "Cool"])
        self.color_scale_combo.setToolTip("Select a colormap for vertex coloring")
        self.color_scale_combo.currentTextChanged.connect(self._update_mesh)
        tb.addWidget(self.color_scale_combo)
        
        # add taphonomy controls
        tb.addSeparator()
        tb.addWidget(QLabel("Taphonomy:"))
        self.flatten_chk = QCheckBox("Flatten")
        self.flatten_chk.setToolTip("Toggle flatten distortion")
        self.flatten_chk.stateChanged.connect(self._schedule_update)
        tb.addWidget(self.flatten_chk)
        self.flatten_axis = QComboBox()
        self.flatten_axis.addItems(["X","Y","Z"])
        self.flatten_axis.setToolTip("Axis to flatten")
        self.flatten_axis.currentTextChanged.connect(self._schedule_update)
        tb.addWidget(self.flatten_axis)
        self.flatten_factor = QDoubleSpinBox()
        self.flatten_factor.setRange(0.1, 1.0)
        self.flatten_factor.setSingleStep(0.05)
        self.flatten_factor.setValue(1.0)
        self.flatten_factor.setToolTip("Flattening scale factor (<1 compresses)")
        self.flatten_factor.valueChanged.connect(self._schedule_update)
        tb.addWidget(self.flatten_factor)
        
        self.shear_chk = QCheckBox("Shear")
        self.shear_chk.setToolTip("Toggle shear distortion")
        self.shear_chk.stateChanged.connect(self._schedule_update)
        tb.addWidget(self.shear_chk)
        self.shear_axis = QComboBox()
        self.shear_axis.addItems(["X","Y","Z"])
        self.shear_axis.setToolTip("Axis to shear")
        self.shear_axis.currentTextChanged.connect(self._schedule_update)
        tb.addWidget(self.shear_axis)
        self.shear_dir_box = QComboBox()
        self.shear_dir_box.addItems(["X","Y","Z"])
        self.shear_dir_box.setToolTip("Direction axis for shear")
        self.shear_dir_box.currentTextChanged.connect(self._schedule_update)
        tb.addWidget(self.shear_dir_box)
        self.shear_factor = QDoubleSpinBox()
        self.shear_factor.setRange(-1.0, 1.0)
        self.shear_factor.setSingleStep(0.1)
        self.shear_factor.setValue(0.0)
        self.shear_factor.setToolTip("Shear factor")
        self.shear_factor.valueChanged.connect(self._schedule_update)
        tb.addWidget(self.shear_factor)
        
        self.nacre_spin = QDoubleSpinBox()
        self.nacre_spin.setRange(0.0, 1.0)
        self.nacre_spin.setSingleStep(0.01)
        self.nacre_spin.setValue(0.0)
        self.nacre_spin.setToolTip("Nacre layer thickness")
        self.nacre_spin.valueChanged.connect(self._schedule_update)
        tb.addWidget(QLabel("Nacre:"))
        tb.addWidget(self.nacre_spin)
        # Restore original geometry checkbox
        self.restore_chk = QCheckBox("Restore")
        self.restore_chk.setToolTip("Invert taphonomic distortions")
        self.restore_chk.stateChanged.connect(self._schedule_update)
        tb.addWidget(self.restore_chk)

    def _build_ui(self):
        # central widget with horizontal layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.view = LitGLViewWidget()
        # lighting will be set up once the OpenGL context is initialized in showEvent
        layout.addWidget(self.view, 1)

        # organize controls into tabs
        form = QtWidgets.QFormLayout()
        self.sliders = {}
        params = [
            ("W", 0.1, 5.0, 1.3),  # expanded range for tighter coils
            ("D", 0.0, 2.0, 0.5),
            ("T", 0.0, 5.0, 0.0),
            ("S", 0.01,1.0, 0.1),
            ("Turns", 1, 10, 5),
            ("C", 0.0, 1.0, 0.0),  # curvature
            ("Tau", 0.0, 1.0, 0.0),  # torsion
        ]
        for name, mn, mx, default in params:
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(int((default - mn) / (mx - mn) * 1000))
            slider.valueChanged.connect(self._schedule_update)
            slider.sliderReleased.connect(self._update_mesh)
            # rename D slider to 'Tightness (D)' for clarity
            label = 'Tightness (D)' if name == 'D' else name
            form.addRow(label, slider)
            self.sliders[name] = (slider, mn, mx)

        # ontogeny function expressions
        self.W_expr = QLineEdit(str(params[0][3]))
        self.W_expr.textChanged.connect(self._schedule_update)
        form.addRow("W(θ)", self.W_expr)
        self.D_expr = QLineEdit(str(params[1][3]))
        self.D_expr.textChanged.connect(self._schedule_update)
        form.addRow("D(θ)", self.D_expr)
        self.T_expr = QLineEdit(str(params[2][3]))
        self.T_expr.textChanged.connect(self._schedule_update)
        form.addRow("T(θ)", self.T_expr)
        self.S_expr = QLineEdit(str(params[3][3]))
        self.S_expr.textChanged.connect(self._schedule_update)
        form.addRow("S(θ)", self.S_expr)
        # septa controls
        self.septa_spin = QSpinBox()
        self.septa_spin.setRange(0, 50)
        self.septa_spin.valueChanged.connect(self._schedule_update)
        self.septa_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Septa count", self.septa_spin)
        self.septum_combo = QComboBox()
        self.septum_combo.addItems(["none", "synclastic", "anticlastic"])
        self.septum_combo.currentTextChanged.connect(self._schedule_update)
        self.septum_combo.currentTextChanged.connect(lambda: self._update_mesh())
        form.addRow("Septum shape", self.septum_combo)
        # advanced septa spacing controls
        self.septa_spacing_spin = QSpinBox()
        self.septa_spacing_spin.setRange(0, 1000)
        self.septa_spacing_spin.setSingleStep(1)
        self.septa_spacing_spin.setValue(0)
        self.septa_spacing_spin.valueChanged.connect(self._schedule_update)
        self.septa_spacing_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Septa spacing", self.septa_spacing_spin)
        self.septa_indices_edit = QLineEdit()
        self.septa_indices_edit.setPlaceholderText("e.g. 10,20,30")
        self.septa_indices_edit.textChanged.connect(self._schedule_update)
        self.septa_indices_edit.editingFinished.connect(self._update_mesh)
        form.addRow("Custom septa indices", self.septa_indices_edit)
        # involute/evolute preset
        self.shell_type = QComboBox()
        self.shell_type.addItems(["custom", "involute", "evolute"])
        self.shell_type.currentTextChanged.connect(self._schedule_update)
        self.shell_type.currentTextChanged.connect(lambda: self._update_mesh())
        form.addRow("Shell type", self.shell_type)
        # siphuncle controls
        self.siphun_combo = QComboBox()
        self.siphun_combo.addItems(["none", "central", "marginal"])
        self.siphun_combo.currentTextChanged.connect(self._schedule_update)
        self.siphun_combo.currentTextChanged.connect(lambda: self._update_mesh())
        form.addRow("Siphuncle pos", self.siphun_combo)
        self.siphun_spin = QDoubleSpinBox()
        self.siphun_spin.setRange(0.001, 0.5)
        self.siphun_spin.setSingleStep(0.005)
        self.siphun_spin.setValue(0.02)
        self.siphun_spin.valueChanged.connect(self._schedule_update)
        self.siphun_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Siphuncle radius", self.siphun_spin)
        # cross-section exponent control
        self.cross_exp_spin = QDoubleSpinBox()
        self.cross_exp_spin.setRange(0.1, 3.0)
        self.cross_exp_spin.setSingleStep(0.1)
        self.cross_exp_spin.setValue(1.0)
        self.cross_exp_spin.valueChanged.connect(self._schedule_update)
        self.cross_exp_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Cross-exp exponent", self.cross_exp_spin)
        # initial radius control
        self.r0_spin = QDoubleSpinBox()
        self.r0_spin.setRange(0.1, 5.0)
        self.r0_spin.setSingleStep(0.1)
        self.r0_spin.setValue(1.0)
        self.r0_spin.valueChanged.connect(self._schedule_update)
        self.r0_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Initial radius (r0)", self.r0_spin)
        # cross-section shape controls
        self.cross_shape_combo = QComboBox()
        self.cross_shape_combo.addItems(["circular", "elliptical", "lobate"])
        self.cross_shape_combo.currentTextChanged.connect(self._schedule_update)
        form.addRow("Cross-section", self.cross_shape_combo)
        # lobate parameters
        self.lobate_freq_spin = QSpinBox()
        self.lobate_freq_spin.setRange(0, 10)
        self.lobate_freq_spin.setValue(5)
        self.lobate_freq_spin.valueChanged.connect(self._schedule_update)
        form.addRow("Lobate freq", self.lobate_freq_spin)
        self.lobate_amp_spin = QDoubleSpinBox()
        self.lobate_amp_spin.setRange(0.0, 1.0)
        self.lobate_amp_spin.setSingleStep(0.05)
        self.lobate_amp_spin.setValue(0.2)
        self.lobate_amp_spin.valueChanged.connect(self._schedule_update)
        form.addRow("Lobate amp", self.lobate_amp_spin)
        # septa bending exponent control
        self.bend_exp_spin = QDoubleSpinBox()
        self.bend_exp_spin.setRange(0.1, 2.0)
        self.bend_exp_spin.setSingleStep(0.1)
        self.bend_exp_spin.setValue(0.5)
        self.bend_exp_spin.valueChanged.connect(self._schedule_update)
        self.bend_exp_spin.editingFinished.connect(self._update_mesh)
        form.addRow("Septa bending exp", self.bend_exp_spin)
        # ornamentation controls: ribs
        self.rib_freq_spin = QSpinBox()
        self.rib_freq_spin.setRange(0, 50)
        self.rib_freq_spin.setValue(0)
        self.rib_freq_spin.valueChanged.connect(self._schedule_update)
        form.addRow("Rib frequency", self.rib_freq_spin)
        self.rib_amp_spin = QDoubleSpinBox()
        tabs = QTabWidget()
        # parameters tab
        params_tab = QWidget()
        params_v = QVBoxLayout(params_tab)
        params_v.addLayout(form)
        # presets & export tab
        export_tab = QWidget()
        exp_v = QVBoxLayout(export_tab)
        self.preset_box = QComboBox()
        self.preset_box.addItems(["Select preset","Nautilus","Spirula","Baculites"])
        self.preset_box.currentTextChanged.connect(self.load_preset)
        exp_v.addWidget(self.preset_box)
        btn_exp = QPushButton("Export OBJ")
        btn_exp.clicked.connect(self.export_obj)
        exp_v.addWidget(btn_exp)
        # fitting & saving equations
        fit_septa_btn = QPushButton("Fit & Save Septa Eqn")
        fit_septa_btn.clicked.connect(self.fit_save_septa)
        exp_v.addWidget(fit_septa_btn)
        fit_shell_btn = QPushButton("Fit & Save Shell Eqn")
        fit_shell_btn.clicked.connect(self.fit_save_shell)
        exp_v.addWidget(fit_shell_btn)
        # load equations
        load_septa_btn = QPushButton("Load Septa Eqn")
        load_septa_btn.clicked.connect(self.load_septa_equation)
        exp_v.addWidget(load_septa_btn)
        load_shell_btn = QPushButton("Load Shell Eqn")
        load_shell_btn.clicked.connect(self.load_shell_equation)
        exp_v.addWidget(load_shell_btn)
        tabs.addTab(params_tab, "Parameters")
        tabs.addTab(export_tab, "Presets & Export")
        layout.addWidget(tabs)

    def _schedule_update(self):
        """Start debounced update timer."""
        self.update_timer.start()

    def _update_mesh(self):
        vals = {}
        for name, (slider, mn, mx) in self.sliders.items():
            v = slider.value() / 1000 * (mx - mn) + mn
            vals[name] = int(v) if name == "Turns" else v
        # update parameter view for reproducibility
        param_lines = []
        param_lines.append(f"W={vals['W']:.4f}, D={vals['D']:.4f}, T={vals['T']:.4f}, S={vals['S']:.4f}, Turns={vals['Turns']}")
        # additional parameters
        param_lines.append(f"C={vals.get('C',0):.4f}, Tau={vals.get('Tau',0):.4f}, r0={self.r0_spin.value():.4f}, cross-exp={self.cross_exp_spin.value():.4f}")
        param_lines.append(f"Shell type={self.shell_type.currentText()}, Septa count={self.septa_spin.value()}, Septum shape={self.septum_combo.currentText()}")
        param_lines.append(f"Siphuncle pos={self.siphun_combo.currentText()}, siphuncle radius={self.siphun_spin.value():.4f}")
        param_lines.append(f"Cross-section shape={self.cross_shape_combo.currentText()}, lobate freq={self.lobate_freq_spin.value()}, lobate amp={self.lobate_amp_spin.value():.4f}")
        param_lines.append(f"Septa bending exp={self.bend_exp_spin.value():.4f}, Rib freq={self.rib_freq_spin.value()}, Rib amp={self.rib_amp_spin.value() if hasattr(self,'rib_amp_spin') else 0}")
        param_lines.append(f"Color mode={self.color_mode_combo.currentText()}, Color scale={self.color_scale_combo.currentText()}")
        param_lines.append(f"Butterfly={self.butterfly}, Axis={self.butterfly_axis.currentText()}, Cross-section enabled={getattr(self,'cross_section',False)}, Cross-section slider={self.cross_section_slider.value()}%")
        self.param_view.setPlainText("\n".join(param_lines))

        # use slider values directly for W, D, T, S
        W_fun = lambda θ: vals["W"]
        D_fun = lambda θ: vals["D"]
        T_fun = lambda θ: vals["T"]
        Sw_fun = lambda θ: vals["S"]
        Sh_fun = lambda θ: vals["S"]
        # read cross-section exponent
        cross_exp = self.cross_exp_spin.value()
        # read initial radius
        r0 = self.r0_spin.value()
        verts, faces = generate_shell(
            W=W_fun, D=D_fun, T=T_fun,
            Sw=Sw_fun, Sh=Sh_fun,  # normal & binormal radii
            turns=vals["Turns"],
            r0=r0,
            C=vals.get("C", 0.0), Tau=vals.get("Tau", 0.0),
            cross_exp=cross_exp,
            shell_coil=self.shell_type.currentText(),
            septa_count=self.septa_spin.value(),
            septum_shape=self.septum_combo.currentText(),
            siphuncle_pos=self.siphun_combo.currentText(),
            siphuncle_radius=self.siphun_spin.value(),
            cross_shape=self.cross_shape_combo.currentText(),
            lobate_freq=self.lobate_freq_spin.value(),
            lobate_amp=self.lobate_amp_spin.value()
        )
        # apply taphonomic distortions
        flatten_cfg = None
        if self.flatten_chk.isChecked():
            flatten_cfg = {'axis': self.flatten_axis.currentText().lower(), 'factor': self.flatten_factor.value()}
        shear_cfg = None
        if self.shear_chk.isChecked():
            shear_cfg = {'shear_axis': self.shear_axis.currentText().lower(), 'direction_axis': self.shear_dir_box.currentText().lower(), 'factor': self.shear_factor.value()}
        nacre_thickness = self.nacre_spin.value()
        res = apply_taphonomy(verts, faces, flatten=flatten_cfg, shear=shear_cfg, nacre_thickness=nacre_thickness)
        verts, faces = res['verts'], res['faces']
        # restore geometry if requested
        if hasattr(self, 'restore_chk') and self.restore_chk.isChecked():
            verts = restore_taphonomy(verts, faces, flatten=flatten_cfg, shear=shear_cfg)
        
        # store verts for fit_view
        self.last_verts = verts

        # Cross-section view: slice mesh at a Z position
        if hasattr(self, 'cross_section') and getattr(self, 'cross_section', False):
            import numpy as _np
            v_arr = _np.array(verts, dtype=float)
            f_arr = _np.array(faces, dtype=int)
            # Get Z range
            zmin, zmax = v_arr[:,2].min(), v_arr[:,2].max()
            # Slider value 0-100 -> z position
            frac = self.cross_section_slider.value() / 100.0
            zcut = zmin + frac * (zmax - zmin)
            # Only keep faces whose centroid is above the cut
            keep = []
            for face in f_arr:
                zc = v_arr[list(face),2].mean()
                if zc >= zcut:
                    keep.append(face)
            faces = _np.array(keep, dtype=int)
            verts = v_arr

        # butterfly view: longitudinal cut along shell tube axis and mirror
        if self.butterfly:
            import numpy as _np
            # convert to arrays
            v_arr = _np.array(verts, dtype=float)
            f_arr = _np.array(faces, dtype=int)
            
            # Find the center of the shell by averaging all vertices
            shell_center = _np.mean(v_arr, axis=0)
            
            # Get the butterfly axis selected by the user
            selected_axis = self.butterfly_axis.currentText()
            
            # Define the cutting plane normal based on selected axis
            if selected_axis == "X":
                # Cut along X axis (normal is perpendicular to X in XY plane)
                plane_n = _np.array([0.0, 1.0, 0.0])
            elif selected_axis == "Y":
                # Cut along Y axis (normal is perpendicular to Y in XY plane)
                plane_n = _np.array([1.0, 0.0, 0.0])
            elif selected_axis == "Z":
                # Cut along Z axis (normal is in XY plane)
                plane_n = _np.array([0.0, 0.0, 1.0])
            else:  # "Auto" - use principal component analysis
                # Project all points to XY plane and find their covariance
                points_xy = v_arr[:, 0:2]
                center_xy = shell_center[0:2]
                
                # Calculate covariance matrix of XY coordinates
                cov_matrix = _np.cov((points_xy - center_xy).T)
                
                # Get eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = _np.linalg.eigh(cov_matrix)
                
                # The cutting plane normal should be in the XY plane
                # We'll use the principal axis (eigenvector with largest eigenvalue)
                principal_axis = eigenvectors[:, 1]  # Largest eigenvalue's eigenvector
                
                # Create cutting plane normal (perpendicular to principal axis in XY plane)
                plane_n = _np.array([principal_axis[1], -principal_axis[0], 0.0])
            
            # Ensure the normal is normalized
            plane_n = plane_n / _np.linalg.norm(plane_n)
            
            # Calculate bounding box to get a sense of shell size
            bbox_min = _np.min(v_arr, axis=0)
            bbox_max = _np.max(v_arr, axis=0)
            bbox_size = bbox_max - bbox_min
            
            # Scale separation based on shell size
            separation_distance = _np.linalg.norm(bbox_size) * 0.5
            
            # Sort vertices into left and right sides of the cutting plane
            left_mask = (_np.dot(v_arr - shell_center, plane_n) >= 0)
            right_mask = ~left_mask
            
            # Get indices for each side
            left_idx = _np.where(left_mask)[0]
            right_idx = _np.where(right_mask)[0]
            
            # Get vertices for each side
            v_left = v_arr[left_idx].copy()
            v_right = v_arr[right_idx].copy()
            
            # Calculate rotation axis (perpendicular to normal and up vector)
            # This will be used to rotate the halves outward
            up_vector = _np.array([0.0, 0.0, 1.0])
            rotation_axis = _np.cross(plane_n, up_vector)
            
            # If rotation axis is close to zero (plane_n parallel to up_vector),
            # use a different vector for cross product
            if _np.linalg.norm(rotation_axis) < 1e-6:
                rotation_axis = _np.cross(plane_n, _np.array([1.0, 0.0, 0.0]))
            
            # Normalize rotation axis
            rotation_axis = rotation_axis / _np.linalg.norm(rotation_axis)
            
            # Rotation angle (in radians) - this controls how much to flip out the pieces
            flip_angle = _np.radians(30)  # 30 degrees outward rotation
            
            # Create rotation matrices for left and right halves
            def rotation_matrix(axis, theta):
                """Return the rotation matrix for rotating theta radians about axis"""
                axis = axis / _np.linalg.norm(axis)
                a = _np.cos(theta / 2.0)
                b, c, d = -axis * _np.sin(theta / 2.0)
                aa, bb, cc, dd = a*a, b*b, c*c, d*d
                bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
                return _np.array([
                    [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
                ])
            
            # Create rotation matrices
            rot_left = rotation_matrix(rotation_axis, -flip_angle)  # Negative angle for left half
            rot_right = rotation_matrix(rotation_axis, flip_angle)  # Positive angle for right half
            
            # Calculate centers of each half
            left_center = _np.mean(v_left, axis=0)
            right_center = _np.mean(v_right, axis=0)
            
            # Translate vertices to origin, rotate, and translate back
            # For left half
            v_left = v_left - left_center  # Translate to origin
            v_left = _np.dot(v_left, rot_left.T)  # Apply rotation
            v_left = v_left + left_center  # Translate back
            
            # For right half
            v_right = v_right - right_center  # Translate to origin
            v_right = _np.dot(v_right, rot_right.T)  # Apply rotation
            v_right = v_right + right_center  # Translate back
            
            # Now translate the halves apart along the plane normal
            v_left = v_left + plane_n * (separation_distance / 2)
            v_right = v_right - plane_n * (separation_distance / 2)
            
            # Remap vertices for each side
            left_mapping = -_np.ones(len(v_arr), dtype=int)
            left_mapping[left_idx] = _np.arange(len(left_idx))
            
            right_mapping = -_np.ones(len(v_arr), dtype=int)
            right_mapping[right_idx] = _np.arange(len(right_idx))
            
            # Filter triangular faces - only keep those with all vertices on one side
            left_faces = []
            for face in f_arr:
                if left_mask[face[0]] and left_mask[face[1]] and left_mask[face[2]]:
                    # remap indices to new, smaller vertex array
                    new_face = (left_mapping[face[0]], left_mapping[face[1]], left_mapping[face[2]])
                    left_faces.append(new_face)
            
            right_faces = []
            for face in f_arr:
                if right_mask[face[0]] and right_mask[face[1]] and right_mask[face[2]]:
                    # remap indices to new, smaller vertex array
                    new_face = (right_mapping[face[0]], right_mapping[face[1]], right_mapping[face[2]])
                    right_faces.append(new_face)
            
            # Combine for final mesh
            v_all = _np.vstack([v_left, v_right])
            offset = len(v_left)
            f_all = left_faces + [(f[0]+offset, f[1]+offset, f[2]+offset) for f in right_faces]
            
            verts, faces = v_all, _np.array(f_all, dtype=int)

        # remove any mirror item when exiting butterfly mode
        if hasattr(self, 'mirror_item'):
            self.view.removeItem(self.mirror_item)
        # rebuild full mesh
        mesh = gl.MeshData(vertexes=verts, faces=faces)
        # apply color mapping based on user selection
        color_mode = self.color_mode_combo.currentText()
        color_scale = self.color_scale_combo.currentText()
        if (color_mode != "None"):
            import matplotlib.cm as cm
            import numpy as _np
            v_arr = _np.array(verts, dtype=float)
            if color_mode == "Radius":
                vals = _np.linalg.norm(v_arr[:, :2], axis=1)
            elif color_mode == "Height":
                vals = v_arr[:, 2]
            normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            cmap = getattr(cm, color_scale.lower())
            colors = cmap(normed)
            mesh.setVertexColors(colors)
        # remove previous main item if present
        if hasattr(self, 'item'):
            self.view.removeItem(self.item)
        # Use fixed-function pipeline for better OpenGL lighting
        self.item = gl.GLMeshItem(
            meshdata=mesh,
            smooth=True,
            glOptions='opaque'
        )
        self.view.addItem(self.item)

    # stubs for new functionality
    def export_obj(self): pass
    def _toggle_grid(self, state): self.view.opts['showGrid']=state
    def load_preset(self, name):
        """
        Load predefined shell parameters for classic taxa.
        """
        # golden ratio for Nautilus
        phi = (1 + 5**0.5) / 2
        if name.lower() == 'nautilus':
            self.W_expr.setText(str(phi))
            self.D_expr.setText('0.5')
            self.T_expr.setText('0')
            self.S_expr.setText('0.1')
            self.shell_type.setCurrentText('involute')
            self.septa_spin.setValue(30)
            self.septum_combo.setCurrentText('synclastic')
        elif name.lower() == 'spirula':
            self.W_expr.setText('1.1')
            self.D_expr.setText('1.0')
            self.T_expr.setText('0')
            self.S_expr.setText('0.05')
            self.shell_type.setCurrentText('custom')
            self.septa_spin.setValue(30)
            self.septum_combo.setCurrentText('synclastic')
        elif name.lower() == 'baculites':
            self.W_expr.setText('1.05')
            self.D_expr.setText('2.0')
            self.T_expr.setText('0')
            self.S_expr.setText('0.05')
            self.shell_type.setCurrentText('custom')
            self.septa_spin.setValue(20)
            self.septum_combo.setCurrentText('anticlastic')
        # trigger mesh update
        self._update_mesh()
    def _animate_growth(self):
        """Animate the shell growth step by step with user confirmation to continue."""
        # Store original turns value
        original_turns = int(self.sliders['Turns'][0].value() / 1000 * 
                            (self.sliders['Turns'][2] - self.sliders['Turns'][1]) + 
                            self.sliders['Turns'][1])
        
        # Calculate step size based on original turns
        step_size = max(1, original_turns // 10)  # Divide into at least 10 steps
        current_turns = step_size  # Start with first step
        
        # Create dialog buttons with more informative options
        continue_btn = QtWidgets.QPushButton("Continue to iterate")
        finish_btn = QtWidgets.QPushButton("Finish All")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        
        # Create a custom dialog with more information
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Shell Growth Animation")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Add informative label
        info_label = QtWidgets.QLabel("Animating shell growth...")
        layout.addWidget(info_label)
        
        # Progress indicators
        progress_label = QtWidgets.QLabel(f"Turn {current_turns} of {original_turns}")
        layout.addWidget(progress_label)
        
        # Add progress bar
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, original_turns)
        progress_bar.setValue(current_turns)
        layout.addWidget(progress_bar)
        
        # Button layout
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(continue_btn)
        btn_layout.addWidget(finish_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect button signals
        continue_btn.clicked.connect(lambda: dialog.done(1))  # Continue
        finish_btn.clicked.connect(lambda: dialog.done(2))    # Finish all
        cancel_btn.clicked.connect(lambda: dialog.done(0))    # Cancel
        
        # Set tooltips for better user guidance
        continue_btn.setToolTip("Continue to the next growth stage")
        finish_btn.setToolTip("Skip to the final growth stage")
        cancel_btn.setToolTip("Stop the animation and keep current state")
        
        # Animation loop
        while current_turns <= original_turns:
            # Calculate slider value from turns
            slider_value = int((current_turns - self.sliders['Turns'][1]) / 
                               (self.sliders['Turns'][2] - self.sliders['Turns'][1]) * 1000)
            
            # Set current number of turns
            self.sliders['Turns'][0].setValue(slider_value)
            
            # Update the mesh with current turns
            self._update_mesh()
            
            # Process events to ensure UI updates
            QtWidgets.QApplication.processEvents()
            
            # Update progress in dialog
            progress_label.setText(f"Turn {current_turns} of {original_turns}")
            progress_bar.setValue(current_turns)
            
            # If we've reached the last step, don't show the dialog
            if current_turns >= original_turns:
                break
            
            # Show dialog with "Continue to iterate?" question
            info_label.setText(f"Continue to iterate? (Currently at turn {current_turns})")
            result = dialog.exec_()
            
            if result == 0:  # Cancel
                self.statusBar().showMessage(f"Animation stopped at turn {current_turns}")
                break
            elif result == 1:  # Continue to next step
                current_turns += step_size
                current_turns = min(current_turns, original_turns)  # Don't exceed max
            elif result == 2:  # Finish all at once
                # Jump to final state
                current_turns = original_turns
                self.statusBar().showMessage("Animation completed")
        
        # Ensure we end with the original number of turns
        final_slider_value = int((original_turns - self.sliders['Turns'][1]) / 
                              (self.sliders['Turns'][2] - self.sliders['Turns'][1]) * 1000)
        self.sliders['Turns'][0].setValue(final_slider_value)
        self._update_mesh()

    def fit_view(self):
        """Center and scale the view to encompass the entire shell."""
        import numpy as _np
        if not hasattr(self, 'last_verts') or self.last_verts.size == 0:
            return
        # compute centroid and max radius
        pts = self.last_verts
        center = _np.mean(pts, axis=0)
        max_r = float(_np.max(_np.linalg.norm(pts - center, axis=1)))
        # update view center and distance
        self.view.opts['center'] = QtGui.QVector3D(*center)
        # factor to ensure full visibility
        self.view.opts['distance'] = max_r * 2.5
        self.view.update()

    def toggle_xray(self, enabled):
        """Toggle mesh transparency to let you see internal structures."""
        if not hasattr(self, 'item'):
            return
        if enabled:
            # semi-transparent additive rendering
            self.item.setGLOptions('additive')
            self.item.setColor((1.0, 1.0, 1.0, 0.3))
        else:
            # restore opaque shading
            self.item.setGLOptions('opaque')
            self.item.setColor((1.0, 1.0, 1.0, 1.0))

    def toggle_butterfly(self, enabled):
        """Toggle butterfly cut-and-mirror view."""
        self.butterfly = enabled
        self._update_mesh()

    def toggle_cross_section(self, enabled):
        """Toggle cross-section view to examine internal shell structures."""
        self.cross_section = enabled
        self._update_mesh()
        
        # Remove any previous cross-section plane if it exists
        if hasattr(self, 'cross_plane'):
            self.view.removeItem(self.cross_plane)
            
        # If enabled, schedule an update to render the cross-section
        if enabled:
            self._schedule_update()
        else:
            # When disabled, ensure we rebuild the complete shell
            self._update_mesh()

    def fit_save_septa(self):
        """Fit a polynomial to the current septa indices and save the equation."""
        import numpy as _np
        from datetime import datetime
        # determine septa indices
        text = self.septa_indices_edit.text().strip()
        if text:
            try:
                indices = [int(x) for x in text.split(',')]
            except:
                self.statusBar().showMessage("Invalid septa indices format.")
                return
        else:
            count = self.septa_spin.value()
            indices = _np.linspace(0, self.theta_steps-1, count, endpoint=False, dtype=int).tolist()
        # perform fitting
        coeffs = sf.fit_septa_indices(indices, self.theta_steps, self.sliders['Turns'][0].value()/1000*(10-1)+1)
        # save equation
        name = f"septa_{datetime.now():%Y%m%d_%H%M%S}"
        path = sf.save_septa_equation(coeffs, name)
        self.statusBar().showMessage(f"Saved septa equation to {path}")

    def fit_save_shell(self):
        """Fit the shell spiral growth equation and save it."""
        import numpy as _np
        from datetime import datetime
        # reshape last_verts into rings
        try:
            arr = self.last_verts.reshape(self.theta_steps, self.phi_steps, 3)
        except:
            self.statusBar().showMessage("Shell data not available for fitting.")
            return
        # compute radii from origin in XY
        radii = _np.linalg.norm(arr[:,:,0:2], axis=2)[:,0]
        turns = self.sliders['Turns'][0].value()/1000*(10-1)+1
        coeffs = sf.fit_shell_growth(radii, self.theta_steps, turns)
        # save equation
        name = f"shell_{datetime.now():%Y%m%d_%H%M%S}"
        path = sf.save_shell_equation(coeffs, name)
        self.statusBar().showMessage(f"Saved shell equation to {path}")

    def load_septa_equation(self):
        """Load and apply septa equation from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(self, 'Load Septa Equation', '.septa_equations', 'JSON Files (*.json)')
        if not path:
            return
        try:
            coeffs = sf.load_septa_equation(path)
            # number of septa to generate
            count = self.septa_spin.value()
            # build theta array for full shell
            full = np.linspace(0, 2*np.pi * (self.sliders['Turns'][0].value()/1000*(10-1)+1),
                               self.theta_steps+1)
            thetas = full[1:]
            # evaluate polynomial to get septal angles
            x = np.arange(count)
            theta_vals = np.polyval(coeffs, x)
            # find closest indices in thetas
            inds = [int(np.argmin(np.abs(thetas - t))) for t in theta_vals]
            # update UI
            self.septa_indices_edit.setText(','.join(str(i) for i in inds))
            self.statusBar().showMessage(f"Loaded septa eqn: applied {count} positions")
            self._update_mesh()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading septa eqn: {e}")

    def load_shell_equation(self):
        """Load and apply shell growth equation from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(self, 'Load Shell Equation', '.shell_equations', 'JSON Files (*.json)')
        if not path:
            return
        try:
            coeffs = sf.load_shell_equation(path)
            # coeffs = [k, log_r0]
            k, logr0 = coeffs
            # compute W per revolution
            Wval = np.exp(k * 2 * np.pi)
            # update UI sliders
            # W slider normalized
            mn, mx = self.sliders['W'][1], self.sliders['W'][2]
            pos = int((Wval - mn) / (mx - mn) * 1000)
            self.sliders['W'][0].setValue(max(0, min(1000, pos)))
            # r0 control
            self.r0_spin.setValue(np.exp(logr0))
            self.statusBar().showMessage(f"Loaded shell eqn: W={Wval:.3f}, r0={np.exp(logr0):.3f}")
            self._update_mesh()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading shell eqn: {e}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = ShellViewer()
    viewer.show()
    sys.exit(app.exec_())