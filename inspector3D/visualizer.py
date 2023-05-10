# default imports
import os
import cv2
from datetime import datetime
import numpy as np
import logging
import pyqtgraph as pg
from OpenGL.GL import GL_RGBA
import pyqtgraph.opengl as gl
import PyQt6 as qt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Visualizer(object):
    def __init__(self, frames_per_sec: float = 20, background_color: str = 'w', grid: bool = False,
                 resolution: list = [2560, 1600]):
        '''
        Visualizer for scatter plots and mesh items. It can be used to animate and compare simulations.
        :param frames_per_sec: {float} default: 20, amount of visualized frames per second
        :param background_color: {str} default: 'w'', string describing a color in which the background is drawn
        :param grid: {bool} default: False, if true a grid is drawn in the animation
        :param resolution: {list} default: [2560, 1600], resolution of the animation window
        '''
        self.run_animation = True
        self.refresh_rate = 1 / frames_per_sec * 1000
        self.mode_number = 1
        self.mode_ampl = 10
        self.save = False
        self.timestep = 0
        self.app = qt.QtWidgets.QApplication([])
        self.background_color = background_color
        self.grid = grid
        self.resolution = resolution
        self.close_on_end = False
        self.vertex_colors = None
        self.drawEdges = False
        self.save_format = 'gif'

        self.main_window = qt.QtWidgets.QMainWindow()
        self.main_window.setGeometry(0, 110, resolution[0], resolution[1])
        self.main_widget = qt.QtWidgets.QWidget()

        # Menubar
        menu_bar = self.main_window.menuBar()
        menu_bar.setNativeMenuBar(False)
        view_menu = qt.QtWidgets.QMenu("&View", self.main_window)
        menu_bar.addMenu(view_menu)
        color_action = qt.QtGui.QAction("&Backgound Color", view_menu)
        point_size_action = qt.QtGui.QAction("&Point Size", view_menu)
        rotation_action = qt.QtGui.QAction('Rotate%', view_menu, checkable=True)
        color_action.triggered.connect(self._set_background_color)
        point_size_action.triggered.connect(self._set_point_size)
        rotation_action.triggered.connect(self._set_rotation)
        view_menu.addAction(color_action)
        view_menu.addAction(point_size_action)
        view_menu.addAction(rotation_action)

        # Toolbar
        exitAct = qt.QtGui.QAction('Save', self.main_window)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(self._set_save)

        self.toolbar = self.main_window.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        # self.main_window.addToolBar("Save")

        self.layout = qt.QtWidgets.QGridLayout()
        self._add_ui_elements(False)
        self.main_widget.setLayout(self.layout)
        self.main_window.show()

    def animate(self, coordinates, times, faces=None, view: list = [0, 0],
                color=[], color_scale_limits: list = None, color_bar: str = 'linear', shift: bool = True,
                save_animation: bool = False, save_format='gif', animation_name: str = '', point_size: int = 1,
                rotate_camera: bool = False, camera_distance: float = None, _mode_ui: bool = False,
                close_on_end: bool = False, colormap='viridis', drawEdges=False):
        """
        create a timeseries of pointclouds and animate it
        :param coordinates: {array like} of shape {n_time_steps, n_points, 3} timeseries of coordinates
        :param times: {array like} of shape {n_time_steps,} discrete set of points in time
        :param faces: {array like} of shape {n_faces, 3} describing which points belong to which face
        :param view: {float list} default: [90, 30] describing the elevation and azimuth angle
            the camera is rotated around the individual axes to adjust its view around (rotation order: xyz)
                elevation {float} the elevation is the vertical angular distance between view's center and the observer
                azimuth {float} the azimuth is the angle between center positoin measured clockwise around the observer
        :param color: description of the color for the elements. This can be:
                {string} describing a color which is applied on the complete model (e.g., 'blue')
                {string list} n_elements x 1: describing a color for each element (e.g., ['r', 'r',...,'b')
                {array} 3x1: describing an rgb triplet which is applied on the complete model (e.g., [1, 0, 0]])
                {array} n_elements x 1: one value for each element which is mapped to a color
                {array} n_elements x 3: an rgb triplet for each element
                {array} n_elements x n_timesteps:
                    a time series for each element which is mapped to a color (e.g., an error quantity)
        :param color_scale_limits: instance of simulation class which contains timeseries of coordinates
        :param color_bar: instance of simulation class which contains timeseries of coordinates
        :param shift: {bool} if true, the individual simulations are shifted in y-directions, so that they do not overlap
        :param save_animation: {bool} to determine whether the animation is saved
        :param animation_name: {string} name of gif file, which will be saved if save_animation is true
        :param point_size: {float} default: 1, specifying points sizes applied to all points
        :param rotate_camera: {bool} default: False, if true the camera rotates around the center
        :param camera_distance: {float} default: None, distance from view's camera to center
        :return:
        """
        self._clear_all_widgets()
        self._add_ui_elements(_mode_ui)
        self.times = times
        self.times_end = len(times) - 1

        # check how many models are given
        if isinstance(coordinates, list):
            n_models = len(coordinates)
        else:
            n_models = 1
        # ensure that required quantities are a list of correct length so we can iterate over it
        quantities = [coordinates, faces]
        for i, quantity in enumerate(quantities):
            if not isinstance(quantity, list):
                quantities[i] = [quantity] * n_models
        coordinates, faces = quantities

        # update attributes
        self.run_animation = True
        self.save_format = save_format
        self.timestep = 0
        self.faces = faces
        self.color = color
        self.color_scale_limits = color_scale_limits
        self.color_bar = color_bar
        self.colormap = colormap
        self.save = save_animation
        self.close_on_end = close_on_end
        self.point_size = point_size
        self.drawEdges = drawEdges
        self.frames = []
        self.animation_name = animation_name
        self.shift = shift
        self.rotate = rotate_camera

        # process coordinates and set camera distance
        self._update_coordinates(coordinates, times, camera_distance)
        # calculate colors for complete animation
        self._update_colors(color, self.coords, times)
        self.w.setCameraParams(elevation=view[0], azimuth=view[1])

        # create frames
        self._add_plot_items()
        self.main_window.show()
        self.app.exec()
        self.play()

    def _set_save(self):
        self.save = True
        self.timestep = 0
        self._update()

    def _clear_all_widgets(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

    def _add_ui_elements(self, _mode_ui):
        '''
        add ui elements such as buttons, sliders, views, etc.
        :param _mode_ui: {bool} if modes are animated additional ui elements are added
        '''

        # main animation window
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 5
        self.w.setWindowTitle('MorMl')
        self.w.setBackgroundColor(self.background_color)
        self.w.setGeometry(0, 110, 1920, 1080)
        if self.grid:
            g = gl.GLGridItem()
            self.w.addItem(g)

        # create timer to start animation in seperate thread
        self.timer = qt.QtCore.QTimer()
        self.timer.timeout.connect(self._update)

        ## add ui elements
        self.play_button = qt.QtWidgets.QPushButton('Play')
        self.slider = qt.QtWidgets.QSlider(qt.QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100000)
        # self.slider.setSingleStep(1)
        self.slider.setValue(0)
        self.slider_refresh_rate = qt.QtWidgets.QSlider(qt.QtCore.Qt.Orientation.Horizontal)
        self.slider_refresh_rate.setRange(0, 200)
        self.slider_refresh_rate.setValue(100)

        self.mode_edit_label = qt.QtWidgets.QLabel()
        self.mode_edit_label.setText('Mode:')
        int_validator = qt.QtGui.QIntValidator()
        self.mode_edit = qt.QtWidgets.QLineEdit()
        self.mode_edit.setText(str(self.mode_number + 1))
        self.mode_edit.setValidator(int_validator)
        self.mode_ampl_label = qt.QtWidgets.QLabel()
        self.mode_ampl_label.setText('Amplification:')
        self.mode_ampl_edit = qt.QtWidgets.QLineEdit()
        self.mode_ampl_edit.setText(str(self.mode_ampl))
        double_validator = qt.QtGui.QDoubleValidator()
        self.mode_ampl_edit.setValidator(double_validator)
        self.colorbar = pg.GradientWidget(orientation='right', interactive=True)
        # add functions to buttons
        self.play_button.clicked.connect(self.play)
        self.slider.sliderMoved.connect(self._slide)
        self.slider_refresh_rate.valueChanged.connect(self._set_refresh_rate)
        self.mode_edit.editingFinished.connect(self.set_mode_number)
        self.mode_ampl_edit.editingFinished.connect(self._set_mode_amplification)

        # define layout
        self.layout.addWidget(self.w, 0, 0, 1, 7)
        self.layout.addWidget(self.play_button, 1, 0)
        self.layout.addWidget(self.slider, 1, 1)
        self.layout.addWidget(self.slider_refresh_rate, 1, 2)
        self.layout.addWidget(self.mode_edit_label, 1, 3)
        self.layout.addWidget(self.mode_edit, 1, 4)
        self.layout.addWidget(self.mode_ampl_label, 1, 5)
        self.layout.addWidget(self.mode_ampl_edit, 1, 6)
        self.layout.addWidget(self.colorbar, 0, 8)

        # if modes are animated additional ui elements are added
        if not _mode_ui:
            self.mode_edit_label.hide()
            self.mode_edit.hide()
            self.mode_ampl_label.hide()
            self.mode_ampl_edit.hide()

        # add the main widget to the window
        self.main_window.setCentralWidget(self.main_widget)

    def play(self):
        if self.run_animation == True:
            self.play_button.setText('Stop')
            self.timer.start(self.refresh_rate)
            self.run_animation = False
        else:
            self.play_button.setText('Play')
            self.timer.stop()
            self.run_animation = True

    def _set_point_size(self):
        point_size, ok = qt.QtWidgets.QInputDialog.getText(self.main_window, 'input dialog', 'Set Point Size')
        self.point_size = float(point_size)
        self._update()

    def _set_rotation(self):
        self.rotate = not self.rotate

    def _set_background_color(self):
        background_color, ok = qt.QtWidgets.QInputDialog.getText(self.main_window, 'input dialog', 'Set Point Size')
        self.background_color = background_color
        self.w.setBackgroundColor(self.background_color)
        self._update()

        # self.main_widget.show()

    def _set_mode_amplification(self):
        self.mode_ampl = float(self.mode_ampl_edit.text())
        if hasattr(self, 'color_mode'):
            coordinates, times = self._calculate_current_color_mode()
        else:
            coordinates, times = self._calculate_current_mode()
        self._update_coordinates(coordinates, times)
        self._update()

    def set_mode_number(self):
        self.mode_number = int(self.mode_edit.text()) - 1
        if hasattr(self, 'color_mode'):
            coordinates, times = self._calculate_current_color_mode()
        else:
            coordinates, times = self._calculate_current_mode()
        if not isinstance(coordinates, list):
            coordinates = [coordinates]
        self._update_coordinates(coordinates, times)
        self._update()

    def _slide(self):
        self.timestep = int(np.ceil(self.slider.value() / 100000 * self.times_end))
        self._update_plot_items()

    def _set_refresh_rate(self):
        self.timer.setInterval(self.refresh_rate / max(0.001, self.slider_refresh_rate.value() / 100))

    def _update(self):
        '''
        update viewwidget with new coordinates and colors of plot items
        '''

        # if animation reaches its end, replay it from the start
        if self.timestep > self.times_end:
            if self.close_on_end:
                self.app.quit()
                return
            else:
                self.timestep = 0

        self._update_plot_items()

        # adjust slider to fit to current time step
        self.timestep += 1
        self.slider.setValue(self.timestep / (self.times_end + 1) * 100000)
        # save frames in case it is requested
        if self.save:
            self.frames.append(self.w.renderToArray((self.resolution[0], self.resolution[1]), format=GL_RGBA))
            if len(self.frames) == len(self.times):
                self._save_animation(self.save_format)
                self.save = False
                self.frames = []

    def _update_plot_items(self):
        '''
        update coordinates and times of plot items
        '''
        if self.rotate:
            self.w.orbit(360 / self.times_end, 0)

        # update the plot items
        for i, coord in enumerate(self.coords):
            # in case mesh items are plotted the vertices (nodes) and their colors are updated
            if self.faces[i] is not None:
                v = self.coords[i][self.timestep, :, :3]
                if self.vertex_colors:
                    self.plot_items[i].setMeshData(vertexes=v, faces=self.faces[i],
                                                   vertexColors=self.colors[i][self.timestep])
                else:
                    self.plot_items[i].setMeshData(vertexes=v, faces=self.faces[i],
                                                   faceColors=self.colors[i][self.timestep])

            # for scatter items their positions and their color is updated
            else:
                self.plot_items[i].setData(color=self.colors[i][self.timestep],
                                           pos=self.coords[i][self.timestep, :, :3],
                                           size=self.point_size)

    def _add_plot_items(self):
        '''
        add plot items (scatter or mesh items) to the view widget for each given simulation
        '''
        # create list of plot elements, which will be animated
        self.plot_items = []
        # iterate over all given simulation saved in self.coords
        for i, coord in enumerate(self.coords):
            # in case faces are given, a mesh object can be visualized
            if self.faces[i] is not None:
                meshdata = pg.opengl.MeshData(
                    vertexes=coord[0, :, :3],
                    faces=self.faces[i],
                    faceColors=self.colors[i][0]
                )

                gl.shaders.Shaders.append(gl.shaders.ShaderProgram(
                    'lighting', [gl.shaders.VertexShader("""varying vec3 vN;
                                                            varying vec3 v;
                                                            varying vec4 color;
                                                            void main(void)  
                                                            {     
                                                               v = vec3(gl_ModelViewMatrix * gl_Vertex);       
                                                               vN = normalize(gl_NormalMatrix * gl_Normal);
                                                               color = gl_Color;
                                                               gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  
                                                            }"""),
                                 gl.shaders.FragmentShader("""varying vec3 vN;
                                                                varying vec3 v; 
                                                                varying vec4 color;
                                                                #define MAX_LIGHTS 1 
                                                                void main (void) 
                                                                { 
                                                                   vec3 N = normalize(vN);
                                                                   vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
                                                                   
                                                                   for (int i=0;i<MAX_LIGHTS;i++)
                                                                   {
                                                                      vec3 L = normalize(gl_LightSource[i].position.xyz - v); 
                                                                      vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0) 
                                                                      vec3 R = normalize(-reflect(L,N)); 
                                                                   
                                                                      vec4 Iamb = gl_LightSource[i].ambient; 
                                                                      vec4 Idiff = gl_LightSource[i].diffuse * max(dot(N,L), 0.0);
                                                                      Idiff = clamp(Idiff, 0.0, 1.0); 
                                                                      vec4 Ispec = gl_LightSource[i].specular * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);
                                                                      Ispec = clamp(Ispec, 0.0, 1.0); 
   
                                                                      finalColor += Iamb + Idiff + 0.01*Ispec;
                                                                   }
                                                                   gl_FragColor = color * finalColor; 
                                                                }""")
                                 ]))

                sp = pg.opengl.GLMeshItem(
                    meshdata=meshdata,
                    color=self.colors[i][0],
                    drawEdges=self.drawEdges,
                    edgeColor=(0.2, 0.2, 0.2, 1),
                    shader='lighting',  # viewNormalColor, shaded
                    glOptions='opaque',
                    computeNormals=True
                )

            # if no faces are given, a scatterplot is used for each animated simulation
            else:
                sp = gl.GLScatterPlotItem(
                    pos=self.coords[i][0, :, :3],
                    color=self.colors[i][0],
                    size=self.point_size,
                )
                sp.setGLOptions('opaque')
            self.plot_items.append(sp)
            # the plot items must be added to the Viewwidget to be seen
            self.w.addItem(sp)

    def _update_coordinates(self, coordinates, times, camera_distance=None):
        '''
        bring given coordinates in the correct form for further processing and calculate colors for all given simulations
        :param coordinates: {list} : list of arrays the x-, y-, and z-coordinates of simulations
                                    {array} each included array is of shape n_timestepsx3
        :param times: {array} n_timestepsx1: vector including all timesteps
        '''

        if camera_distance is None:
            x_lim, y_lim, z_lim = self.get_axes_limits(coordinates)
            dist = max([x_lim[0] + x_lim[1], y_lim[0] + y_lim[1], z_lim[0] + z_lim[1]])
            camera_distance = dist / 2 / np.tan(22.5)
        self.w.setCameraParams(distance=camera_distance)

        # center data around [0, 0, 0] so that camera points on center
        all_coordinates = np.concatenate(coordinates, axis=1)
        mean_coordinates = all_coordinates.mean(axis=(0, 1))
        y_ranges = np.zeros([len(coordinates), ])
        if self.shift:
            for i in range(len(y_ranges)):
                y_ranges[i] = coordinates[i][:, :, 1].max() - coordinates[i][:, :, 1].min()
            y_ranges = [np.array(y_ranges).max()] * np.array(range(0, len(y_ranges)))
            # y_ranges = np.array([0] + y_ranges)
        # coordinates = [(XYZ - mean_coordinates) + np.array([0, 1.1 * y_ranges[i], 0]) for i, XYZ in
        #                enumerate(coordinates)]

        # generate random points from -10 to 10, z-axis positive]
        self.coords = []
        for coord in coordinates:
            self.coords.append(coord)
        # self.app.exec()

    def _update_colors(self, color, coordinates, times):
        '''
        bring given coordinates in the correct form for further processing and calculate colors for all given simulations
        :param color: {array-like} : including colors for the animation
        :param coordinates: {list} : list of arrays the x-, y-, and z-coordinates of simulations
                                    {array} each included array is of shape n_timestepsx3
        :param times: {array} n_timestepsx1: vector including all timesteps
        '''
        n_elements = coordinates[0].shape[1]
        try:
            n_faces = self.faces.shape[0]
        except:
            n_faces = 0
        n_colors = [n_elements, n_faces]

        # if an error is given we will use a normed error for colorization
        self.colors = []

        if len(color) == 0:
            color = [coords[:, :, 2] for coords in coordinates]
        # in case there are as many entries in color as in XYZ_data we asume that the user wants to give each entry
        # another color
        if len(color) == len(coordinates):
            for individual_color in color:
                self.colors.append(
                    self._get_colors(individual_color, self.color_scale_limits, times, n_colors,
                                     color_bar=self.color_bar, colormap=self.colormap))
        # if not we just suppose that each geometry should be colored with the some color
        else:
            for _ in coordinates:
                self.colors.append(self._get_colors(color, self.color_scale_limits, times, n_colors,
                                                    color_bar=self.color_bar, colormap=self.colormap))

    def _save_animation(self, format='gif'):
        animation_path, name = os.path.split(self.animation_name)
        # set name for saving the gif file
        if not name:
            # if no name for the gif file is given just name it after current time stamp
            name = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        # check if results directory exists, if not generate it
        if not os.path.isdir(f'results/videos/'):
            os.makedirs(os.path.join(f'results/videos/'))
        # check if results directory exists, if not generate it
        if not os.path.isdir(f'results/videos/{animation_path}/'):
            os.makedirs(os.path.join(f'results/videos/{animation_path}/'))
        frames = [Image.fromarray(frame) for frame in self.frames]
        png_files = []
        for i, frame in enumerate(frames):
            png_file = os.path.abspath(f'results/videos/{animation_path}/{name}_{i}.png')
            frame.save(png_file)
            png_files.append(png_file)
        video_name = os.path.abspath(f'results/videos/{animation_path}/{name}.{format}')
        if format == 'gif':
            # save file as format in results directory
            frames[0].save(video_name,
                           append_images=frames[1:],
                           save_all=True,
                           duration=50, loop=0)
        elif format == 'mp4':
            frame = cv2.imread(png_files[0])
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
            for png_file in png_files:
                video.write(cv2.imread(png_file))
            cv2.destroyAllWindows()
            video.release()
        if self.color_scale_limits is not None:
            color_limits = np.array([[self.color_scale_limits[0], self.color_scale_limits[1]]])
            plt.figure(figsize=(4, 0.5))
            plt.imshow(color_limits, cmap=self.colormap)
            plt.gca().set_visible(False)
            cax = plt.axes([0.1, 0.5, 0.8, 0.5])
            plt.colorbar(orientation="horizontal", cax=cax)
            plt.savefig(os.path.abspath(f'results/videos/{animation_path}/{name}_colorbar.pdf'))

    def _get_colors(self, color, color_scale_limits, times, n_colors, color_bar='linear', colormap='plasma'):
        '''
        create list with n_timesteps entries of the size of n_elements, i.e.,
        for each animation step the color of each element is defined in this list
        :param color: description of the color for the elements. This can be:
                {string} describing a color which is applied on the complete model (e.g., 'blue')
                {string list} n_elements x 1: describing a color for each element (e.g., ['r', 'r',...,'b')
                {array} 3x1: describing an rgb triplet which is applied on the complete model (e.g., [1, 0, 0]])
                {array} n_elements x 1: one value for each element which is mapped to a color
                {array} n_elements x 3: an rgb triplet for each element
                {array} n_elements x n_timesteps:
                    a time series for each element which is mapped to a color (e.g., an error quantity)
        :param n_elements: {integer} amount of elements (e.g., points or triangles) which are animated
        :param times: {array} list of timesteps which will be visualized
        :param color_scale_limits: {tuplet} range of colorbar
        :param color_bar: {string: 'linear' or 'log'} defines whether the colorbar is scaled linear or logarithmic
        :return colors: {list} containing of color information per timestep
        '''
        # amount of time steps which is animated
        n_timesteps = len(times)

        # colormap used to visualize map values to color (not used in all cases)
        colormap = plt.get_cmap(colormap)
        color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # if the name (i.e., a string) of a color is given map it to corresponding rgb color
        if isinstance(color, str):
            color = list(mcolors.ColorConverter.to_rgb(color_dict[color]))
        # if a list of strings for each element is given
        if isinstance(color, list):
            if isinstance(color[0], str):
                color = [list(mcolors.ColorConverter.to_rgb(color_dict[current_color])) for current_color in color]
        # make sure color is an numpy array for following manipulations
        if not isinstance(color, np.ndarray):
            color = np.array(color)

        # check if colors are given per edge/node or per face
        n_elements = n_colors[0]
        n_faces = n_colors[1]
        if n_elements in color.shape:
            self.vertex_colors = True
        elif n_faces in color.shape:
            self.vertex_colors = False

        # in case only single values (like error quantities) are given which must be mapped to a color using a colormap
        if (color.shape == (n_elements, 1) or color.shape == (n_elements,) or color.shape == (
                n_timesteps, n_elements) or
                color.shape == (n_faces, 1) or color.shape == (n_faces,) or color.shape == (n_timesteps, n_faces)):
            # normalize values between [0, 1] linearly or logarithmic for colormap
            if not color_scale_limits:
                color_scale_limits = [color.min(), color.max()]
            if color_bar == 'linear':
                color_bar = mcolors.Normalize(vmin=color_scale_limits[0], vmax=color_scale_limits[1])
            elif color_bar == 'log':
                if color_scale_limits[0] == 0:
                    color_scale_limits[0] = color_scale_limits[0] + np.nextafter(0, 1)
                color_bar = mcolors.LogNorm(vmin=color_scale_limits[0], vmax=color_scale_limits[1])
            # calculate normalized values
            color_normed = color_bar(color)
            # map normalized values to color
            try:
                color = colormap(color_normed)[:, :, :].squeeze()
            except IndexError:
                color = colormap(color_normed)[:, :].squeeze()

        # if color only consists of 1 axis (one color for all elements) then fill out all elements
        if len(color.shape) == 1:
            # in case color is a rbg triplet or rgba tuplet paint all elements for all time steps with that color
            if color.shape[0] == 3:
                color = np.repeat(np.expand_dims(color, axis=1), n_elements, axis=1).T

        # if no axis for the time is given repeat color for all timesteps
        if len(times) not in color.shape:
            color = np.repeat(np.expand_dims(color, axis=0), n_timesteps, axis=0)

        # in case only 3 values per color are given append them (rgb + alpha):
        if color.shape[-1] != 4:
            color = np.concatenate([color, np.ones([color.shape[0], color.shape[1], 1])], axis=2)

        # create list with time items as entries
        colors = []
        for i_time, _ in enumerate(times):
            colors.append(color[i_time])

        # save colormap
        if color_scale_limits is not None:
            for tick in self.colorbar.listTicks():
                self.colorbar.item.removeTick(tick[0])
            # create tick values for colorbar in range from 0 to 1
            tick_values = np.linspace(color_scale_limits[0], color_scale_limits[1], len(colormap.colors))
            tick_values = [(value - tick_values.min()) / (tick_values.max() - tick_values.min()) for value in
                           tick_values]
            n_ticks = 5
            tick_values = np.linspace(0, 1, n_ticks)
            for i, value in np.ndenumerate(tick_values):
                tick_color = colormap(value)
                tick_color = qt.QtGui.QColor(tick_color[0] * 255, tick_color[1] * 255, tick_color[2] * 255)
                self.colorbar.item.addTick(x=tick_values[i], color=tick_color)
        return colors

    def animation_to_obj_sequence(self, coordinates, faces, normals=None, dirname="results/obj_sequence/",
                                  filename="frame",
                                  frame_rate=None):
        """
        Save a sequence of obj files for a given animation so that it can be animated in 3d engines like unity.
        """

        n_obj = len(coordinates)

        # center coordinates so that the origin of the animation has no offset
        coordinates = coordinates - coordinates[0].mean(axis=0)

        # we don't want to save too many obj files
        if frame_rate is None:
            frame_rate = int(n_obj / 10)
        # compute normals if not given
        if normals is None:
            normals = np.empty(coordinates.shape)
            for i, coordinate in enumerate(coordinates):
                meshdata = pg.opengl.MeshData(
                    vertexes=coordinate,
                    faces=faces
                )
                normals[i] = meshdata.vertexNormals()

        # create series of obj files:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        for i in np.arange(0, n_obj, frame_rate):
            obj_file = open(os.path.join(dirname, f'{filename}_{i:03d}.obj'), 'w')
            for item in coordinates[i]:
                obj_file.write(f"v {item[0]} {item[1]} {item[2]}\n")

            for item in normals[i]:
                obj_file.write(f"vn {item[0]} {item[1]} {item[2]}\n")

            # we increase the face index by 1 as unity starts counting at 1
            for item in faces + 1:
                obj_file.write(f"f {item[0]}//{item[0]} {item[1]}//{item[1]} {item[2]}//{item[2]}\n")
            obj_file.close()
            logging.info(f'created .obj file no {int(i / frame_rate)}/{int(n_obj / frame_rate)}')

    def mode_to_obj_sequence(self, reduction, reference_coordinates, faces, mode_number: int = 0,
                             amplification: float = 100, dirname="results/obj_sequence",
                             filename="frame", frame_rate=None):
        '''
        function to save a sequence of obj files for a given mode so that it can be animated in 3d engines like unity.
        :param reduction: instance of class Reduction (which represents certain reduction methods like POD)
        :param reference_coordinates: reference coordinates to which the mode displacement will be added
        :param mode_number: {int} default: 0, number of visualized mode (e.g., mode_number=0 will display first mode)
        :param amplification: {float} default: 100, amplification factor by which the mode will be amplified
        :param kwargs: for all further parameter look in the definition of animate()
        :return:
        '''
        self.reduction = reduction
        self.mode_number = mode_number
        self.mode_ampl = amplification
        self.reference_coordinates = reference_coordinates
        coordinates, _ = self._calculate_current_mode()
        self.animation_to_obj_sequence(coordinates, faces, dirname=dirname, filename=filename,
                                       frame_rate=frame_rate)

    @staticmethod
    def get_axes_limits(XYZ_data):
        """
        In order to visualize the results the plots axes limits must fit to the simulation results.
        Thus, we need to know the minimum and maximum values for each coordinate direction
        :param simulations: simulation results of the full and reduced model
        :return xlim, ylim, zlim: the axes limits in each coordinate direction
        """
        # ensure that XYZ_data is a list
        if not isinstance(XYZ_data, list):
            XYZ_data = [XYZ_data]

        # find maximum values of all nodes in each direction and over all simulation
        x_max = np.max([np.max(XYZ[:, :, 0]) for XYZ in XYZ_data])
        y_max = np.max([np.max(XYZ[:, :, 1]) for XYZ in XYZ_data])
        z_max = np.max([np.max(XYZ[:, :, 2]) for XYZ in XYZ_data])

        # find minimum values of all nodes in each direction and over all simulation
        x_min = np.min([np.max(XYZ[:, :, 0]) for XYZ in XYZ_data])
        y_min = np.min([np.max(XYZ[:, :, 1]) for XYZ in XYZ_data])
        z_min = np.min([np.max(XYZ[:, :, 2]) for XYZ in XYZ_data])

        # difference between maximum and minimum values
        mean_x = x_min + 0.5 * (x_max - x_min)
        mean_y = y_min + 0.5 * (y_max - y_min)
        mean_z = z_min + 0.5 * (z_max - z_min)
        # in order to receive equal axes, all of them will have a range of 110% of the maximum range (diff)
        diff = np.max((x_max - x_min, y_max - y_min, z_max - z_min))

        # the axes limits correspond to their mean values +/- 55% of the maximum difference in one coordinate direction
        x_lim = (mean_x - 0.35 * diff, mean_x + 0.35 * diff)
        y_lim = (mean_y - 0.35 * diff, mean_y + 0.35 * diff)
        z_lim = (mean_z - 0.35 * diff, mean_z + 0.35 * diff)
        return x_lim, y_lim, z_lim
