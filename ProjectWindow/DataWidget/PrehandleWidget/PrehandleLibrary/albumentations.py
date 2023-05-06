from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.DataWidget.PrehandleWidget.PrehandleLibrary.base_layer import PrehandleLayer, PrehandleLayerMenu
from ProjectWindow.utils import MenuContainer, Config
from utils import RangeSpinbox, FixedMultiSpinbox

import albumentations as A


class AInvertImgMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        self.params = {
            'p': p
        }

    @property
    def name(self):
        return 'InvertImg'

    @property
    def description(self):
        return 'Инвертирует входное изображение, вычитая значение пикселей из 255'


class AInvertImg(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AInvertImg, self).__init__(menu_container, config, AInvertImgMenu, 'InvertImg', A.InvertImg)


class AEqualizeMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        mode = QComboBox()
        mode.addItems(['cv', 'pil'])
        mode.setCurrentText('cv')

        by_channels = QCheckBox()
        by_channels.setChecked(True)

        self.params = {
            'p': p,
            'mode': mode,
            'by_channels': by_channels
        }

    @property
    def name(self):
        return 'Equalize'

    @property
    def description(self):
        return 'Выравнивает гистограмму изображения'


class AEqualize(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AEqualize, self).__init__(menu_container, config, AEqualizeMenu, 'Equalize', A.Equalize)


class ACLAHEMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        clip_limit = RangeSpinbox()
        clip_limit.set_value((1, 4))

        tile_grid_size = FixedMultiSpinbox(2, False)
        tile_grid_size.set_value((8, 8))

        self.params = {
            'p': p,
            'clip_limit': clip_limit,
            'tile_grid_size': tile_grid_size
        }

    @property
    def name(self):
        return 'CLAHE'

    @property
    def description(self):
        return 'Применяет адаптивную гистограммнуб коррекцию с ограниченным контрастом к входному изображению'


class ACLAHE(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ACLAHE, self).__init__(menu_container, config, ACLAHEMenu, 'CLAHE', A.CLAHE)


class AToSepiaMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        self.params = {
            'p': p
        }

    @property
    def name(self):
        return 'ToSepia'

    @property
    def description(self):
        return 'Накладывает на изображение сепию'


class AToSepia(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AToSepia, self).__init__(menu_container, config, AToSepiaMenu, 'ToSepia', A.ToSepia)


class AGaussianBlurMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        blur_limit = RangeSpinbox(0)
        blur_limit.set_value((3, 7))

        sigma_limit = RangeSpinbox(0, double=True)
        sigma_limit.set_value((0, 0))

        self.params = {
            'p': p,
            'blur_limit': blur_limit,
            'sigma_limit': sigma_limit
        }

    @property
    def name(self):
        return 'GaussianBlur'

    @property
    def description(self):
        return 'Размывает изображение Гауссовским фильтром'


class AGaussianBlur(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AGaussianBlur, self).__init__(menu_container, config, AGaussianBlurMenu, 'GaussianBlur', A.GaussianBlur)


class AHueSaturationValueMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        hue_shift_limit = RangeSpinbox(-100)
        hue_shift_limit.set_value((-20, 20))

        sat_shift_limit = RangeSpinbox(-100)
        sat_shift_limit.set_value((-30, 30))

        val_shift_limit = RangeSpinbox(-100)
        val_shift_limit.set_value((-20, 20))

        self.params = {
            'p': p,
            'hue_shift_limit': hue_shift_limit,
            'sat_shift_limit': sat_shift_limit,
            'val_shift_limit': val_shift_limit
        }

    @property
    def name(self):
        return 'HueSaturationValue'

    @property
    def description(self):
        return 'Случайно изменяет оттенок, насыщенность и значения пикселей входного изображения'


class AHueSaturationValue(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AHueSaturationValue, self).__init__(menu_container, config, AHueSaturationValueMenu, 'HueSaturationValue',
                                                  A.HueSaturationValue)


class ARandomContrastMenu(PrehandleLayerMenu):
    def parameters(self):
        p = QDoubleSpinBox()
        p.setMaximum(1)
        p.setSingleStep(0.1)
        p.setValue(0.5)

        limit = RangeSpinbox(-1, 1, double=True)
        limit.set_value((-0.2, 0.2))

        self.params = {
            'p': p,
            'limit': limit
        }

    @property
    def name(self):
        return 'RandomContrast'

    @property
    def description(self):
        return 'Случайно изменяет контраст изображения'


class ARandomContrast(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(ARandomContrast, self).__init__(menu_container, config, ARandomContrastMenu, 'RandomContrast',
                                              A.RandomContrast)


class AResizeMenu(PrehandleLayerMenu):
    def parameters(self):
        height = QSpinBox()
        height.setValue(32)
        height.setMaximum(10000)

        width = QSpinBox()
        width.setValue(32)
        width.setMaximum(10000)

        interpolation = QSpinBox()
        interpolation.setMaximum(4)

        self.params = {
            'height': height,
            'width': width,
            'interpolation': interpolation
        }

    @property
    def name(self):
        return 'Resize'

    @property
    def description(self):
        return 'Изменяет размер изображения'


class AResize(PrehandleLayer):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super(AResize, self).__init__(menu_container, config, AResizeMenu, 'Resize', A.Resize)
