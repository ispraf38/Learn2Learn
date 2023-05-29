from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from ProjectWindow.utils import Config, MenuWidget, WidgetWithMenu, MenuContainer
from ProjectWindow.ModelTab.LossWidget.ParameterLibrary.base_parameter_widget import BaseParameterMenu,\
    BaseParameterWidget
from utils import FixedMultiSpinbox

from torch import optim


class AdamWOptimMenu(BaseParameterMenu):
    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def name(self):
        return 'AdamW'

    @property
    def description(self):
        return 'Алгоритм AdamW'

    def parameters(self):
        lr = QDoubleSpinBox()
        lr.setMaximum(1)
        lr.setDecimals(6)
        lr.setValue(1e-3)

        betas = FixedMultiSpinbox(2, True)
        betas.set_value((0.9, 0.999))

        eps = QDoubleSpinBox()
        eps.setDecimals(12)
        eps.setValue(1e-8)

        weight_decay = QDoubleSpinBox()
        weight_decay.setValue(1e-2)

        amsgrad = QCheckBox()
        amsgrad.setChecked(False)

        maximize = QCheckBox()
        maximize.setChecked(False)

        capturable = QCheckBox()
        capturable.setChecked(False)

        differentiable = QCheckBox()
        differentiable.setChecked(False)

        self.params = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
            'maximize': maximize,
            # 'capturable': capturable,
            # 'differentiable': differentiable
        }


class AdamWOptimWidget(BaseParameterWidget):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, optim.AdamW, AdamWOptimMenu)

    def compile(self, model_params):
        self.param_function = self.function(model_params, **self.get_params())
        return self.param_function


class SGDOptimMenu(BaseParameterMenu):
    def __init__(self, config: Config):
        super().__init__(config)

    @property
    def name(self):
        return 'SGD'

    @property
    def description(self):
        return 'Стохастический градиентный спуск'

    def parameters(self):
        lr = QDoubleSpinBox()
        lr.setMaximum(1)
        lr.setDecimals(6)
        lr.setValue(1e-3)

        momentum = QDoubleSpinBox()
        momentum.setValue(0)

        weight_decay = QDoubleSpinBox()
        weight_decay.setValue(1e-2)

        dampening = QDoubleSpinBox()
        dampening.setValue(0)

        nesterov = QCheckBox()
        nesterov.setChecked(False)

        maximize = QCheckBox()
        maximize.setChecked(False)

        differentiable = QCheckBox()
        differentiable.setChecked(False)

        self.params = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'dampening': dampening,
            'nesterov': nesterov,
            'maximize': maximize,
            # 'differentiable': differentiable
        }


class SGDOptimWidget(BaseParameterWidget):
    def __init__(self, menu_container: MenuContainer, config: Config):
        super().__init__(menu_container, config, optim.SGD, SGDOptimMenu)

    def compile(self, model_params):
        self.param_function = self.function(model_params, **self.get_params())
        return self.param_function
