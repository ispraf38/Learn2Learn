from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, InputLayer, OutputLayer

import torch.nn as nn
from typing import List, Tuple
from loguru import logger

class Model(nn.Module):
    def __init__(self, layers: List[Layer], input_layer: InputLayer, output_layer: OutputLayer,):
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        for layer in layers:
            self.__setattr__(f'_{str(layer)}_module', layer.F)

        self.layers = {
            layer: {
                'outputs': {},
                'handled': False
            } for layer in layers
        }
        layers_to_run = [self.input_layer]
        while layers_to_run:
            layer = layers_to_run.pop(0)
            logger.debug(layer)
            output, ok = layer.forward_test({i: self.layers[l]['outputs'][o]
                                             for (i, (l, o)) in layer.previous_layers.items()})
            self.layers[layer]['handled'] = True
            self.layers[layer]['outputs'] = output
            for l in layers:
                if not self.layers[l]['handled']:
                    previous_handled = True
                    for p_l, _ in l.previous_layers.values():
                        if p_l is not None:
                            previous_handled = previous_handled and\
                                               self.layers[p_l]['handled'] and p_l.state.state == 'ok'
                        else:
                            previous_handled = False
                    if previous_handled:
                        layers_to_run.append(l)
            if not ok:
                layer.state.error()
                layer.update()
            else:
                layer.state.ok()
                layer.update()

