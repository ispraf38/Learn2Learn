from ProjectWindow.ModelTab.ConstructorWidget.LayersLibrary.base_layer import Layer, InputLayer, OutputLayer

import torch.nn as nn
from typing import List, Tuple
from loguru import logger


class Model(nn.Module):
    def __init__(self, layers: List[Layer], input_layer: InputLayer, output_layer: OutputLayer, ):
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
        self.layers_order = []
        label = None
        layers_to_run = [layer for layer in layers if layer.previous_layers == {}]
        while layers_to_run:
            layer = layers_to_run.pop(0)
            logger.debug(layer)
            if layer not in [self.input_layer, self.output_layer]:
                self.layers_order.append(layer)
                output, ok = layer.forward_test({i: self.layers[l]['outputs'][o]
                                                 for (i, (l, o)) in layer.previous_layers.items()})
            elif layer == self.input_layer:
                output, ok = layer.forward_test({i: self.layers[l]['outputs'][o]
                                                 for (i, (l, o)) in layer.previous_layers.items()})
                label = layer.current_label
            elif layer == self.output_layer:
                output, ok = layer.forward_test({i: self.layers[l]['outputs'][o]
                                                 for (i, (l, o)) in layer.previous_layers.items()}, label)
            self.layers[layer]['handled'] = True
            self.layers[layer]['outputs'] = output
            for o, next_layers in layer.next_layers.items():
                for l, i in next_layers:
                    if not self.layers[l]['handled'] and l not in layers_to_run:
                        previous_handled = True
                        for p_l, _ in l.previous_layers.values():
                            if p_l is not None:
                                previous_handled = previous_handled and \
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

    def forward(self, input):
        self.layers[self.input_layer]['outputs']['out'] = input
        for layer in self.layers_order:
            # logger.debug(layer)
            input = {i: self.layers[l]['outputs'][o] for (i, (l, o)) in layer.previous_layers.items()}
            # input = {}
            # for (i, (l, o)) in layer.previous_layers.items():
            #     logger.debug(f'({i}, ({l}, {o}))')
            #     logger.debug(self.layers[l]['outputs'][o])
            #     input[i] = self.layers[l]['outputs'][o]
            try:
                output = layer.forward(input)
            except Exception as e:
                logger.error(e)
                layer.state.error()
                logger.info(layer)
                logger.info({k: v.shape for k, v in input.items()})
                raise e
            self.layers[layer]['outputs'] = output
        return output['out']
