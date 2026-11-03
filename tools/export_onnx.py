#!/usr/bin/env python3
"""Exporta modelo.pt (ConvNeXt-T de entrenamiento_fuerza_normal) a ONNX.

El grafo incluye la normalización ImageNet de la entrada y la desnormalización de la salida
(y·sigma + mu), así que: entrada float32 RGB en [0,1] NCHW 1x3x288x384 → salida = fuerza normal en N.
Uso: python export_onnx.py modelo.pt ../models/normal_force.onnx
"""
import sys
import numpy as np
import timm
import torch
from torch import nn

ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
red = timm.create_model(ck['modelo'], pretrained=False, num_classes=1)
red.load_state_dict(ck['estado'])
red.eval()


class Envuelto(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = red
        self.register_buffer('m', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mu, self.sigma = float(ck['mu']), float(ck['sigma'])

    def forward(self, x):
        return self.red((x - self.m) / self.s) * self.sigma + self.mu


m = Envuelto().eval()
x = torch.rand(1, 3, ck['alto'], ck['ancho'])
torch.onnx.export(m, x, sys.argv[2], input_names=['image'], output_names=['force_n'], opset_version=17,
                  dynamo=False)
# Metadatos: el nodo lee el tamaño de la forma de entrada; μ/σ y el origen quedan documentados en el propio modelo
import onnx
mp = onnx.load(sys.argv[2])
for k, v in {'mu': m.mu, 'sigma': m.sigma, 'signo': ck['signo'], 'modelo': ck['modelo'],
             'salida': 'fuerza normal en N (+ empuja la red, - tira)', 'origen': sys.argv[1]}.items():
    mp.metadata_props.add(key=k, value=str(v))
onnx.save(mp, sys.argv[2])

import onnxruntime as ort
s = ort.InferenceSession(sys.argv[2], providers=['CPUExecutionProvider'])
a = s.run(None, {'image': x.numpy()})[0]
b = m(x).detach().numpy()
print('mu', m.mu, 'sigma', m.sigma, 'signo', ck['signo'], 'tam', ck['ancho'], ck['alto'], '| diff max', np.abs(a - b).max())
