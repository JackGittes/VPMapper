import torch
from utils.pipeline import post_processing
from utils.sim_tool import Simulation
from zoo.mobilefacev2.mbv2 import MBV2Face, InvertedResidual


def script_params_converter(param_path):
    net = MBV2Face()
    net.load_state_dict(torch.load(param_path)['state_dict'])
    for m in net.modules():
        if isinstance(m, InvertedResidual):
            m.turn_on_add(bit_width=8)

    post_processing(model=net, pth_path='result/pth/quantized.pth',
                    info_path='report/info.csv', bit_width=8)
    sim = Simulation()
    sim.convert(net)
