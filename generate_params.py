import torch
from utils.sim_tool import Simulation
from utils.pipeline import post_processing
from model.mbv2 import MBV2Face
from model.backbone.mobilenetv2_q import InvertedResidual


net = MBV2Face()
net.load_state_dict(torch.load('./model/params.pth')['state_dict'])

for m in net.modules():
    if isinstance(m, InvertedResidual):
        m.turn_on_add()

post_processing(model=net, pth_path='result/pth/quantized.pth',
                info_path='report/info.csv', bit_width=8)
#
# sim = Simulation()
#
# sim.convert(net)
#
#
# # from model.shipnet import ShipNet
# # net = ShipNet()
# #
# # # net.load_state_dict(torch.load('./model/model_029.pth'))
# #
# # post_processing(model=net, pth_path='result/pth/quantized.pth',
# #                 info_path='report/info.csv', bit_width=8)
# #
# # sim = Simulation()
# #
# # sim.convert(net)

# from torchvision.models import mobilenet_v2
# from model.mobilenet_v2_q import mobilenet_v2, InvertedResidual
# net = mobilenet_v2(pretrained=True)
#
# for m in net.modules():
#     if isinstance(m, InvertedResidual):
#         m.turn_on_add()
#
# post_processing(model=net, pth_path='result/pth/quantized.pth',
#                 info_path='report/info.csv', bit_width=8)

sim = Simulation()

sim.convert(net)
