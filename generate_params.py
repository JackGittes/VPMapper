import torch
from utils.sim_tool import Simulation
from utils.pipeline import post_processing
from model.mbv2 import MBV2Face



# net = MBV2Face()
# net.load_state_dict(torch.load('./model/039.pth')['state_dict'])
#
# post_processing(model=net, pth_path='result/pth/quantized.pth',
#                 info_path='report/info.csv', bit_width=8)
#
# sim = Simulation()
#
# sim.convert(net)


from model.shipnet import ShipNet
net = ShipNet()

# net.load_state_dict(torch.load('./model/model_029.pth'))

post_processing(model=net, pth_path='result/pth/quantized.pth',
                info_path='report/info.csv', bit_width=8)

sim = Simulation()

sim.convert(net)
