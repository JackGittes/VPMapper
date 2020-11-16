from zoo.mobilefacev2.quantize import search_quantization
from zoo.mobilefacev2.sim_generator import script_params_converter


class Convert:
    NetReg = ['MobileNetv2', 'MobileFaceNet', 'ShipNet']

    def __init__(self, net_type, param_path, data_path):
        assert net_type in self.NetReg
        self.net = net_type
        self.param_path = param_path
        self.data_path = data_path

    def quantize_with_config(self):
        if self.net == self.NetReg[0]:
            pass
        elif self.net == self.NetReg[1]:
            search_quantization(params_path=self.param_path,
                                calibration_path=self.data_path)
        elif self.net == self.NetReg[2]:
            pass

    def convert_with_config(self):
        if self.net == self.NetReg[0]:
            pass
        elif self.net == self.NetReg[1]:
            script_params_converter()
        elif self.net == self.NetReg[2]:
            pass

