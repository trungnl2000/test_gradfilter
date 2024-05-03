import logging
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from classification.model import ClassificationModel
from dataloader.pl_dataset import ClsDataset
import torch

logging.basicConfig(level=logging.INFO)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logger.save_dir", default='./runs')
        parser.add_argument("--logger.exp_name", default='test')

    def instantiate_trainer(self, **kwargs):
        if 'fit' in self.config.keys():
            cfg = self.config['fit']
        elif 'validate' in self.config.keys():
            cfg = self.config['validate']
        else:
            cfg = self.config
        logger_name = cfg['logger']['exp_name'] + "_" + cfg['data']['name']
        if 'logger_postfix' in kwargs:
            logger_name += kwargs['logger_postfix']
            kwargs.pop('logger_postfix')
        logger = TensorBoardLogger(cfg['logger']['save_dir'], logger_name)
        kwargs['logger'] = logger
        
        # kwargs['accelerator']='gpu'
        # kwargs['devices']="auto"
        trainer = super(CLI, self).instantiate_trainer(**kwargs)
        return trainer

from util import attach_hooks_for_conv


def run():
    cli = CLI(ClassificationModel, ClsDataset, run=False, save_config_overwrite=True) # Dùng cli để load model và file config cho model
    model = cli.model # Khởi tạo model từ cli
    trainer = cli.trainer # Khởi tạo trainer từ cli
    data = cli.datamodule # Khởi tạo data từ cli
    # logging.info(str(model))
    # train_data_size = torch.tensor([3, data.width, data.height]) # 3 kênh màu, data.width x data.height kích thước
    # logging.info(f"activation size: {model.get_activation_size(consider_active_only=True, unit='KB', train_data_size=train_data_size)}") # Dùng cho kiểu hook mới
    # logging.info(f"FLOPs: {model.compute_Conv2d_flops(consider_active_only=True, unit='MB', train_data_size=train_data_size)}") # Dùng cho kiểu hook mới
    
    # attach_hooks_for_conv(model, True)
    # model.activate_hooks(True)
    # trainer.validate(model, datamodule=data)

    # logging.info(f"activation size: {model.get_activation_size(consider_active_only=True, unit='MB', train_data_size=train_data_size)}") # Dùng cho kiểu hook mới

    trainer.fit(model, data)
    trainer.validate(model, datamodule=data)


run()
