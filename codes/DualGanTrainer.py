import logging
import warnings
from typing import Any, Dict, List, Tuple
# from torcheeg.datasets.constants.motor_imagery.strokes import *
from strokes import *
from strokesdict import *
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from tensorboardX import SummaryWriter
from torcheeg.trainers.generative.utils import FrechetInceptionDistance
from torcheeg.utils import plot_raw_topomap,plot_feature_topomap

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader

log = logging.getLogger('torcheeg')






def gradient_penalty(model, real, fake, *args, **kwargs):
    device = real.device
    real = real.data
    fake = fake.data
    alpha = torch.rand(real.size(0), *([1] * (len(real.shape) - 1))).to(device)
    inputs = alpha * real + ((1 - alpha) * fake)
    inputs.requires_grad_()

    outputs = model(inputs, *args, **kwargs)
    gradient = autograd.grad(outputs=outputs,
                             inputs=inputs,
                             grad_outputs=torch.ones_like(outputs).to(device),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
    gradient = gradient.flatten(1)
    return ((gradient.norm(2, dim=1) - 1)**2).mean()


class DFTrainer(pl.LightningModule):


    def __init__(self,
                 generator1: nn.Module,
                 generator2: nn.Module,
                 discriminator1: nn.Module,
                 discriminator2: nn.Module, 
                 generator_lr: float = 1e-4,
                 discriminator_lr: float = 1e-4,
                 consistent_loss_lr: float = 20,
                 weight_decay: float = 0.0,
                 weight_gradient_penalty: float = 1.0,
                 latent_channels: int = None,
                 channel_list= STROKEPATIENTSMI_CHANNEL_LIST,
                 devices: int = 1,
                 show_imgs: bool = False,
                 log_path = "\tmp",
                 accelerator: str = "cpu",
                 metrics: List[str] = [],
                 metric_extractor: nn.Module = None,
                 metric_classifier: nn.Module = None,
                 metric_num_features: int = None):
        super().__init__()
        self.automatic_optimization = False
        self.channel_list = channel_list
        self.generator1 = generator1
        self.discriminator1 = discriminator1

        self.generator2 = generator2
        self.discriminator2 = discriminator2

        self.consistent_loss_lr = consistent_loss_lr
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.weight_decay = weight_decay
        self.weight_gradient_penalty = weight_gradient_penalty

        if hasattr(generator1, 'in_channels') and latent_channels is None:
            warnings.warn(
                f'No latent_channels specified, use generator.in_channels ({generator1.in_channels}) as latent_channels.'
            )
            latent_channels = generator1.in_channels
        
        if hasattr(generator2, 'in_channels') and latent_channels is None:
            warnings.warn(
                f'No latent_channels specified, use generator.in_channels ({generator2.in_channels}) as latent_channels.'
            )
            latent_channels = generator2.in_channels

        assert not latent_channels is None, 'The latent_channels should be specified.'
        self.latent_channels = latent_channels

        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics

        self.bce_fn = nn.BCEWithLogitsLoss()
        self.consistent_loss_fn = nn.L1Loss()

        self.metric_extractor = metric_extractor
        self.metric_classifier = metric_classifier
        self.metric_num_features = metric_num_features
        self.init_metrics(metrics)

        
        self.writer = SummaryWriter(log_dir=log_path,
                       comment='Train GAN')
        self.show_img = show_imgs

    def init_metrics(self, metrics) -> None:
        self.train_g_loss = torchmetrics.MeanMetric()
        self.train_d_loss = torchmetrics.MeanMetric()

        self.val_g_loss = torchmetrics.MeanMetric()
        self.val_d_loss = torchmetrics.MeanMetric()

        self.test_g_loss = torchmetrics.MeanMetric()
        self.test_d_loss = torchmetrics.MeanMetric()

        if 'fid' in metrics:
            assert not self.metric_extractor is None, 'The metric_extractor should be specified.'
            if hasattr(self.metric_extractor,
                       'in_channels') and self.metric_num_features is None:
                warnings.warn(
                    f'No metric_num_features specified, use metric_extractor.in_channels ({self.metric_extractor.in_channels}) as metric_num_features.'
                )
                self.metric_num_features = self.metric_extractor.in_channels
            assert not self.metric_num_features is None, 'The metric_num_features should be specified.'
            self.test_fid = FrechetInceptionDistance(self.metric_extractor,
                                                     self.metric_num_features)

        if 'is' in metrics:
            assert not self.metric_extractor is None, 'The metric_classifier should be specified.'
            self.test_is = InceptionScore(self.metric_classifier)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             inference_mode=False,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, _ = batch
        latent = torch.normal(mean=0,
                              std=1,
                              size=(x.shape[0], self.latent_channels))
        latent = latent.type_as(x)
        return self(latent)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx_: int) -> torch.Tensor:
        x,[y_task,y_side] = batch

        generator_optimizer, discriminator_optimizer1, discriminator_optimizer2 = self.optimizers()

        x1 = x[y_side==0]
        x2 = x[y_side==1]
        g_loss = torch.zeros(1,requires_grad=True).cuda()   
        # train generator
        self.toggle_optimizer(generator_optimizer,0)
        if x1.any():
               
                gen_x1 = self.generator1(x1)
                g_loss = g_loss - torch.mean(self.discriminator2(gen_x1)) * self.generator_lr
                x1_rec = self.generator2(gen_x1)
                g_loss = g_loss + self.consistent_loss_lr * self.consistent_loss_fn(x1_rec, x1)
        if x2.any():
                
                gen_x2 = self.generator2(x2)
                g_loss = g_loss - torch.mean(self.discriminator1(gen_x2)) * self.generator_lr
                x2_rec = self.generator1(gen_x2)
                g_loss = g_loss + self.consistent_loss_lr * self.consistent_loss_fn(x2_rec, x2)
        
        g_loss.backward()
        
        generator_optimizer.step()
        generator_optimizer.zero_grad()
       

        self.untoggle_optimizer(0)

        # train discriminator
        # self.toggle_optimizer(discriminator_optimizer1,1)
        # self.toggle_optimizer(discriminator_optimizer2,2)

        real_loss= torch.zeros(1,requires_grad=True).cuda()
        fake_loss= torch.zeros(1,requires_grad=True).cuda()
        real_loss2= torch.zeros(1,requires_grad=True).cuda()
        fake_loss2= torch.zeros(1,requires_grad=True).cuda()

        if x1.any():
            real_loss = self.discriminator1(x1)
            fake_loss2 = self.discriminator2(gen_x1.detach())

        #gp_term = gradient_penalty(self.discriminator1, x1, gen_x2)
        
        if x2.any():
            real_loss2 = self.discriminator2(x2)
            fake_loss = self.discriminator1(gen_x2.detach())
        
        d_loss = -torch.mean(real_loss) + torch.mean(fake_loss) 
        

        d_loss.backward()
       
        d_loss2 = -torch.mean(real_loss2) + torch.mean(
            fake_loss2)
       
        d_loss2.backward()
      
        discriminator_optimizer1.step()
        discriminator_optimizer1.zero_grad()
        discriminator_optimizer2.step()
        discriminator_optimizer2.zero_grad()
        # self.untoggle_optimizer(1)
        # self.untoggle_optimizer(2)

        self.log("train_g_loss",
                 self.train_g_loss(g_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("train_d_loss",
                 self.train_d_loss((d_loss+d_loss2)/2),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        

    def on_train_epoch_end(self) -> None:
        self.log("train_g_loss",
                 self.train_g_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("train_d_loss",
                 self.train_d_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.train_g_loss.reset()
        self.train_d_loss.reset()

    
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x,[y_task,y_side] = batch

        

        x1 = x[y_side==0]
        y1 = y_task[y_side == 0]
        x2 = x[y_side==1]
        y2 = y_task[y_side==1]
        
        g_loss = torch.zeros(1).cuda()   
        

        if x1.any():
                
                gen_x1 = self.generator1(x1)
                g_loss -= torch.mean(self.discriminator2(gen_x1)) * self.generator_lr
                x1_rec = self.generator2(gen_x1)
                g_loss += self.consistent_loss_lr * self.consistent_loss_fn(x1_rec, x1)
                vis_batch = 100
                if batch_idx % vis_batch == 0 and self.show_img:
                        signal = gen_x1[0][0].cpu()
                        vis_channel_list = self.channel_list
                        top_img = plot_raw_topomap(
                                    torch.tensor(signal),
                                    channel_list=vis_channel_list,
                                    sampling_rate=128,
                                    )
                        self.writer.add_image(f'{batch_idx}/gen_1-{y1[0]}',
                             top_img,
                             batch_idx//vis_batch,
                             dataformats='HWC')
        if x2.any():
               
                gen_x2 = self.generator2(x2)
                g_loss -= torch.mean(self.discriminator1(gen_x2)) * self.generator_lr
                x2_rec = self.generator1(gen_x2)
                g_loss += self.consistent_loss_lr * self.consistent_loss_fn(x2_rec, x2)
    
                vis_batch = 100
                if batch_idx % vis_batch == 0 and self.show_img:
                        signal = gen_x2[0][0].cpu()
                        vis_channel_list = self.channel_list
                        top_img = plot_raw_topomap(
                                    torch.tensor(signal),
                                    channel_list=vis_channel_list,
                                    sampling_rate=128,
                                    )
                        self.writer.add_image(f'{batch_idx}/gen_1-{y2[0]}',
                             top_img,
                             batch_idx//vis_batch,
                             dataformats='HWC')
       

        real_loss= torch.zeros(1,requires_grad=True).cuda()
        fake_loss= torch.zeros(1,requires_grad=True).cuda()
        real_loss2= torch.zeros(1,requires_grad=True).cuda()
        fake_loss2= torch.zeros(1,requires_grad=True).cuda()
        if x1.any():
            real_loss = self.discriminator1(x1)
            fake_loss2 = self.discriminator2(gen_x1.detach())

        #gp_term = gradient_penalty(self.discriminator1, x1, gen_x2)
        
        if x2.any():
            real_loss2 = self.discriminator2(x2)
            fake_loss = self.discriminator1(gen_x2.detach())
        
        d_loss = -torch.mean(real_loss) + torch.mean(fake_loss) 
        d_loss2 = -torch.mean(real_loss2) + torch.mean(fake_loss2)
       

        self.log("val_g_loss",
                 self.train_g_loss(g_loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        self.log("val_d_loss",
                 self.train_d_loss((d_loss+d_loss2)/2),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)
        

    def on_validation_epoch_end(self) -> None:
        self.log("val_g_loss",
                 self.val_g_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        self.log("val_d_loss",
                 self.val_d_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)

        # print the metrics
        str = "\n[VAL] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        log.info(str + '\n')

        # reset the metrics
        self.val_g_loss.reset()
        self.val_d_loss.reset()


    def configure_optimizers(self):

        g_parameters = list(self.generator1.parameters())+list(self.generator2.parameters())
        g_trainable_parameters = list(
            filter(lambda p: p.requires_grad, g_parameters))
        generator_optimizer = torch.optim.Adam(  g_trainable_parameters,
                                               lr=self.generator_lr,
                                               weight_decay=self.weight_decay)
        
        d_parameters =  list(self.discriminator1.parameters())
        d_trainable_parameters = list(
            filter(lambda p: p.requires_grad, d_parameters))
        
        discriminator_optimizer1 = torch.optim.Adam(
            d_trainable_parameters,
            lr=self.discriminator_lr,
            weight_decay=self.weight_decay)
        
        
        d_parameters_2 =  list(self.discriminator2.parameters())
        d_trainable_parameters_2 = list(
            filter(lambda p: p.requires_grad, d_parameters_2))


        discriminator_optimizer2 = torch.optim.Adam(
            d_trainable_parameters_2,
            lr=self.discriminator_lr,
            weight_decay=self.weight_decay)
        
        return [generator_optimizer, discriminator_optimizer1, discriminator_optimizer2], []

