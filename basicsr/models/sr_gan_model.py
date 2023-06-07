import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr.models.archs.arch_util import NetD
from torch.autograd import Variable

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class SRGanModel(BaseModel):
    """Base SR Gan model for single image super-resolution."""

    def __init__(self, opt):
        super(SRGanModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # 定义判别器
        self.net_d = NetD()
        self.net_d = self.model_to_device(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def load_network(self, net_g, net_d, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net_g = self.get_bare_model(net_g)
        net_d = self.get_bare_model(net_d)
        logger.info(
            f'Loading model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)

        if param_key is not None:
            load_net_g = load_net[param_key]["net_g"]
            load_net_d = load_net[param_key]["net_d"]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net_g).items():
            if k.startswith('module.'):
                load_net_g[k[7:]] = v
                load_net_g.pop(k)
        self._print_different_keys_loading(net_g, load_net_g, strict)
        net_g.load_state_dict(load_net_g, strict=strict)

        for k, v in deepcopy(load_net_d).items():
            if k.startswith('module.'):
                load_net_d[k[7:]] = v
                load_net_d.pop(k)
        self._print_different_keys_loading(net_d, load_net_d, strict)
        net_d.load_state_dict(load_net_d, strict=strict)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # 训练判别器
        self.net_d.zero_grad()

        D_result = self.net_d(self.gt[:, 0, ...]).squeeze()
        D_real_loss = -D_result.mean()

        G_result = self.net_g(self.lq)
        D_result = self.net_d(G_result[:, 0, ...]).squeeze()

        D_fake_loss = D_result.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        self.optimizer_d.step()

        # gradient penalty
        self.net_d.zero_grad()
        alpha = torch.rand(self.gt.size(0), 1, 1, 1)
        alpha1 = alpha.to(self.device).expand_as(self.gt[:, 0, ...])
        interpolated1 = Variable(alpha1 * self.gt[:, 0, ...] + (1 - alpha1) * G_result[:, 0, ...], requires_grad=True)
        
        out = self.net_d(interpolated1).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated1,
                                   grad_outputs=torch.ones(out.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = 10 * d_loss_gp

        gp_loss.backward()
        self.optimizer_d.step()


        # 训练生成器
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.gt.ndim == 5:
                l_pix = self.cri_pix(self.output, self.gt)
            elif self.gt.ndim == 4:
                t = self.output.size(1)
                l_pix = self.cri_pix(self.output[:, t//2], self.gt)
            else:
                print("output and gt shape error!")
            
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            
            if len(self.output.shape) == len(self.gt.shape):
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            elif len(self.output.shape) == len(self.gt.shape) + 1:
                l_percep, l_style = self.cri_perceptual(self.output[:, self.output.shape[2]//2], self.gt)
            else:
                print("output and gt shape error!")
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            output = self.net_g(self.lq)
            if type(output) is tuple:
                self.output = output[0]
            else:
                self.output = output
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['key'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation on {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if self.gt.ndim == 5:
            out_dict['result'] = self.output.detach().cpu()
        else:
            t = self.output.size(1)
            out_dict['result'] = self.output[:, t//2].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
