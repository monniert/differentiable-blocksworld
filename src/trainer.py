import argparse
from pathlib import Path
import time
import warnings

from toolz import merge, keyfilter, valmap
import numpy as np
from pytorch3d.structures import Meshes
import torch
from torchvision.transforms import functional as F

from dataset import create_train_val_test_loader
from model import create_model
from optimizer import create_optimizer
from scheduler import create_scheduler
from utils import use_seed, path_exists, path_mkdir, load_yaml, dump_yaml
from utils.dtu_eval import evaluate_mesh
from utils.image import ImageLogger
from utils.logger import create_logger, print_log, print_warning
from utils.metrics import Metrics
from utils.path import CONFIGS_PATH, RUNS_PATH, DATASETS_PATH
from utils.plot import plot_lines, VisdomVisualizer, get_fancy_cmap, create_visualizer
from utils.pytorch import get_torch_device, torch_to


LIGHT_MEMORY_RESULTS = True
LOG_FMT = 'Epoch [{}/{}], Iter [{}/{}], {}'.format
N_VIZ_SAMPLES = 4
torch.backends.cudnn.benchmark = True  # XXX accelerate training if fixed input size for each layer
warnings.filterwarnings('ignore')
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    """Pipeline to train a model on a particular dataset, both specified by a config cfg."""
    @use_seed()
    def __init__(self, cfg, run_dir, gpu=None, rank=None, world_size=None):
        self.run_dir = path_mkdir(run_dir)
        self.device = get_torch_device(gpu, verbose=True)
        self.train_loader, self.val_loader, self.test_loader = create_train_val_test_loader(cfg)
        self.model = create_model(cfg, self.train_loader.dataset.img_size).to(self.device)
        self.optimizer = create_optimizer(cfg, self.model)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        self.epoch_start, self.batch_start = 1, 1
        self.n_epoches, self.n_batches = cfg['training'].get('n_epoches'), len(self.train_loader)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.load_from(cfg)
        print_log(f'Training state: epoch={self.epoch_start}, batch={self.batch_start}, lr={self.cur_lr}')

        # Logging metrics
        append = self.epoch_start > 1
        self.train_stat_interval = cfg['training']['train_stat_interval']
        self.val_stat_interval = cfg['training']['val_stat_interval']
        self.save_epoches = cfg['training'].get('save_epoches', [])
        names = self.model.loss_names if hasattr(self.model, 'loss_names') else ['loss']
        self.train_metrics = Metrics(*['time/img'] + names, log_file=self.run_dir / 'train_metrics.tsv', append=append)
        names = [f'alpha{k}' for k in range(self.model.n_blocks)]
        self.val_metrics = Metrics(*names, log_file=self.run_dir / 'val_metrics.tsv', append=append)

        # Logging visuals
        with use_seed(12345):
            samples, labels = next(iter(self.val_loader if len(self.val_loader) > 0 else self.train_loader))
        self.viz_samples = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], samples)
        self.viz_labels = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], labels)
        out_ext = 'jpg' if LIGHT_MEMORY_RESULTS else 'png'
        self.rec_logger = ImageLogger(self.run_dir / 'reconstructions', self.viz_samples, out_ext=out_ext)
        self.rec2_logger = ImageLogger(self.run_dir / 'reconstructions_hard', self.viz_samples, out_ext=out_ext)
        self.rec3_logger = ImageLogger(self.run_dir / 'reconstructions_syn', self.viz_samples, out_ext='png')
        self.txt_logger = ImageLogger(self.run_dir / 'txt_blocks', out_ext=out_ext)
        if self.with_training:
            self.visualizer = create_visualizer(cfg, self.run_dir)
        else:  # no visualizer (VisdomVisualizer without port does nothing) if eval only
            self.visualizer = VisdomVisualizer(None, self.run_dir)

    @property
    def with_training(self):
        return self.epoch_start < self.n_epoches

    @property
    def dataset(self):
        return self.train_loader.dataset

    def load_from(self, cfg):
        pretrained, resume = cfg['training'].get('pretrained'), cfg['training'].get('resume')
        assert not (pretrained is not None and resume is not None)
        tag = pretrained or resume
        if tag is not None:
            path = path_exists(RUNS_PATH / self.dataset.name / tag / 'model.pkl')
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            if resume is not None:
                if checkpoint['batch'] == self.n_batches:
                    self.epoch_start, self.batch_start = checkpoint['epoch'] + 1, 1
                else:
                    self.epoch_start, self.batch_start = checkpoint['epoch'], checkpoint['batch'] + 1
                self.model.set_cur_epoch(checkpoint['epoch'])
                print_log(f'epoch_start={self.epoch_start}, batch_start={self.batch_start}')
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                except ValueError:
                    print_warning("ValueError: loaded optim state contains parameters that don't match")
                scheduler_state = keyfilter(lambda k: k in ['last_epoch', '_step_count'], checkpoint['scheduler_state'])
                self.scheduler.load_state_dict(scheduler_state)
                self.cur_lr = self.scheduler.get_last_lr()[0]
                print_log(f'scheduler state_dict: {self.scheduler.state_dict()}')
            print_log(f'Checkpoint {tag} loaded')

    @use_seed()
    def run(self):
        cur_iter = (self.epoch_start - 1) * self.n_batches + self.batch_start
        log_iters = np.unique(
            np.geomspace(1, self.n_batches * self.n_epoches, 500).astype(int))
        self.log_dataset_visualization(cur_iter)
        self.log_visualizations(cur_iter)
        for epoch in range(self.epoch_start, self.n_epoches + 1):
            batch_start = self.batch_start if epoch == self.epoch_start else 1
            for batch, (images, labels) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue

                self.run_single_batch_train(images, labels)

                if cur_iter % self.train_stat_interval == 0:
                    self.log_train_metrics(cur_iter, epoch, batch)

                if cur_iter % self.val_stat_interval == 0 or cur_iter in log_iters:
                    self.run_val_and_log(cur_iter, epoch, batch)
                    self.log_visualizations(cur_iter)
                    self.save(epoch=epoch, batch=batch)

                cur_iter += 1
            self.step(epoch + 1, batch=1)
            if epoch in self.save_epoches:
                self.save(epoch=epoch, batch=batch, checkpoint=True)

        N, B = (self.n_epoches, self.n_batches) if self.with_training else (self.epoch_start, self.batch_start)
        self.save(epoch=N, batch=B)
        self.save_metric_plots()
        self.evaluate()
        print_log('Training over')

    def run_single_batch_train(self, images, labels):
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(torch_to(images, self.device), torch_to(labels, self.device))
        loss['total'].backward()
        dict_loss = {f'loss_{k}': v.detach().mean().item() for k, v in loss.items()}
        self.optimizer.step()

        B = len(images['imgs'])
        self.train_metrics.update(merge({'time/img': (time.time() - start_time) / B}, dict_loss), N=B)

    @torch.no_grad()
    def run_val_and_log(self, it, epoch, batch):
        metrics = self.val_metrics
        opacities = self.model.get_opacities()
        if (opacities > 0.01).sum() == 0:
            raise RuntimeError('No more blocks....')
        named_values = [(f'alpha{k}', a.item()) for k, a in enumerate(opacities)]
        metrics.update(dict(named_values))
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'val_metrics: {metrics}')[:1000])
        cmap = get_fancy_cmap()
        colors = (cmap(np.linspace(0, 1, len(named_values) + 1)[1:]) * 255).astype(np.uint8)
        self.visualizer.log_scalars(it, metrics.get_named_values(), title='opacities', colors=colors)
        metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def step(self, epoch, batch):
        self.scheduler.step()
        self.model.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'LR update: lr={lr}'))

    def log_train_metrics(self, it, epoch, batch):
        metrics = self.train_metrics
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'train_metrics: {metrics}')[:1000])
        self.visualizer.log_scalars(it, metrics.get_named_values(lambda s: 'loss' in s), title='train_losses')
        metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    @torch.no_grad()
    def log_visualizations(self, cur_iter):
        self.model.eval()

        self.visualizer.log_model(cur_iter, self.model)

        # Log soft reconstructions
        renders = self.model.predict(self.viz_samples, self.viz_labels)
        self.rec_logger.save(renders, cur_iter)
        imgs = F.resize(self.viz_samples['imgs'], renders.shape[-2:], antialias=True)
        self.visualizer.log_renders(cur_iter, renders, 'recons')

        # Log hard reconstructions with edges
        renders = self.model.predict(self.viz_samples, self.viz_labels, w_edges=True, filter_transparent=True)
        self.rec2_logger.save(renders, cur_iter)
        self.visualizer.log_renders(cur_iter, renders, 'recons_hard_w_edges')

        # Log hard reconstructions wo edges
        renders = self.model.predict(self.viz_samples, self.viz_labels, filter_transparent=True)
        self.visualizer.log_renders(cur_iter, renders, 'recons_hard_wo_edges')

        # Log rendering with synthetic colors
        renders = self.model.predict_synthetic(self.viz_samples, self.viz_labels)
        self.rec3_logger.save(renders, cur_iter)
        self.visualizer.log_renders(cur_iter, renders, 'recons_syn')

        # Log textures
        txt = self.model.get_arranged_block_txt()
        self.txt_logger.save(txt, cur_iter)
        self.visualizer.log_textures(cur_iter, txt[0], 'textures', max_size=256)

    @torch.no_grad()
    def log_dataset_visualization(self, cur_iter):
        self.visualizer.log_dataset(cur_iter, self.dataset)

    def save(self, epoch, batch, checkpoint=False):
        state = {
            'epoch': epoch, 'batch': batch, 'model_name': self.model.name, 'model_kwargs': self.model.init_kwargs,
            'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        name = f'model_{epoch}.pkl' if checkpoint else 'model.pkl'
        torch.save(state, self.run_dir / name)
        print_log(f'Model saved at {self.run_dir / name}')

    @torch.no_grad()
    def save_metric_plots(self):
        self.model.eval()
        df = self.train_metrics.read_log()
        if len(df) == 0:
            print_log('No metrics or plots to save')
            return None

        # Charts
        loss_names = list(filter(lambda col: 'loss' in col, df.columns))
        plot_lines(df, loss_names, title='Loss').savefig(self.run_dir / 'loss.pdf')
        df = self.val_metrics.read_log()
        alpha_names = list(filter(lambda col: 'alpha' in col, df.columns))
        colors = get_fancy_cmap()(np.linspace(0, 1, len(alpha_names) + 1)[1:])
        plot_lines(df, alpha_names, title='Opacity', colors=colors).savefig(self.run_dir / 'opacity.pdf')

        # Images / renderings
        rec = self.model.predict(self.viz_samples, self.viz_labels, w_edges=True)
        self.rec_logger.save(rec)
        self.rec_logger.save_video(rmtree=LIGHT_MEMORY_RESULTS)
        rec = self.model.predict(self.viz_samples, self.viz_labels, filter_transparent=True)
        self.rec2_logger.save(rec)
        self.rec2_logger.save_video(rmtree=LIGHT_MEMORY_RESULTS)
        rec = self.model.predict_synthetic(self.viz_samples, self.viz_labels)
        self.rec3_logger.save(rec)
        self.rec3_logger.save_video(rmtree=LIGHT_MEMORY_RESULTS)
        self.txt_logger.save(self.model.get_arranged_block_txt())
        self.txt_logger.save_video(rmtree=LIGHT_MEMORY_RESULTS)
        print_log('Metrics and plots saved')

    def evaluate(self):
        self.model.eval()

        # qualitative
        out = path_mkdir(self.run_dir / 'quali_eval')
        self.model.qualitative_eval(self.test_loader, self.device, path=out)

        # quantitative
        scores = self.model.quantitative_eval(self.test_loader, self.device, hard_inference=True)
        print_log('final_scores: ' + ', '.join(["{}={:.5f}".format(k, v) for k, v in scores.items()]))
        with open(self.run_dir / 'final_scores.tsv', mode='w') as f:
            f.write("\t".join(scores.keys()) + "\n")
            f.write("\t".join(map('{:.5f}'.format, scores.values())) + "\n")

        # official DTU eval
        if self.dataset.name == 'dtu':
            scan_id = int(self.dataset.tag.replace('scan', ''))
            scale = self.dataset.scale_mat.to(self.device)

            # Blocks only
            scene = self.model.build_blocks(filter_transparent=True, as_scene=True)
            verts, faces = scene.get_mesh_verts_faces(0)
            scene = Meshes((verts @ scale[:3, :3] + scale[:3, 3])[None], faces[None])
            evaluate_mesh(scene, scan_id, DATASETS_PATH / 'DTU', self.run_dir, save_viz=False)

            # Blocks + floor
            # scene = self.model.build_scene(filter_transparent=True, masked_txt=True, w_bkg=False, reduce_floor=True)
            # verts, faces = scene.get_mesh_verts_faces(0)
            # scene = Meshes((verts @ scale[:3, :3] + scale[:3, 3])[None], faces[None])
            # evaluate_mesh(scene, scan_id, DATASETS_PATH / 'DTU', self.run_dir, suffix="_wfloor", save_viz=False)

        print_log('Evaluation over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to train a NN model specified by a YML config')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-c', '--config', nargs='?', type=str, required=True, help='Config file name')
    parser.add_argument('-d', '--default', nargs='?', type=str, help='Default config file name')
    args = parser.parse_args()
    assert args.tag != '' and args.config != ''

    default_path = None if (args.default is None or args.default == '') else CONFIGS_PATH / args.default
    cfg = load_yaml(CONFIGS_PATH / args.config, default_path)
    seed, dataset = cfg['training'].get('seed', 4321), cfg['dataset']['name']
    if (RUNS_PATH / dataset / args.tag).exists():
        run_dir = RUNS_PATH / dataset / args.tag
    else:
        run_dir = path_mkdir(RUNS_PATH / dataset / args.tag)
    create_logger(run_dir)
    dump_yaml(cfg, run_dir / Path(args.config).name)

    print_log(f'Trainer init: config_file={args.config}, run_dir={run_dir}')
    trainer = Trainer(cfg, run_dir, seed=seed)
    trainer.run(seed=seed)
