import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig, \
    ImageGranularity
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE, IMAGE_INPUT
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import ActivationExtractor


class DataHolder:
    def __init__(self,
                 init_dataset: DataLoader,
                 num_batches: int,
                 model: Module,
                 data_generation_config: DataGenerationConfig,
                 activation_extractor: ActivationExtractor):
        self.num_batches = num_batches
        self.data_generation_config = data_generation_config
        self.activation_extractor = activation_extractor
        self.batched_images_for_optimization = []
        self.labels = []
        if data_generation_config.image_granularity in [ImageGranularity.AllImages]:
            self.use_all_data_stats = True
        else:
            self.use_all_data_stats = False
        for data_input in init_dataset:
            if isinstance(data_input, list):
                # This is the case in which the data loader holds both images and labels
                batched_images, labels = data_input
            else:
                batched_images = data_input
                labels = torch.randint(1000, [data_input.size(0)])
            self.batched_images_for_optimization.append(batched_images.to(DEVICE))
            self.labels.append(labels.to(DEVICE))
        self.batched_images_for_optimization = torch.concatenate(self.batched_images_for_optimization, dim=0)
        self.optimizer = data_generation_config.optimizer([self.batched_images_for_optimization], lr=data_generation_config.initial_lr)
        self.scheduler = data_generation_config.scheduler(self.optimizer)
        if self.use_all_data_stats:
            self.stats_holder = StatsHolder(self.batched_images_for_optimization.shape[0])
            # run model on images to get current activations
            output = model(self.batched_images_for_optimization)
            # save activation statistics
            self.stats_holder.calc_and_save_stats(
                image_indexes=range(self.batched_images_for_optimization.shape[0]),
                activation_extractor=self.activation_extractor,
                to_differentiate=False)

    def get_images_for_optimization(self, iter_index):
        pass


class StatsHolder:
    def __init__(self, n_images):
        self.mean_per_image = [0 for _ in range(n_images)]
        self.second_moment_per_image = [0 for _ in range(n_images)]

    def calc_and_save_stats(self, image_indexes, activation_extractor, to_differentiate=False):
        for layer_num, hook in enumerate(activation_extractor.hooks):
            bn_layer_name = hook.bn_layer_name
            activations = hook.input
            if to_differentiate is False:
                activations = activations.detach()

            collected_mean = torch.mean(activations, dim=[2, 3])
            collected_2nd_raw_moment = torch.mean(torch.pow(activations, 2.0), dim=[2, 3])

    def get_stats(self, image_indexes):
        pass


class DataStatsHolder:
    def __init__(self, layer_names, n_images):
        self.stats_holder_per_layer = {name: StatsHolder(n_images) for name in layer_names}

    def calc_and_save_stats(self, image_indexes, activation_extractor, to_differentiate=False):
        for layer_num, hook in enumerate(activation_extractor.hooks):
            bn_layer_name = hook.bn_layer_name
            activations = hook.input
            if to_differentiate is False:
                activations = activations.detach()

            collected_mean = torch.mean(activations, dim=[2, 3])
            collected_2nd_raw_moment = torch.mean(torch.pow(activations, 2.0), dim=[2, 3])

    def get_stats(self, image_indexes):
        pass



class BatchOptimization:
    def __init__(self, images, labels, optimizer):
        self.labels = labels.to(DEVICE)
        self.images = images.to(DEVICE)
        self.images.requires_grad = True
        self.optimizer = optimizer


    # def set_optimizer_scheduler(self, init_lr=0.05, num_iterations=500, optimizer_type='adam'):
    #     self.init_lr = init_lr
    #     if optimizer_type == 'adam':
    #         self.optimizer = optim.Adam([self.gaussian_data], lr=self.init_lr)
    #     elif 'radam' in optimizer_type:
    #         self.optimizer = optim.RAdam([self.gaussian_data], lr=self.init_lr)
    #     else:
    #         self.optimizer = optim.SGD([self.gaussian_data], lr=self.init_lr, momentum=0.9)
    #     if optimizer_type == 'adam':
    #         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    #                                                               min_lr=1e-4,
    #                                                               verbose=False,
    #                                                               patience=200)
    #
    #     else:
    #         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
    #                                                    verbose=False,
    #                                                    gamma=0.5 if '1' in optimizer_type else 0.1,
    #                                                    step_size=int(num_iterations / 12)if '1' in optimizer_type else int(num_iterations/4))
    #     self.acc_grad = np.zeros_like(self.gaussian_data.detach().cpu().numpy())
    #
    # def reset_optimizer(self):
    #     self.optimizer = optim.Adam([self.gaussian_data], lr=self.init_lr)
    #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
    #                                                           min_lr=1e-4,
    #                                                           verbose=True,
    #                                                           patience=100)
    #
    # def get_data(self):
    #     self.gaussian_data.requires_grad = False
    #     return self.gaussian_data.detach().clone()
    #
    # def add_noise(self, noise_factor, noise_sample):
    #     self.gaussian_data.data = (1 - noise_factor) * self.gaussian_data.data + noise_factor * noise_sample.data
    #
    # def save_acc_grad(self):
    #     self.acc_grad += np.abs(self.gaussian_data.grad.detach().cpu().numpy())

    def remove(self):
        del self.gaussian_data
        del self.optimizer
        del self.scheduler
        torch.cuda.empty_cache()
