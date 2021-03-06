from collections import OrderedDict

import torch
from abc import abstractmethod, ABC
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset  # noqa F401
from typing import Iterable, Any, Mapping, Dict, List

from catalyst.dl.registry import \
    MODELS, CRITERIONS, OPTIMIZERS, SCHEDULERS, CALLBACKS
from catalyst.dl import utils
from catalyst.dl.callbacks import Callback  # noqa F401
from catalyst.dl.callbacks import \
    LossCallback, OptimizerCallback, SchedulerCallback, CheckpointCallback
from catalyst.dl.fp16 import Fp16Wrap
from catalyst.dl.utils import UtilsFactory
from catalyst.utils.misc import merge_dicts

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


class Experiment(ABC):
    """
    Object containing all information required to run the experiment

    Abstract, look for implementations
    """

    @property
    @abstractmethod
    def logdir(self) -> str:
        pass

    @property
    @abstractmethod
    def stages(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def get_model(self, stage: str) -> _Model:
        pass

    @abstractmethod
    def get_criterion(self, stage: str) -> _Criterion:
        pass

    @abstractmethod
    def get_optimizer(self, stage: str, model) -> _Optimizer:
        pass

    @abstractmethod
    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        pass

    def get_model_stuff(self, model, stage: str):
        criterion = self.get_criterion(stage)
        optimizer = self.get_optimizer(stage, model)
        scheduler = self.get_scheduler(stage, optimizer)
        return criterion, optimizer, scheduler

    @abstractmethod
    def get_callbacks(self, stage: str) -> "List[Callback]":
        pass

    def get_datasets(
        self,
        stage: str,
        **kwargs,
    ) -> "OrderedDict[str, Dataset]":
        raise NotImplementedError

    @abstractmethod
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        raise NotImplementedError

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        raise NotImplementedError


class BaseExperiment(Experiment):
    """
    Super-simple one-staged experiment
        you can use to declare experiment in code
    """

    def __init__(
        self,
        model: _Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "List[Callback]" = None,
        logdir: str = None,
        stage: str = "train",
        criterion: _Criterion = None,
        optimizer: _Optimizer = None,
        scheduler: _Scheduler = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        state_kwargs: Dict = None
    ):
        self._model = model
        self._loaders = loaders
        self._callbacks = callbacks or []

        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._logdir = logdir
        self._stage = stage
        self._num_epochs = num_epochs
        self._valid_loader = valid_loader
        self._main_metric = main_metric
        self._minimize_metric = minimize_metric
        self._verbose = verbose
        self._additional_state_kwargs = state_kwargs or {}

    @property
    def logdir(self):
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return [self._stage]

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        default_params = dict(
            logdir=self.logdir,
            num_epochs=self._num_epochs,
            valid_loader=self._valid_loader,
            main_metric=self._main_metric,
            verbose=self._verbose,
            minimize_metric=self._minimize_metric
        )
        state_params = {
            **self._additional_state_kwargs,
            **default_params
        }
        return state_params

    def get_model(self, stage: str) -> _Model:
        return self._model

    def get_criterion(self, stage: str) -> _Criterion:
        return self._criterion

    def get_optimizer(self, stage: str, model=None) -> _Optimizer:
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer=None) -> _Scheduler:
        return self._scheduler

    def get_callbacks(self, stage: str) -> "List[Callback]":
        return self._callbacks

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        return self._loaders


class SupervisedExperiment(BaseExperiment):

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks = self._callbacks
        if not stage.startswith("infer"):
            default_callbacks = [
                (self._criterion, LossCallback),
                (self._optimizer, OptimizerCallback),
                (self._scheduler, SchedulerCallback),
                ("_default_saver", CheckpointCallback),
            ]

            for key, value in default_callbacks:
                is_already_present = any(
                    isinstance(x, value) for x in callbacks)
                if key is not None and not is_already_present:
                    callbacks.append(value())
        return callbacks


class ConfigExperiment(Experiment):
    STAGE_KEYWORDS = [
        "criterion_params", "optimizer_params", "scheduler_params",
        "data_params", "state_params", "callbacks_params",
    ]

    def __init__(self, config: Dict):
        self._config = config.copy()
        # @TODO: good enough solution?
        self._config["stages"]["state_params"] = merge_dicts(
            self._config["stages"].get("state_params", {}).copy(),
            self._config.get("args", {}).copy()
        )
        self.stages_config = self._prepare_stages_config(config["stages"])
        self._logdir = \
            self._config.get("args", {}).get("logdir", None) \
            or self._prepare_logdir(config)

    def _prepare_stages_config(self, stages_config):
        stages_defaults = {}
        for key in self.STAGE_KEYWORDS:
            stages_defaults[key] = stages_config.pop(key, {})
        for stage in stages_config:
            for key in self.STAGE_KEYWORDS:
                stages_config[stage][key] = merge_dicts(
                    stages_config[stage].get(key, {}).copy(),
                    stages_defaults.get(key, {}).copy()
                )
        return stages_config

    @property
    def logdir(self):
        return self._logdir

    def _prepare_logdir(self, config: Dict):
        raise NotImplementedError

    @property
    def stages(self) -> List[str]:
        stages_keys = list(self.stages_config.keys())
        return stages_keys

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        return self.stages_config[stage]["state_params"]

    def _preprocess_model_for_stage(self, stage: str, model: _Model):
        stage_index = self.stages.index(stage)
        if stage_index > 0:
            checkpoint_path = \
                f"{self.logdir}/checkpoints/best.pth"
            checkpoint = UtilsFactory.load_checkpoint(checkpoint_path)
            UtilsFactory.unpack_checkpoint(checkpoint, model=model)
        return model

    def _postprocess_model_for_stage(self, stage: str, model: _Model):
        return model

    def get_model(self, stage: str) -> _Model:
        model_params = self._config["model_params"]
        fp16 = model_params.pop("fp16", False)

        model = MODELS.get_from_params(**model_params)

        if fp16:
            utils.assert_fp16_available()
            model = Fp16Wrap(model)

        model = self._preprocess_model_for_stage(stage, model)
        model = self._postprocess_model_for_stage(stage, model)
        return model

    def get_criterion(self, stage: str) -> _Criterion:
        criterion_params = \
            self.stages_config[stage].get("criterion_params", {})

        criterion = CRITERIONS.get_from_params(**criterion_params)

        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

    def get_optimizer(self, stage: str, model: nn.Module) -> _Optimizer:
        fp16 = isinstance(model, Fp16Wrap)
        params = utils.prepare_optimizable_params(model.parameters(), fp16)

        optimizer_params = \
            self.stages_config[stage].get("optimizer_params", {})

        optimizer = OPTIMIZERS.get_from_params(
            **optimizer_params,
            params=params
        )
        return optimizer

    def get_scheduler(self, stage: str, optimizer) -> _Scheduler:
        scheduler_params = \
            self.stages_config[stage].get("scheduler_params", {})

        scheduler = SCHEDULERS.get_from_params(
            **scheduler_params,
            optimizer=optimizer
        )
        return scheduler

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        data_conf = dict(self.stages_config[stage]["data_params"])

        batch_size = data_conf.pop("batch_size")
        num_workers = data_conf.pop("num_workers")
        drop_last = data_conf.pop("drop_last", False)

        datasets = self.get_datasets(stage=stage, **data_conf)

        loaders = OrderedDict()
        for name, ds_ in datasets.items():
            loader_params = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": torch.cuda.is_available(),
                "drop_last": drop_last,
            }
            if isinstance(ds_, Dataset):
                loader_params["dataset"] = ds_
                loader_params["shuffle"] = name.startswith("train")
            elif isinstance(ds_, dict):
                assert "dataset" in ds_, \
                    "You need to specify dataset for dataloader"
                loader_params["shuffle"] = (
                    name.startswith("train")
                    and ds_.get("sampler") is None)
                loader_params = merge_dicts(ds_, loader_params)
            else:
                raise NotImplementedError
            loaders[name] = DataLoader(**loader_params)

        return loaders

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks_params = (
            self.stages_config[stage].get("callbacks_params", {}))

        callbacks = []
        for key, callback_params in callbacks_params.items():
            callback = CALLBACKS.get_from_params(**callback_params)
            callbacks.append(callback)

        return callbacks


__all__ = [
    "Experiment",
    "BaseExperiment",
    "SupervisedExperiment",
    "ConfigExperiment"
]
