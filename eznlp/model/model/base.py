# -*- coding: utf-8 -*-
from typing import List

import torch

from ...config import Config
from ...wrapper import Batch


class ModelConfigBase(Config):
    """Configurations of a model.

    model
      ├─decoder
      ├─encoder
      └─embedder
    """

    _all_names = []

    @property
    def valid(self):
        for name in self._all_names:
            attr = getattr(self, name)
            if attr is None:
                continue
            # Handle dict attributes (e.g., ohots, mhots, nested_ohots)
            if isinstance(attr, dict):
                if not all(c.valid for c in attr.values()):
                    return False
            elif not attr.valid:
                return False
        return True

    @property
    def name(self):
        names = []
        for name in self._all_names:
            attr = getattr(self, name)
            if attr is not None:
                # Handle dict attributes (e.g., ohots, mhots, nested_ohots)
                if isinstance(attr, dict):
                    names.extend(c.name for c in attr.values())
                else:
                    names.append(attr.name)
        return self._name_sep.join(names)

    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)

    def build_vocabs_and_dims(self, *partitions):
        raise NotImplementedError("Not Implemented `build_vocabs_and_dims`")

    def exemplify(self, entry: dict, training: bool = True):
        raise NotImplementedError("Not Implemented `exemplify`")

    def batchify(self, batch_examples: List[dict]):
        raise NotImplementedError("Not Implemented `batchify`")

    def instantiate(self):
        raise NotImplementedError("Not Implemented `instantiate`")


class ModelBase(torch.nn.Module):
    def __init__(self, config: ModelConfigBase):
        super().__init__()
        for name in config._all_names:
            attr = getattr(config, name)
            if attr is not None:
                # Handle dict attributes (e.g., ohots, mhots, nested_ohots)
                if isinstance(attr, dict):
                    setattr(self, name, torch.nn.ModuleDict({
                        k: c.instantiate() for k, c in attr.items()
                    }))
                else:
                    setattr(self, name, attr.instantiate())

    def pretrained_parameters(self):
        raise NotImplementedError("Not Implemented `pretrained_parameters`")

    def forward2states(self, batch: Batch):
        raise NotImplementedError("Not Implemented `forward2states`")

    def forward(self, batch: Batch, return_states: bool = False):
        states = self.forward2states(batch)
        losses = self.decoder(batch, **states)

        # Return `states` for the `decode` method, to avoid duplicated computation.
        if return_states:
            return losses, states
        else:
            return losses

    def decode(self, batch: Batch, **states):
        if len(states) == 0:
            states = self.forward2states(batch)

        return self.decoder.decode(batch, **states)
