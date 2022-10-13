import shutil
from time import sleep
import pytest
import numpy as np
import tempfile

import torch
from ding.data.model_loader import FileModelLoader
from ding.data.storage_loader import FileStorageLoader
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware.distributer import ContextExchanger, ModelExchanger
from ding.framework.parallel import Parallel
from ding.utils.default_helper import set_pkg_seed
from os import path

from ding.framework.middleware.collector import TransitionList
from ding.framework.context import Context
from typing import Union, Dict, List, Any
import treetensor.torch as ttorch
import dataclasses
from ditk import logging
import multiprocessing as mp


@dataclasses.dataclass
class OnlineRLContextLearnerTest(Context):
    total_step: int = 0
    env_step: int = 0
    env_episode: int = 0
    train_iter: int = 0
    trajectories: List = None
    episodes: List = None
    trajectory_end_idx: List = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.keep(
            'env_step', 'env_episode', 'train_iter', 'last_eval_iter', 'trajectories', 'episodes', 'trajectory_end_idx'
        )

    @classmethod
    def set_method(cls, obj: Any, key: str, value: Any):
        if key == "trajectory_end_idx" or key == "trajectories" or key == "episodes":
            v = getattr(obj, key, [])
            if v is None:
                v = []
            v.extend(value)
            setattr(obj, key, v)
        else:
            setattr(obj, key, value)


def context_exchanger_main():
    with task.start(ctx=OnlineRLContext()):
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        elif task.router.node_id == 1:
            task.add_role(task.role.COLLECTOR)

        task.use(ContextExchanger(skip_n_iter=1))

        if task.has_role(task.role.LEARNER):

            def learner_context(ctx: OnlineRLContext):
                assert len(ctx.trajectories) == 2
                assert len(ctx.trajectory_end_idx) == 4
                assert len(ctx.episodes) == 8
                assert ctx.env_step > 0
                assert ctx.env_episode > 0
                yield
                ctx.train_iter += 1

            task.use(learner_context)
        elif task.has_role(task.role.COLLECTOR):

            def collector_context(ctx: OnlineRLContext):
                if ctx.total_step > 0:
                    assert ctx.train_iter > 0
                yield
                ctx.trajectories = [np.random.rand(10, 10) for _ in range(2)]
                ctx.trajectory_end_idx = [1 for _ in range(4)]
                ctx.episodes = [np.random.rand(10, 10) for _ in range(8)]
                ctx.env_step += 1
                ctx.env_episode += 1

            task.use(collector_context)

        task.run(max_step=3)


@pytest.mark.unittest
def test_context_exchanger():
    Parallel.runner(n_parallel_workers=2)(context_exchanger_main)


def context_exchanger_with_storage_loader_main():
    with task.start(ctx=OnlineRLContext()):
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        elif task.router.node_id == 1:
            task.add_role(task.role.COLLECTOR)

        tempdir = path.join(tempfile.gettempdir(), "test_storage_loader")
        storage_loader = FileStorageLoader(dirname=tempdir)
        try:
            task.use(ContextExchanger(skip_n_iter=1, storage_loader=storage_loader))

            if task.has_role(task.role.LEARNER):

                def learner_context(ctx: OnlineRLContext):
                    assert len(ctx.trajectories) == 2
                    assert len(ctx.trajectory_end_idx) == 4
                    assert len(ctx.episodes) == 8
                    assert ctx.env_step > 0
                    assert ctx.env_episode > 0
                    yield
                    ctx.train_iter += 1

                task.use(learner_context)
            elif task.has_role(task.role.COLLECTOR):

                def collector_context(ctx: OnlineRLContext):
                    if ctx.total_step > 0:
                        assert ctx.train_iter > 0
                    yield
                    ctx.trajectories = [np.random.rand(10, 10) for _ in range(2)]
                    ctx.trajectory_end_idx = [1 for _ in range(4)]
                    ctx.episodes = [np.random.rand(10, 10) for _ in range(8)]
                    ctx.env_step += 1
                    ctx.env_episode += 1

                task.use(collector_context)

            task.run(max_step=3)
        finally:
            storage_loader.shutdown()
            sleep(1)
            if path.exists(tempdir):
                shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_context_exchanger_with_storage_loader():
    Parallel.runner(n_parallel_workers=2)(context_exchanger_with_storage_loader_main)


class MockPolicy:

    def __init__(self) -> None:
        self._model = self._get_model(10, 10)

    def _get_model(self, X_shape, y_shape) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(X_shape, 24), torch.nn.ReLU(), torch.nn.Linear(24, 24), torch.nn.ReLU(),
            torch.nn.Linear(24, y_shape)
        )

    def train(self, X, y):
        loss_fn = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        y_pred = self._model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self._model(X)


def model_exchanger_main():
    with task.start(ctx=OnlineRLContext()):
        set_pkg_seed(0, use_cuda=False)
        policy = MockPolicy()
        X = torch.rand(10)
        y = torch.rand(10)

        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:
            task.add_role(task.role.COLLECTOR)

        task.use(ModelExchanger(policy._model))

        if task.has_role(task.role.LEARNER):

            def train(ctx):
                policy.train(X, y)
                sleep(0.3)

            task.use(train)
        else:
            y_pred1 = policy.predict(X)

            def pred(ctx):
                if ctx.total_step > 0:
                    y_pred2 = policy.predict(X)
                    # Ensure model is upgraded
                    assert any(y_pred1 != y_pred2)
                sleep(0.3)

            task.use(pred)

        task.run(2)


@pytest.mark.unittest
def test_model_exchanger():
    Parallel.runner(n_parallel_workers=2, startup_interval=0)(model_exchanger_main)


def model_exchanger_main_with_model_loader():
    with task.start(ctx=OnlineRLContext()):
        set_pkg_seed(0, use_cuda=False)
        policy = MockPolicy()
        X = torch.rand(10)
        y = torch.rand(10)

        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        else:
            task.add_role(task.role.COLLECTOR)

        tempdir = path.join(tempfile.gettempdir(), "test_model_loader")
        model_loader = FileModelLoader(policy._model, dirname=tempdir)
        task.use(ModelExchanger(policy._model, model_loader=model_loader))

        try:
            if task.has_role(task.role.LEARNER):

                def train(ctx):
                    policy.train(X, y)
                    sleep(0.3)

                task.use(train)
            else:
                y_pred1 = policy.predict(X)

                def pred(ctx):
                    if ctx.total_step > 0:
                        y_pred2 = policy.predict(X)
                        # Ensure model is upgraded
                        assert any(y_pred1 != y_pred2)
                    sleep(0.3)

                task.use(pred)
            task.run(2)
        finally:
            model_loader.shutdown()
            sleep(0.3)
            if path.exists(tempdir):
                shutil.rmtree(tempdir)


@pytest.mark.unittest
def test_model_exchanger_with_model_loader():
    Parallel.runner(n_parallel_workers=2, startup_interval=0)(model_exchanger_main_with_model_loader)


COLLECTOR_PROCESS_NUM = 5


def context_exchanger_multiple_collectors():
    ENV_NUM = 4
    STEPS_NUM = 10
    if task.router.node_id == 0:
        with task.start(ctx=OnlineRLContextLearnerTest()):
            task.add_role(task.role.LEARNER)

            def learner_context(ctx: OnlineRLContextLearnerTest):
                yield
                if ctx.train_iter == STEPS_NUM - 1:
                    assert len(ctx.trajectories) == COLLECTOR_PROCESS_NUM * STEPS_NUM * ENV_NUM
                    assert len(ctx.trajectory_end_idx) == COLLECTOR_PROCESS_NUM * STEPS_NUM * ENV_NUM
                    assert len(ctx.episodes) == COLLECTOR_PROCESS_NUM * STEPS_NUM
                    assert ctx.env_step == COLLECTOR_PROCESS_NUM * STEPS_NUM * ENV_NUM
                    assert ctx.env_episode == COLLECTOR_PROCESS_NUM * STEPS_NUM

                ctx.train_iter += 1

            task.use(ContextExchanger(skip_n_iter=1, debug=True, ctx_set_method=OnlineRLContextLearnerTest.set_method))
            task.use(learner_context)

            task.run(max_step=STEPS_NUM, sync_step_by_step=True)
    else:
        with task.start(ctx=OnlineRLContext()):
            task.add_role(task.role.COLLECTOR)

            def collector_context(ctx: OnlineRLContext):
                if ctx.total_step > 0:
                    assert ctx.train_iter > 0
                yield
                transitionlist = TransitionList(ENV_NUM)
                for env_id in range(ENV_NUM):
                    transition = ttorch.as_tensor(
                        {
                            'done': False,
                            'trajectories': np.random.rand(task.router.node_id, COLLECTOR_PROCESS_NUM)
                        }
                    )
                    if env_id == ENV_NUM - 1:
                        transition.done = True
                    transitionlist.append(env_id, transition)
                ctx.trajectories, ctx.trajectory_end_idx = transitionlist.to_trajectories()
                ctx.episodes = transitionlist.to_episodes()
                ctx.env_step += 4
                ctx.env_episode += 1

            task.use(ContextExchanger(skip_n_iter=1, debug=True))
            task.use(collector_context)
            task.run(max_step=STEPS_NUM, sync_step_by_step=True)


def run_task(rank):
    listen_to = "127.0.0.1"
    port = int("5051{}".format(rank))
    if rank != 0:
        label = 'collector'
        attach_to = ["tcp://127.0.0.1:50510"]
    else:
        label = 'learner'
        attach_to = []

    Parallel.runner(
        node_ids=rank,
        n_parallel_workers=1,
        protocol="tcp",
        topology="star",
        attach_to=attach_to,
        address=listen_to,
        ports=port,
        labels=label,
        world_size=COLLECTOR_PROCESS_NUM + 1
    )(context_exchanger_multiple_collectors)


@pytest.mark.unittest
def test_context_exchanger_multiple_collectors():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=COLLECTOR_PROCESS_NUM + 1) as pool:
        pool.map(run_task, range(COLLECTOR_PROCESS_NUM + 1))
