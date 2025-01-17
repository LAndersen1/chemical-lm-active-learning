import abc
import copy
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import sklearn
import torch
from scipy import stats
from sklearn.base import RegressorMixin, _fit_context
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model._base import LinearModel, _preprocess_data  # noqa
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bayesian_protein.types import VALID_SURROGATE_MODELS, Surrogate

import scipy.stats


class BaseSurrogate(abc.ABC):
    def __init__(self):
        super().__init__()
        self.history_x = []
        self.history_y = []
        self.history_smiles = []

    def update(self, x: np.ndarray, y: List[float], smiles: List[str]):
        """
        Refits the whole model by adding x and y to the dataset
        :param x: (N, D) Feature vectors
        :param y: Targets
        """
        self.history_x.extend(x)
        self.history_y.extend(y)
        self.history_smiles.extend(smiles)

        X = np.array(self.history_x)
        y = np.array(self.history_y).reshape(-1, 1)

        self._check_fit_array(X, y)
        self._fit(X, y)

    def _check_fit_array(self, X, y):
        check_array(X)
        check_array(y)

    def _check_forward_array(self, X):
        check_array(X)

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X: (N, D) Design matrix
        :param y: (N, 1) Targets
        """
        pass

    @abc.abstractmethod
    def forward(
        self, X: np.ndarray, smiles: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns mean and variance of the prediction
        :param X: (N, D) Design matrix
        :param smiles: List of SMILES strings of the input molecules
        :return: (N,) (N,) Vectors with mean and variance
        """
        pass

    @abc.abstractmethod
    def expected_improvement_batch(
        self, X: np.ndarray, best_seen: float, smiles: List[str]
    ) -> np.ndarray:
        """
        :param X: (N, D) Design matrix
        :param best_seen: Function value of the best point so far
        :param smiles: List of SMILES strings of the input molecules
        :return: (N, ) Expected improvement for all points in the design matrix
        """
        ...

    @abc.abstractmethod
    def expected_improvement(
        self, x: np.ndarray, best_seen: float, smiles: List[str]
    ) -> float:
        """
        :param x: (D, ) Feature vector
        :param best_seen: Function value of the best point so fat
        :param smiles: List of SMILES strings of the input molecules
        :return: Expected improvement of this point
        """
        pass


class GaussianSurrogate(BaseSurrogate, abc.ABC):
    """
    Surrogate model with gaussian distribution as output, i.e. GP
    """

    def expected_improvement_batch(
        self, X: np.ndarray, best_seen: float, smiles: List[str]
    ) -> np.ndarray:
        """
        :param X: (N, D) Design matrix
        :param best_seen: Function value of the best point so far
        :param smiles: List of SMILES strings of the input molecules
        :return: (N,) Expected improvement for all points in the design matrix
        """
        mean, std = self.forward(X, smiles)
        improvement = best_seen - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std

        ei = std * (z * stats.norm.cdf(z) + stats.norm.pdf(z))

        # Zero standard deviation, then equal to improvement
        invalid = std == 0
        ei[invalid] = improvement[invalid]

        assert len(ei) == X.shape[0]
        return ei

    def expected_improvement(
        self, x: np.ndarray, best_seen: float, smiles: List[str]
    ) -> float:
        """
        :param x: (D, ) Feature vector
        :param best_seen: Function value of the best point so fat
        :param smiles: List of SMILES strings of the input molecules
        :return: Expected improvement of this point
        """
        mean, std = self.forward(x.reshape(1, -1), smiles)

        improvement = best_seen - mean

        if std != 0.0:
            z = improvement / std
            ei = std * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
        else:
            # Zero standard deviation, then equal to improvement
            ei = improvement
        assert ei.size == 1
        return ei.item()


class SklearnGP(GaussianSurrogate):
    def __init__(
        self, kernel: Optional[sklearn.gaussian_process.kernels.Kernel] = None
    ):
        super().__init__()

        if kernel is None:
            kernel = sklearn.gaussian_process.kernels.RBF(length_scale=1)

        self._kernel = kernel
        self._model = GaussianProcessRegressor(
            kernel=self._kernel, normalize_y=True
        )  # Training data does not have zero mean

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)

    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        self._check_forward_array(X)

        if X.shape[0] > 10_000:
            chunks = np.array_split(X, X.shape[0] // 10_000)
            means = []
            stds = []
            for chunk in chunks:
                mean, std = self._model.predict(chunk, return_std=True)
                means.append(mean)
                stds.append(std)
            mean = np.concatenate(means)
            std = np.concatenate(stds)
        else:
            mean, std = self._model.predict(X, return_std=True)
        return mean, std


class BayesianRidgePrior(RegressorMixin, LinearModel):
    """
    Bayesian linear regression layer with prior over noise variance.
    We follow the "NeuralLinear" approach described in [1].

    We assume that y = w.T @ x + eps where eps ~ N(0, sigma²) and
    p(w|sigma²) = N(mu, sigma²*Cov). Additionally, we place a
    normal-inverse-gamma prior over sigma², i.e. sigma² ~ IG(a, b).

    At initialization, we choose Cov^-1 = lambda*I, a=b=eta > 1,
    where lambda describes the precision of the weights.

    The posterior predictive distribution is then given by a
    multivariate students T distribution.

    [1] Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling,
        Riquelme et al., 2018
    """

    # Required by sklearn
    _parameter_constraints = {"fit_intercept": ["boolean"], "copy_X": ["boolean"]}

    def __init__(
        self, a_init, b_init, precision_init, fit_intercept=True, copy_X: bool = True
    ):
        super().__init__()
        self.a_init = a_init
        self.b_init = b_init
        self.precision_init = precision_init
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True)

        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X, y, self.fit_intercept, copy=self.copy_X
        )

        self.X_offset_ = X_offset_
        self.X_scale = X_scale_
        n_samples, n_features = X.shape

        # Initial values
        precision_init = self.precision_init * np.eye(n_features)
        mean_init = np.zeros(n_features)

        # Update
        precision = X.T @ X + precision_init
        cov = np.linalg.inv(precision)

        mean = cov @ (precision_init @ mean_init + X.T @ y)
        a = self.a_init + n_samples / 2

        b = self.b_init
        b += 0.5 * np.dot(y, y)
        b += 0.5 * mean_init.T @ precision_init @ mean_init
        b -= 0.5 * mean.T @ precision @ mean

        self.a_ = a
        self.b_ = b
        self.coef_ = mean
        self.sigma_ = cov
        self._set_intercept(X_offset_, y_offset_, X_scale_)

    def predict(self, X, return_std: bool = False):
        y_mean = self._decision_function(X)
        if not return_std:
            return y_mean
        else:
            df = 2 * self.a_
            ratio = self.b_ / self.a_
            cov = ratio * ((X @ self.sigma_) * X).sum(axis=1) + ratio
            sigmas_squared_data = df / (df - 1) * cov
            y_std = np.sqrt(sigmas_squared_data)
            return y_mean, y_std


class LinearEmpirical(GaussianSurrogate):
    def __init__(self):
        super().__init__()
        print("INIT MODEL")
        self._model = BayesianRidge()

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X: (N, D) Design matrix
        :param y: (N, 1) Targets
        """
        print("RELOAD MODEL")
        self._model = BayesianRidge(lambda_init=1e-7, alpha_1=1e-02, alpha_2=1e-02, lambda_1=1e-02, lambda_2=1e-02)
        self._model.fit(X, y.reshape(-1))
        print(f"ALPHA: {self._model.alpha_}")
        print(f"LAMBDA: {self._model.lambda_}")

    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        self._check_forward_array(X)
        mean, std = self._model.predict(X, return_std=True)
        return mean, std


class LinearPrior(BaseSurrogate):
    def __init__(self, a_init, b_init, precision_init):
        super().__init__()
        self._model = BayesianRidgePrior(a_init, b_init, precision_init)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y.reshape(-1))

    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        self._check_forward_array(X)
        mean, std = self._model.predict(X, return_std=True)
        return mean, std

    def expected_improvement_batch(
        self, X: np.ndarray, best_seen: float, smiles: List[str]
    ) -> np.ndarray:
        df = 2 * self._model.a_
        mean, std = self.forward(X, smiles)

        improvement = best_seen - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std

        ei = improvement * stats.t.cdf(z, df)
        ei += (df / (df - 1)) * (1 + z**2 / df) * std * stats.t.pdf(z, df)

        invalid = std == 0
        ei[invalid] = improvement[invalid]

        assert len(ei) == X.shape[0]
        return ei

    def expected_improvement(
        self, x: np.ndarray, best_seen: float, smiles: List[str]
    ) -> float:
        """
        For a students-t distribution the expected improvement can be
        calculated as described in [1]

        [1] Upgrading from Gaussian Processes to Student's-T Processes, Tracey et al., 2018

        :param smiles:
        :param x: Test point
        :param best_seen: Best seen value sof far
        """
        df = 2 * self._model.a_
        mean, std = self.forward(x.reshape(1, -1), smiles)
        improvement = best_seen - mean

        if std != 0.0:
            z = improvement / std
            ei = improvement * stats.t.cdf(z, df)
            ei += (df / (df - 1)) * (1 + z**2 / df) * std * stats.t.pdf(z, df)
        else:
            ei = improvement

        assert ei.size == 1
        return ei.item()

class RFSurrogate(GaussianSurrogate):

    def __init__(self):
        super().__init__()
        self._model = RandomForestRegressor(random_state=0)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        :param X: (N, D) Design matrix
        :param y: (N, 1) Targets
        """
        self._model.fit(X, y.reshape(-1))

    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        self._check_forward_array(X)
        trees = self._model.estimators_
        predictions = [tree.predict(X) for tree in trees]

        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


class ConstantSurrogate(BaseSurrogate):
    def _fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.zeros((n,)), np.zeros((n,))

    def expected_improvement_batch(
        self, X: np.ndarray, best_seen: float, smiles
    ) -> np.ndarray:
        return np.zeros((X.shape[0],))

    def expected_improvement(self, x: np.ndarray, best_seen: float, smiles) -> float:
        return np.array([0])



class MVEFeedForward(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.GELU(),
                    #nn.Dropout(p=0.1),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    #nn.Dropout(p=0.2),
                    nn.Linear(64, 2),
                ).cuda()
        # Softplus to force variance estimation to be positive
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        """
        Returns mean and variance estimation
        """
        res = self.model.forward(x).reshape((-1,2))
        fin = torch.stack([res[:,0], self.softplus(res[:,1])], dim=1)
        return fin

class MLPSurrogate(GaussianSurrogate):
    def __init__(
        self,
        forward_passes: int = 10,
        retrain: bool = True,
        epochs=350,
        early_stopping_patience: Optional[int] = 25,
    ):
        """
        :param forward_passes: Number of forward passes
        :param retrain: Whether to retrain the model after each update
        :param epochs: Number of training epochs
        :param early_stopping_patience: Number of epochs to wait before stopping after no improvement in validation loss
            set to None to disable early stopping
        """
        super(MLPSurrogate, self).__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forward_passes = forward_passes
        self.retrain = retrain
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self._train_batch_size = 32
        self._inf_batch_size = 512

    def _build_model(self, input_size: int):
        self.model = MVEFeedForward(input_size).to(self.device)

        self._ckpt = copy.deepcopy(self.model.state_dict())

    def _restore(self):
        if self.model is None:
            raise ValueError(
                "Can only be called once 'update' has been called at least once."
            )

        self.model.load_state_dict(self._ckpt)

    def update(self, x: np.ndarray, y: List[float], smiles: List[str]):
        self._build_model(x.shape[1])

        super().update(x, y, smiles)

    def __loss_fn(self, y_pred, y_real, var_pred):
        weighting_pred = 0/2
        weighting_predvar = 2/10
        weighting_var = 8/10

        loss_pred = torch.square(y_real-y_pred)
        loss_predvar = torch.square(y_real-y_pred)/var_pred
        loss_var = torch.square(var_pred)
        #loss_var = var_pred

        losses = weighting_predvar*loss_predvar+weighting_pred*loss_pred+weighting_var*loss_var
        return torch.mean(losses,axis=0)

    def _train_loop(self, loader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler):
        avg = 0
        self.model.train()
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.squeeze().to(self.device)
            optimizer.zero_grad()
            output = self.model.forward(x).reshape((-1,2))
            loss = self.__loss_fn(output[:,0], y, output[:,1])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Update running average of loss
            avg += 1 / i * (loss.mean().item() - avg)

        return avg

    @torch.no_grad()
    def _val_loop(self, loader: DataLoader):
        avg = 0
        self.model.eval()
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.squeeze().to(self.device)
            output = self.model.forward(x).reshape((-1,2))
            loss = self.__loss_fn(output[:,0], y, output[:,1])
            avg += 1 / i * (loss.mean().item() - avg)

        return avg

    def _fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise ValueError(
                "Can only be called once 'update' has been called at least once."
            )
        if self.retrain:
            self._restore()

        # Setup data
        tensor_x = torch.as_tensor(X, dtype=torch.float).cuda()
        tensor_y = torch.as_tensor(y, dtype=torch.float).cuda()
        ds = TensorDataset(tensor_x, tensor_y)
        train_ds, val_ds = random_split(ds, lengths=[0.8, 0.2])
        train_loader = DataLoader(
            train_ds, batch_size=self._train_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=self._inf_batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps =self.epochs*len(train_loader))
        print("_"*50)
        print("REINITING TRAINING")
        print(f"TOTAL STEPS {self.epochs*len(train_ds)}")
        print("_"*50)

        # Train with early stopping
        min_val = None
        best_min_after_n_epochs = 25
        best_model = None
        for current_epoch in range(self.epochs):
            self._train_loop(train_loader, optimizer, lr_scheduler)
            val_loss = self._val_loop(val_loader)
            if current_epoch > best_min_after_n_epochs:
                if min_val is None:
                    min_val = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                else:
                    if val_loss <= min_val:
                        min_val = val_loss
                        best_model = copy.deepcopy(self.model.state_dict())
            print(f"{val_loss} : {min_val} : {lr_scheduler.get_last_lr()}")

        # Implicit early stopping, we load the model with the best validation loss
        self.model.load_state_dict(best_model)

    @torch.no_grad()
    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError(
                "'update' needs to be called at least once before inference."
            )
        self._check_forward_array(X)

        X = torch.as_tensor(X, dtype=torch.float)
        loader = DataLoader(X, batch_size=self._inf_batch_size, shuffle=False)

        mean = []
        std = []
        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model.forward(batch).cpu().numpy().reshape((-1,2))

            mean.append(np.atleast_1d(output[:,0]))
            std.append(np.atleast_1d(np.sqrt(output[:,1])))
            
        return np.concatenate(mean), np.concatenate(std)


class MLPCrossValSurrogate(GaussianSurrogate):
    def __init__(
        self,
        num_models: int = 5,
        retrain: bool = True,
        epochs=150,
    ):
        """
        :param forward_passes: Number of forward passes
        :param retrain: Whether to retrain the model after each update
        :param epochs: Number of training epochs
        """
        super(MLPCrossValSurrogate, self).__init__()
        self.models = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_models = num_models
        self.retrain = retrain
        self.epochs = epochs

        self._train_batch_size = 64
        self._inf_batch_size = 512

    def _build_model(self, input_size: int):
        self.models = []
        for i in range(self.num_models):
            m = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.GELU(),
                    nn.Linear(256, 24),
                    nn.GELU(),
                    nn.Linear(24, 1),
                ).to(self.device)
            self.models.append(m)
        self.known_input_size = input_size

    def _restore(self):
        if self.models is None:
            raise ValueError(
                "Can only be called once 'update' has been called at least once."
            )

        self._build_model(self.known_input_size)

    def update(self, x: np.ndarray, y: List[float], smiles: List[str]):
        self._build_model(x.shape[1])

        super().update(x, y, smiles)

    def _train_loop(self, model, loader: DataLoader, optimizer: torch.optim.Optimizer, lr_scheduler):
        if len(loader) <= 0:
            return 99.9
        avg = 0
        model.train()
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.squeeze().to(self.device)
            optimizer.zero_grad()
            output = model.forward(x).squeeze()
            loss = nn.functional.huber_loss(output, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Update running average of loss
            avg += 1 / i * (loss.mean().item() - avg)

        return avg

    @torch.no_grad()
    def _val_loop(self, model, loader: DataLoader):
        if len(loader) <= 0:
            return 99.9
        avg = 0
        model.eval()
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(self.device), y.squeeze().to(self.device)
            output = model.forward(x).squeeze()
            loss = nn.functional.huber_loss(output, y)
            avg += 1 / i * (loss.mean().item() - avg)

        return avg

    def _fit(self, X: np.ndarray, y: np.ndarray):
        if self.models is None:
            raise ValueError(
                "Can only be called once 'update' has been called at least once."
            )
        if self.retrain:
            self._restore()

        # Setup data
        tensor_x = torch.as_tensor(X, dtype=torch.float).to(self.device)
        tensor_y = torch.as_tensor(y, dtype=torch.float).to(self.device)
        ds = TensorDataset(tensor_x, tensor_y)
        # We split the data into self.num_models parts, model i is trained on all blocks but i
        blocks = random_split(ds, lengths=[1/self.num_models for _ in range(self.num_models)])
        for i in range(self.num_models):
            train_ds_block = [block for j, block in enumerate(blocks) if i != j]
            train_ds = torch.utils.data.ConcatDataset(train_ds_block)
            val_ds = blocks[i]
            # In the first iterations we only have a single sample
            # hence we use the full dataset until we can split it among the models
            if len(X) <= self.num_models:
                train_ds = torch.utils.data.ConcatDataset(blocks)
            if len(X) <= self.num_models:
                val_ds = torch.utils.data.ConcatDataset(blocks)
            self._fit_fold(i, train_ds, val_ds)

    def _fit_fold(self, fold_nr, train_ds, val_ds):
        model = self.models[fold_nr]
        train_loader = DataLoader(
            train_ds, batch_size=self._train_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=self._inf_batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps =self.epochs*len(train_loader))
        print("_"*50)
        print("REINITING TRAINING")
        print(f"TOTAL STEPS {self.epochs*len(train_ds)}")
        print("_"*50)

        min_val = None
        best_min_after_n_epochs = 25
        best_model = None
        for current_epoch in range(self.epochs):
            self._train_loop(model, train_loader, optimizer, lr_scheduler)
            val_loss = self._val_loop(model, val_loader)
            if current_epoch > best_min_after_n_epochs:
                if(min_val is None):
                    min_val = val_loss
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    if val_loss <= min_val:
                        min_val = val_loss
                        best_model = copy.deepcopy(model.state_dict())
            print(f"{val_loss} : {min_val} : {lr_scheduler.get_last_lr()}")
        model.load_state_dict(best_model)

    @torch.no_grad()
    def forward(self, X: np.ndarray, smiles) -> Tuple[np.ndarray, np.ndarray]:
        if self.models is None:
            raise ValueError(
                "'update' needs to be called at least once before inference."
            )
        self._check_forward_array(X)

        X = torch.as_tensor(X, dtype=torch.float)
        loader = DataLoader(X, batch_size=self._inf_batch_size, shuffle=False)

        mean = []
        sem = []
        for batch in loader:
            batch = batch.to(self.device)
            res = []
            for model in self.models:
                model.eval()
                res.append(model.forward(batch).squeeze().cpu().numpy())
            res = np.asarray(res)
            mean.append(np.atleast_1d(np.mean(res, axis=0)))
            sem.append(np.atleast_1d(scipy.stats.sem(res)))

        # We return SEM here, we need to check if this leads to theoretical issues
        # for expected improvement.
        return np.concatenate(mean), np.concatenate(sem)


class MolformerSurrogate(GaussianSurrogate):
    class TokenizedDataset(Dataset):
        def __init__(self, encodings, targets):
            self.targets = targets
            self.encodings = encodings

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}, self.targets[idx]

    def __init__(
        self,
        forward_passes: int = 10,
        retrain: bool = True,
        epochs=50,
        early_stopping_patience: Optional[int] = 5,
    ):
        """
        :param forward_passes: Number of forward passes
        :param retrain: Whether to retrain the model after each update
        :param epochs: Number of training epochs
        :param early_stopping_patience: Number of epochs to wait before stopping after no improvement in validation loss
            set to None to disable early stopping
        """
        super(MolformerSurrogate, self).__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forward_passes = forward_passes
        self.retrain = retrain
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            num_labels=1,
            classifier_dropout_prob=0.2,  # For Probability estimates
        ).to(self.device)

        self._train_batch_size = 64
        self._inf_batch_size = 512

    def _train_loop(self, loader: DataLoader, optimizer: torch.optim.Optimizer):
        avg = 0
        self.model.train()
        for i, (x, y) in enumerate(loader, start=1):
            x = {k: v.to(self.device) for k, v in x.items()}
            y = y.squeeze().to(self.device)
            optimizer.zero_grad()
            output = self.model.forward(**x, labels=y)
            output.loss.backward()
            optimizer.step()

            # Update running average of loss
            avg += 1 / i * (output.loss.mean().item() - avg)

        return avg

    @torch.no_grad()
    def _val_loop(self, loader: DataLoader):
        avg = 0
        self.model.eval()
        for i, (x, y) in enumerate(loader, start=1):
            x = {k: v.to(self.device) for k, v in x.items()}
            y = y.squeeze().to(self.device)
            output = self.model.forward(**x, labels=y)
            avg += 1 / i * (output.loss.mean().item() - avg)

        return avg

    def _fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise ValueError(
                "Can only be called once 'update' has been called at least once."
            )

        # Setup data
        tokens = self.tokenizer.batch_encode_plus(
            self.history_smiles,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )
        tensor_y = torch.as_tensor(y, dtype=torch.float)
        ds = self.TokenizedDataset(tokens, tensor_y)
        train_ds, val_ds = random_split(ds, lengths=[0.8, 0.2])
        train_loader = DataLoader(
            train_ds, batch_size=self._train_batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=self._inf_batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters())

        # Train with early stopping
        val_history = deque(maxlen=self.early_stopping_patience)
        model_history = deque(maxlen=self.early_stopping_patience)
        for _ in trange(self.epochs, desc="Training"):
            self._train_loop(train_loader, optimizer)
            val_loss = self._val_loop(val_loader)

            # Validation loss has not improved in the last x epochs
            if self.early_stopping_patience is not None:
                if len(val_history) > self.early_stopping_patience and all(
                    val_loss > other for other in val_history
                ):
                    idx, _ = min(enumerate(val_history), key=lambda x: x[1])
                    best_model = model_history[idx]
                    self.model.load_state_dict(best_model)
                    break
                else:
                    # Current validation loss is better than the one of the last x epochs
                    val_history.append(val_loss)
                    model_history.append(copy.deepcopy(self.model.state_dict()))

    @torch.no_grad()
    def forward(
        self, X: np.ndarray, smiles: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenizer.batch_encode_plus(
            smiles,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )
        ds = self.TokenizedDataset(tokens, torch.zeros(len(X)))
        loader = DataLoader(ds, batch_size=self._inf_batch_size, shuffle=False)

        mean = []
        var = []
        self.model.train()  # Dropout needs to be enabled for probability estimates
        for x, _ in tqdm(loader, desc="Inference"):
            x = {k: v.to(self.device) for k, v in x.items()}
            outputs = np.stack(
                [
                    self.model.forward(**x).logits.squeeze().cpu().numpy()
                    for _ in range(self.forward_passes)
                ]
            )
            # With a single item mean returns a np.float instead of array which leads to an error when concatenating
            mean.append(np.atleast_1d(outputs.mean(axis=0)))
            var.append(np.atleast_1d(outputs.var(axis=0)))

        return np.concatenate(mean), np.concatenate(var)


def surrogate_factory(surrogate_model: Surrogate) -> BaseSurrogate:
    if surrogate_model == "linear-prior":
        return LinearPrior(a_init=1, b_init=1, precision_init=0.5)
    elif surrogate_model == "linear-empirical":
        return LinearEmpirical()
    elif surrogate_model == "gp":
        return SklearnGP()
    elif surrogate_model == "constant":
        return ConstantSurrogate()
    elif surrogate_model == "mlp":
        return MLPSurrogate()
    elif surrogate_model == "mlp-cross":
        return MLPCrossValSurrogate()
    elif surrogate_model == "molformer":
        return MolformerSurrogate()
    elif surrogate_model == "rf":
        return RFSurrogate()
    else:
        raise ValueError(
            f"Invalid surrogate model {surrogate_model}. Valid choices are {', '.join(VALID_SURROGATE_MODELS)}"
        )
