# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the inference risk."""

from typing import List, Optional, Union, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from anonymeter.neighbors.mixed_types_kneighbors import MixedTypeKNeighbors
from anonymeter.stats.confidence import EvaluationResults, PrivacyRisk


def _run_attack(
        target: pd.DataFrame,
        syn: pd.DataFrame,
        n_attacks: int,
        aux_cols: List[str],
        secret: str,
        n_jobs: int,
        naive: bool,
        regression: Optional[bool],
        ml_model: Optional[Any],
        sample_attacks: bool
) -> tuple[int, Union[Tuple[npt.NDArray, npt.NDArray], pd.Series, None], DataFrame]:
    if regression is None:
        regression = pd.api.types.is_numeric_dtype(target[secret])

    # We only sample if `ml_model` is set to None, i.e MixedTypeKNeighbors is used,
    # or `sample_attacks` is set to True by default.
    if ml_model is None or sample_attacks:
        targets = target.sample(n_attacks, replace=False)
    else:
        # When an `ml_model` is passed we don't want to sample, but rather predict for all targets
        # as we assume it can scale to all target samples, better rather than using MixedTypeKNeighbors
        targets = target
        n_attacks = targets.shape[0]  # this is needed for consistency, for the naive attack below

    if naive:
        guesses = syn.sample(n_attacks)[secret]
    else:
        if ml_model is not None:
            guesses = ml_model.predict(targets)
        else:
            nn = MixedTypeKNeighbors(n_jobs=n_jobs, n_neighbors=1).fit(candidates=syn[aux_cols])
            guesses_idx = nn.kneighbors(queries=targets[aux_cols])
            if isinstance(guesses_idx, tuple):
                raise RuntimeError("guesses_idx cannot be a tuple")

            guesses = syn.iloc[guesses_idx.flatten()][secret]

    return evaluate_inference_guesses(guesses=guesses, secrets=targets[secret],
                                      regression=regression).sum(), guesses, targets


def evaluate_inference_guesses(
        guesses: pd.Series, secrets: pd.Series, regression: bool, tolerance: float = 0.05
) -> npt.NDArray:
    """Evaluate the success of an inference attack.

    The attack is successful if the attacker managed to make a correct guess.

    In case of regression problems, when the secret is a continuous variable,
    the guess is correct if the relative difference between guess and target
    is smaller than a given tolerance. In the case of categorical target
    variables, the inference is correct if the secrets are guessed exactly.

    Parameters
    ----------
    guesses : pd.Series
        Attacker guesses for each of the targets.
    secrets : pd.Series
        Array with the true values of the secret for each of the targets.
    regression : bool
        Whether or not the attacker is trying to solve a classification or
        a regression task. The first case is suitable for categorical or
        discrete secrets, the second for numerical continuous ones.
    tolerance : float, default is 0.05
        Maximum value for the relative difference between target and secret
        for the inference to be considered correct.

    Returns
    -------
    np.array
        Array of boolean values indicating the correcteness of each guess.

    """
    guesses_np = guesses.to_numpy()
    secrets_np = secrets.to_numpy()

    if regression:
        rel_abs_diff = np.abs(guesses_np - secrets_np) / (guesses_np + 1e-12)
        value_match = rel_abs_diff <= tolerance
    else:
        value_match = guesses_np == secrets_np

    nan_match = np.logical_and(pd.isnull(guesses_np), pd.isnull(secrets_np))

    return np.logical_or(nan_match, value_match)


class InferenceEvaluator:
    """Privacy evaluator that measures the inference risk.

    The attacker's goal is to use the synthetic dataset to learn about some
    (potentially all) attributes of a target record from the original database.
    The attacker has a partial knowledge of some attributes of the target
    record (the auxiliary information AUX) and uses a similarity score to find
    the synthetic record that matches best the AUX. The success of the attack
    is compared to the baseline scenario of the trivial attacker, who guesses
    at random.

    .. note::
       For a thorough interpretation of the attack result, it is recommended to
       set aside a small portion of the original dataset to use as a *control*
       dataset for the Inference Attack. These control records should **not**
       have been used to generate the synthetic dataset. For good statistical
       accuracy on the attack results, 500 to 1000 control records are usually
       enough.

       Comparing how successful the attack is when targeting the *training* and
       *control* dataset allows for a more sensitive measure of eventual
       information leak during the training process. If, using the synthetic
       data as a base, the attack is more successful against the original
       records in the training set than it is when targeting the control data,
       this indicates that specific information about some records have been
       transferred to the synthetic dataset.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe with the target records whose secrets the attacker
        will try to guess. This is the private dataframe from which
        the synthetic one has been derived.
    syn : pd.DataFrame
        Dataframe with the synthetic records. It is assumed to be
        fully available to the attacker.
    control : pd.DataFrame (optional)
        Independent sample of original records **not** used to
        create the synthetic dataset. This is used to evaluate
        the excess privacy risk.
    aux_cols : list of str
        Features of the records that are given to the attacker as auxiliary
        information.
    secret : str
        Secret attribute of the targets that is unknown to the attacker.
        This is what the attacker will try to guess.
    regression : bool, optional
        Specifies whether the target of the inference attack is quantitative
        (regression = True) or categorical (regression = False). If None
        (default), the code will try to guess this by checking the type of
        the variable.
    n_attacks : int, default is 500
        Number of attack attempts.
    ml_model: Any
        An ml model fitted on `syn` as training data, and `secret` as target, that supports ::predict(x).
        If not None, it will be used over the MixedTypeKNeighbors in the attack.
    sample_attacks: bool, optional
        Specifies whether we should sample `n_attacks` samples from the `ori` or `control` dataset
        in the attack phase. When a custom `ml_model` is being passed which can scale to more attacks,
        `sample_attacks` can be set to False so that we predict the values for all samples in `ori` and `control`.

    """

    def __init__(
            self,
            ori: pd.DataFrame,
            syn: pd.DataFrame,
            aux_cols: List[str],
            secret: str,
            regression: Optional[bool] = None,
            n_attacks: int = 500,
            control: Optional[pd.DataFrame] = None,
            ml_model: Optional[Any] = None,
            sample_attacks: Optional[bool] = True
    ):
        self._ori = ori
        self._syn = syn
        self._control = control
        self._n_attacks = n_attacks
        self._ml_model = ml_model
        self._sample_attacks = sample_attacks
        if not self._sample_attacks:
            self._n_attacks_ori = self._ori.shape[0]
            self._n_attacks_baseline = self._ori.shape[0]
            self._n_attacks_control = self._control.shape[0]
        else:
            self._n_attacks_ori = self._n_attacks
            self._n_attacks_baseline = self._n_attacks
            self._n_attacks_control = self._n_attacks

        # check if secret is a string column
        if not isinstance(secret, str):
            raise ValueError("secret must be a single column name")

        # check if secret is present in the original dataframe
        if secret not in ori.columns:
            raise ValueError(f"secret column '{secret}' not found in ori dataframe")

        self._secret = secret
        self._regression = regression
        self._aux_cols = aux_cols
        self._evaluated = False
        self._data_groups = self._ori[self._secret].unique().tolist()

    def _attack(self, target: pd.DataFrame, naive: bool, n_jobs: int) -> tuple[
        int, Union[Tuple[npt.NDArray, npt.NDArray],
        pd.Series, None], DataFrame]:
        return _run_attack(
            target=target,
            syn=self._syn,
            n_attacks=self._n_attacks,
            aux_cols=self._aux_cols,
            secret=self._secret,
            n_jobs=n_jobs,
            naive=naive,
            regression=self._regression,
            ml_model=self._ml_model,
            sample_attacks=self._sample_attacks
        )

    def evaluate(self, n_jobs: int = -2) -> "InferenceEvaluator":
        r"""Run the inference attack.

        Parameters
        ----------
        n_jobs : int, default is -2
            The number of jobs to run in parallel.

        Returns
        -------
        self
            The evaluated ``InferenceEvaluator`` object.

        """
        self._n_baseline, _, _ = self._attack(target=self._ori, naive=True, n_jobs=n_jobs)
        self._n_success, self.guesses_success, self.target = self._attack(target=self._ori, naive=False,
                                                                            n_jobs=n_jobs)
        self._n_control, self.guesses_control, self.target_control = (
            None if self._control is None else self._attack(target=self._control, naive=False, n_jobs=n_jobs)
        )

        self._evaluated = True
        return self

    def results(self, confidence_level: float = 0.95) -> EvaluationResults:
        """Raw evaluation results.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.

        Returns
        -------
        EvaluationResults
            Object containing the success rates for the various attacks.

        """
        if not self._evaluated:
            raise RuntimeError("The inference evaluator wasn't evaluated yet. Please, run `evaluate()` first.")

        return EvaluationResults(
            n_attacks=self._n_attacks,
            n_attacks_ori=self._n_attacks_ori,
            n_attacks_baseline=self._n_attacks_baseline,
            n_attacks_control=self._n_attacks_control,
            n_success=self._n_success,
            n_baseline=self._n_baseline,
            n_control=self._n_control,
            confidence_level=confidence_level,
        )

    def risk(self, confidence_level: float = 0.95, baseline: bool = False) -> PrivacyRisk:
        """Compute the inference risk from the success of the attacker.

        This measures how much an attack on training data outperforms
        an attack on control data. An inference risk of 0 means that
        the attack had no advantage on the training data (no inference
        risk), while a value of 1 means that the attack exploited the
        maximally possible advantage.

        Parameters
        ----------
        confidence_level : float, default is 0.95
            Confidence level for the error bound calculation.
        baseline : bool, default is False
            If True, return the baseline risk computed from a random guessing
            attack. If False (default) return the risk from the real attack.

        Returns
        -------
        PrivacyRisk
            Estimate of the inference risk and its confidence interval.

        """
        results = self.results(confidence_level=confidence_level)
        return results.risk(baseline=baseline)

    def risk_for_groups(self, confidence_level: float = 0.95) -> dict[
        str, dict[str, Union[EvaluationResults, PrivacyRisk]]]:
        """Compute the attack risks on a group level, for every unique value of `self._data_groups`.

            Parameters
            ----------
            confidence_level : float, default is 0.95
                Confidence level for the error bound calculation.

            Returns
            -------
            dict[str, dict[str, EvaluationResults | PrivacyRisk]]
                The group as a key, and then for every group the results (EvaluationResults),
                and the risks (PrivacyRisk).

            """
        if not self._evaluated:
            self.evaluate(n_jobs=-2)

        all_results = {}

        # For every unique group in `self._data_groups`
        for group in self._data_groups:
            # Get the targets for the current group
            target = self.target[self.target[self._secret] == group]

            # Get the guesses for the current group
            guess = self.guesses_success.loc[target.index]

            # Count the number of success attacks
            n_success = evaluate_inference_guesses(guesses=guess,
                                                   secrets=target[self._secret],
                                                   regression=self._regression).sum()

            target_control = None
            guesses_control = None
            if self._control:
                # Get the targets for the current control group
                target_control = self.target_control[self.target_control[self._secret] == group]

                # Get the guesses for the current control group
                guesses_control = self.guesses_control.loc[target_control.index]

                # Count the number of success control attacks
                n_control = evaluate_inference_guesses(guesses=guesses_control,
                                                        secrets=target_control[self._secret],
                                                        regression=self._regression).sum()
            else:
                n_control = None

            # Recreate the EvaluationResults for the current group
            results = EvaluationResults(
                n_attacks=self._n_attacks,  # passing for
                n_attacks_ori=self._n_attacks_ori,
                n_attacks_baseline=self._n_attacks_baseline,
                # We leave the overall n_attacks_baseline here, it doesn't change the risk
                n_attacks_control=self._n_attacks_control,
                n_success=n_success,
                n_baseline=self._n_baseline,  # We leave the overall _n_baseline here, it doesn't change the risk
                n_control=n_control,
                confidence_level=confidence_level,
            )
            # Compute the risk
            risk = results.risk()

            all_results[group] = {
                "results": results,
                "risk": risk
            }

        return all_results
