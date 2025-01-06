import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from lightgbm import LGBMClassifier
from math import erf, sqrt, pi
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from universal_evasion_attacks import Attack, HillClimbing, SimulatedAnnealing, TabuSearch, Neighborhood, Radar

STANDARD_TEST_SIZE = 0.2
DATASET_NAME = "Unknown"

###########################################################
#   VERBOSE and custom error
###########################################################

VERBOSE = 2

def set_verbose(level):
    global VERBOSE
    if isinstance(level, bool):
        print("'verbose' can take values 0, 1 or 2, but boolean value was given. Converting into integer.")
        level = int(level)
    VERBOSE = level

class HarError(Exception):
    """Error type for HAR protocol."""
    pass

###########################################################
#   CHECK FUNCTIONS
###########################################################
def check_data(X, y, rigid=False):
    """
    - The dataset must be a pandas DataFrame with only numerical values.
      Should the need arise, make a one hot encoding of the features
      and label encode the target (level 2 of verbosity).
    - Single-label classification only is supported.
    - No missing values are allowed.
      Should the need arise, drop the rows with missing values (level 2 of verbosity).
    - Returns "True" if the dataset is valid, "False" otherwise (level 1 of verbosity).
      Also returns the possibly modified dataset.

    If set on "rigid", the function will return "False" if at least 1 condition is not met.
    No changes will be made to the dataset.
    Otherwise, it will try to adjust the dataset as much as possible
    and return "False" only if it is impossible to do so.
    """
    try:
        # Ensure X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            if rigid:
                print("CHECK FAILED [Stage 1]: X is not a DataFrame.")
                return False, X, y
            try:
                X = pd.DataFrame(X)
                if VERBOSE > 1:
                    print("Converted X to a DataFrame.")
            except Exception as e:
                print(f"CHECK FAILED [Stage 1]: Failed to convert X to DataFrame: {e}")
                return False, X, y

        # Ensure y is a pandas Series or convert to one
        if not isinstance(y, pd.Series):
            if rigid and not isinstance(y, (list, np.ndarray)):
                print("CHECK FAILED [Stage 1]: y is not a Series.")
                return False, X, y
            try:
                y = pd.Series(y)
                if VERBOSE > 1:
                    print("Converted y to a Series.")
            except Exception as e:
                print(f"CHECK FAILED [Stage 1]: Failed to convert y to Series: {e}")
                return False, X, y

        # Check for missing values
        if X.isnull().any().any() or y.isnull().any():
            if rigid:
                print("CHECK FAILED [Stage 1]: Missing values detected.")
                return False, X, y
            else:
                if VERBOSE > 1:
                    print("Missing values detected. Dropping rows with missing values.")
                combined = pd.concat([X, y], axis=1)
                combined = combined.dropna()
                X = combined.iloc[:, :-1]
                y = combined.iloc[:, -1]

        # Check if all columns in X are numerical
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            if rigid:
                print("CHECK FAILED [Stage 1]: Non-numerical columns detected.")
                return False, X, y
            else:
                if VERBOSE > 1:
                    print("Non-numerical columns detected. Applying one-hot encoding to categorical features.")
                X = pd.get_dummies(X, drop_first=True)

        # Ensure y is label encoded
        if not pd.api.types.is_numeric_dtype(y):
            if rigid:
                print("CHECK FAILED [Stage 1]: Target variable is not label encoded.")
                return False, X, y
            else:
                try:
                    if VERBOSE > 1:
                        print("Target variable is not numerical. Applying label encoding.")
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                except Exception as e:
                    print(f"CHECK FAILED [Stage 1]: Failed to encode target variable: {e}")
                    return False, X, y
                
        # "y" must be able to handle "unique", so convert to a Series if it's not
        if not isinstance(y, pd.Series) and not rigid:
            if VERBOSE > 1:
                print("Converting y to a Series.")
            y = pd.Series(y)
        elif not isinstance(y, pd.Series):
            print("CHECK FAILED [Stage 1]: Target variable is not a Series.")
            return False, X, y

        # Ensure single-label classification
        if len(y.unique()) < 2:
            print("CHECK FAILED [Stage 1]: Target variable must have at least two unique classes.")
            return False, X, y
        
        if VERBOSE > 0:
            print("CHECK PASSED: Dataset is valid for the HAR protocol.")

        return True, X, y

    except Exception as e:
        print(f"CHECK FAILED [Stage 1]: An error occurred: {e}")
        return False, X, y

def check_estimator(X_train, X_test, y_train, y_test, estimator, rigid=False):
    """
    - The estimator must be already trained on the dataset and have a "predict" method.
      The "predict" method must have the same input-output signature as the one 
      from scikit-learn, but it doesn't need to be a scikit-learn estimator.
    - If not "rigid", train the estimator on the provided training set (verbosity level 2).
    - Inform the accuracy and F1 score of the estimator on the test set (verbosity level 1).
    - A test is made on the 5 first samples of the test set to check if the estimator works properly.
    - Returns "True" if the estimator is valid (level 1 of verbosity), "False" otherwise (level 0 of verbosity)
      along with the possibly modified estimator if it was not rigid.
    """
    try:
        # Ensure the estimator has a "predict" method
        if not hasattr(estimator, "predict"):
            print("CHECK FAILED [Stage 2]: Estimator has no 'predict' method.")
            return False, estimator
        
        # If returns probabilities, convert to classes
        class Wrapper:
            def __init__(self, model):
                self.model = model

            def predict_proba(self, X):
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X)
                else:
                    raise AttributeError(f"Model {type(self.model).__name__} does not support predict_proba.")

            def predict(self, X):
                # Ensure correct axis usage for multi-class probabilities
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1)

        # Wrap models with multi-class output for the HAR protocol
        if hasattr(estimator, "predict_proba") and estimator.predict(X_test).ndim > 1:
            if rigid:
                print("CHECK FAILED [Stage 2]: Estimator has multi-class output. Wrap it for hard label prediction.")
                return False, estimator
            if VERBOSE > 1:
                print(f"Wrapping estimator for hard label prediction.")
            estimator = Wrapper(estimator)

        # Check the estimator on the test set
        try:
            y_pred = estimator.predict(X_test)
        except Exception as e:
            # Train the estimator if not rigid
            if not rigid:
                if VERBOSE > 1:
                    print("Training the estimator on the provided training set.")
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
            else:
                print(f"CHECK FAILED [Stage 2]: Estimator failed to predict on the test set: {e}")
                return False, estimator
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        if VERBOSE > 1:
            print(f"Estimator on test set: Accuracy: {acc:.4f}, F1 score: {f1:.4f}")

        # Check the estimator on the first 5 samples of the test set
        y_pred = estimator.predict(X_test[:5])
        if y_pred.shape != (5,):
            print(f"CHECK FAILED [Stage 2]: Estimator is not working properly: shape is {y_pred.shape}.")
            return False, estimator

        if VERBOSE > 0:
            print("CHECK PASSED: Estimator is valid.")

        return True, estimator

    except Exception as e:
        raise(e)
        print(f"CHECK FAILED [Stage 2]: An error occurred: {e}")
        return False, estimator

def simple_check_constraints(constraints):
    """
    Simple check for constraints: it must be a dictionary with the following keys:
    - "equality": list
    - "inequality": list
    - "clip_min": list
    - "clip_max": list
    - "categorical": list

    A more thorough check is done during neighborhood generation.
    """
    if not isinstance(constraints, dict):
        print("CHECK FAILED [Stage 3]: Constraints must be a dictionary.")
        return False
    if not all(key in constraints for key in ["equality", "inequality", "clip_min", "clip_max", "categorical"]):
        print("CHECK FAILED [Stage 3]: Constraints must have the following keys: "
              "'equality', 'inequality', 'clip_min', 'clip_max', 'categorical'.")
        return False
    if not all(isinstance(constraints[key], list) for key in constraints):
        print("CHECK FAILED [Stage 3]: All values in constraints must be lists.")
        return False

    if VERBOSE > 0:
        print("CHECK PASSED: Constraints basic format is valid.")
    return True

def check_neighborhood(neighborhood):
    if not isinstance(neighborhood, Neighborhood):
        print("CHECK FAILED [Stage 4]: Neighborhood must be an instance of the Neighborhood class.")
        return False
    if VERBOSE > 0:
        print("CHECK PASSED: Neighborhood is valid.")
    return True

def check_attacks(attacks):
    if not isinstance(attacks, list):
        print("CHECK FAILED [Stage 5]: Attacks must be a list.")
        return False
    if not all(isinstance(attack, Attack) for attack in attacks):
        print("CHECK FAILED [Stage 5]: All elements in the attacks list must be instances of the Attack class.")
        return False
    if VERBOSE > 0:
        print("CHECK PASSED: Attacks are valid.")
        if VERBOSE > 1:
            print("There are", len(attacks), "attacks in the list.")
    return True

def check_SLARC_weights(omega_k_dict, X_train, rigid=False):
    """
    - Sum of all weights must be 1.
    - All features must have a weight.
      If not rigid, assign a weight of 0 to missing features.
    - Returns "True" if the weights are valid (level 1 of verbosity), 
      "False" otherwise (level 0 of verbosity) along with the possibly updated omega_k_dict.
    """
    try:
        # Ensure omega_k_dict is a dictionary
        if not isinstance(omega_k_dict, dict):
            print("CHECK FAILED [Stage 6]: Weights must be a dictionary.")
            return False, omega_k_dict

        # Ensure the sum of all weights is 1
        total = sum(omega_k_dict.values())
        if not np.isclose(total, 1.0):
            print(f"CHECK FAILED [Stage 6]: Sum of all weights is {total:.4f}, but should be 1.")
            return False, omega_k_dict

        # Ensure all features have a weight
        if len(omega_k_dict) != X_train.shape[1]:
            if rigid:
                print("CHECK FAILED [Stage 6]: Not all features have a weight.")
                return False, omega_k_dict
            else:
                if VERBOSE > 1:
                    print("Not all features have a weight. Assigning 0 to missing features.")
                for col in X_train.columns:
                    if col not in omega_k_dict:
                        omega_k_dict[col] = 0

        if VERBOSE > 0:
            print("CHECK PASSED: Weights are valid.")

        return True, omega_k_dict

    except Exception as e:
        print(f"CHECK FAILED [Stage 6]: An error occurred: {e}")
        return False, omega_k_dict

###########################################################
#   SLARC, SLAAC, LIKELIHOOD, AND CHECKS
###########################################################

def SLARC(x0, x1, mu_k, sigma_k, omega_k, x_min_k, x_max_k):
    """
    Standard Linear Attribute Relative Cost (SLARC) between two scalars x0 and x1 
    for a given label's distribution parameters:
      - mu_k: mean of feature k for label l
      - sigma_k: std  of feature k for label l
      - x_min_k, x_max_k: min/max (e.g. 1st/99th percentile) for feature k for label l
      - omega_k: importance weight of feature k
    This follows equation (5.4) in the text.
    """
    # If there's no change, cost is zero
    if x0 == x1:
        return 0.0

    # The "relative" distance portion (like L1, but see eqn 5.4 and eqn 5.5, 5.6 for ϵ_lk)
    # ρ_lk(x) = (x_max_k - x) / (x - x_min_k)
    def rho(val):
        if (val - x_min_k) == 0:
            return np.inf
        return (x_max_k - val) / (val - x_min_k)

    rho_mu  = rho(mu_k)
    rho_x1  = rho(x1)

    if (rho_mu - 1)*(rho_x1 - 1) >= 0:
        eps = 1.0
    else:
        # eqn (5.6): max(ρ_lk(μ_lk), 1 / ρ_lk(μ_lk))
        eps = max(rho_mu, 1.0/rho_mu)  # if rho_mu were zero, we could handle separately

    # sign factor: sgn(|x1 - mu_k| - |x0 - mu_k|)
    val_sign = np.sign( abs(x1 - mu_k) - abs(x0 - mu_k) )

    # SLARC
    cost = np.sqrt(np.pi/2) * (omega_k / sigma_k) * eps * abs(x1 - x0) * val_sign
    return cost

def SLARC_vectorized(X_train, y_train, omega_k_dict, 
                     clip_percentiles=(0.01, 0.99)):
    """
    Precompute the necessary SLARC stats for each label l and each feature k:
      - mu_lk, sigma_lk, x_lk^min, x_lk^max
      - (already have ω_lk from omega_k_dict)
    Returns a dictionary like:
      slarc_params = {
          label_1: {
              'mu': {...}, 
              'sigma': {...},
              'min': {...},
              'max': {...},
              'omega': {...}  # from omega_k_dict
          },
          label_2: {...}, 
          ...
      }
    """
    unique_labels = np.sort(y_train.unique())
    slarc_params = {}

    # Ensure y_train's index matches X_train
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    for lbl in unique_labels:
        # Filter to this label only
        X_lbl = X_train[y_train == lbl]
        # Compute mu, sigma, min, max
        mu_dict    = X_lbl.mean(numeric_only=True).to_dict()
        sigma_dict = X_lbl.std(numeric_only=True).to_dict()

        # We allow percentiles to exclude outliers
        min_p = X_lbl.quantile(clip_percentiles[0]).to_dict()
        max_p = X_lbl.quantile(clip_percentiles[1]).to_dict()

        # store results
        slarc_params[lbl] = {
            "mu"    : mu_dict,
            "sigma" : sigma_dict,
            "min"   : min_p,
            "max"   : max_p,
            "omega" : omega_k_dict.copy()  # same across all labels
        }
    return slarc_params

def SLAAC(x_vec, label, slarc_params):
    """
    Standard Linear Attribute Absolute Cost (SLAAC) = SLARC_l(x, μ_l).
    i.e. measure the cost of x_vec from the label's "epicenter" mu_l.
    """
    mu_dict    = slarc_params[label]["mu"]
    sigma_dict = slarc_params[label]["sigma"]
    min_dict   = slarc_params[label]["min"]
    max_dict   = slarc_params[label]["max"]
    omega_dict = slarc_params[label]["omega"
    ]

    cost = 0.0
    for k in x_vec.index:
        x_k    = x_vec[k]
        mu_k   = mu_dict[k]
        sigma_k= sigma_dict[k]
        if sigma_k == 0:
            # If there's no variance for feature k, cost is 0 if x_k == mu_k,
            # otherwise large penalty
            cost += 0.0 if x_k == mu_k else 10.0
        else:
            x_min_k = min_dict[k]
            x_max_k = max_dict[k]
            omega_k = omega_dict[k]
            this_SLARC = SLARC(mu_k, x_k, mu_k, sigma_k, omega_k, x_min_k, x_max_k)
            cost += this_SLARC
    return cost

def check_SLARC_hypotheses(
    X_train,
    y_train,
    omega_k_dict,
    target_class_idx,
    target_label,
    method="BC", # "BC" for Bhattacharyya, "KL" for Kullback-Leibler
    max_samples=100_000,
    labels_to_distinguish="all",
    plot_heatmaps=True,
    plot_kde=True,
    stop_after_computing_slarc_params=False,
    precomputed_M=None, # If precomputed, pass them here
    precomputed_S=None
):
    """
    Performs statistical checks on the hypotheses of the SLARC cost function.
    Returns (boolean, confidence_score).

    If the SLARC hypotheses are not met, return False.
    Otherwise True, plus the confidence score.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (with consistent indexing vs. y_train).
    y_train : pd.Series
        Training labels.
    target_class_idx : int
        The target label for evasion attacks.
    omega_k_dict : dict
        Dictionary of feature importance weights, summing to 1.
    method : {"BC", "KL"}, optional
        - "BC": Use the truncated Bhattacharyya-based score = (1 - BC).
        - "KL": Use the Kullback-Leibler-based score = (1 - 1/(1 + KL)).
    max_samples : int, optional
        If the training set is too large, sample max_samples rows at random.
        Set to None to use the entire dataset.
    labels_to_distinguish : "all" or list of labels, optional
        - "all" (default): compute checks among all unique labels in y_train.
        - If a list, e.g., [0, 1], only compute among those labels.
    plot_heatmaps, plot_kde : bool
        If True, show debugging plots for M, S, and estimated PDFs.
    stop_after_computing_slarc_params : bool
        If True, stop after computing the SLA(R)C parameters.

    Returns
    -------
    (bool, float)
        - bool: True if the SLARC hypotheses are considered met (confidence >= 0.5).
        - float: The confidence score.
        - dict: The SLARC parameters (M, S matrices, etc.) for future use.
    """

    # ----------------------
    # 0) Helper definitions
    # ----------------------
    def KL_div_gauss(m1, s1, m2, s2):
        """Kullback-Leibler divergence for two univariate Gaussians."""
        if s1 < 1e-10:
            s1 = 1e-10
        if s2 < 1e-10:
            s2 = 1e-10
        return np.log(s2 / s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5

    def BC_gauss(m1, s1, m2, s2):
        """Closed-form Bhattacharyya coefficient for two univariate Gaussians."""
        if s1 < 1e-10:
            s1 = 1e-10
        if s2 < 1e-10:
            s2 = 1e-10
        denom = (s1**2 + s2**2)
        num = 2 * s1 * s2
        bc = np.sqrt(num / denom) * np.exp(-0.25 * (m1 - m2)**2 / denom)
        return bc

    # 1) Reset indices to ensure alignment
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # 2) Restrict to a subset of samples if needed
    if max_samples is not None and X_train.shape[0] > max_samples:
        if VERBOSE > 1:
            print(f"Using only {max_samples} random samples for M and S calculation.")
        idx_small = np.random.choice(X_train.shape[0], max_samples, replace=False)
        X_train = X_train.iloc[idx_small].reset_index(drop=True)
        y_train = y_train.iloc[idx_small].reset_index(drop=True)

    # 3) Build the SLARC parameters
    #    Replace the following two lines with your actual SLA(R)C-fitting code
    slarc_params = SLARC_vectorized(X_train, y_train, omega_k_dict)
    labels = np.sort(y_train.unique())
    L = len(labels)

    # 4) Build M, S matrices
    if precomputed_M is not None and precomputed_S is not None:
        if VERBOSE > 1:
            print("Using precomputed M, S matrices.")
        M = precomputed_M
        S = precomputed_S
    
    else:
        M = np.zeros((L, L))
        S = np.zeros((L, L))
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                X_j = X_train[y_train == lj]
                if len(X_j) == 0:
                    M[i, j] = 0
                    S[i, j] = 0
                    continue
                costs = []
                for _, row in X_j.iterrows():
                    c = SLAAC(row, li, slarc_params)
                    costs.append(c)
                costs = np.array(costs)
                M[i, j] = costs.mean()
                S[i, j] = costs.std()

    # Save them back into slarc_params for future use
    slarc_params["M"] = M
    slarc_params["S"] = S
    slarc_params["labels_order"] = labels

    if stop_after_computing_slarc_params:
        return True, None, slarc_params

    # 5) Decide which labels to use for the checks
    if labels_to_distinguish == "all":
        label_indices = range(L)
    else:
        subset = set(labels_to_distinguish)
        label_indices = [i for i, lbl in enumerate(labels) if lbl in subset]

    if VERBOSE > 1 and labels_to_distinguish != "all":
        print(f"Div. checks among labels: {labels[label_indices]}")

    # 6) Evaluate the divergences/coefficients, compute the confidence scores and plot graphs
    all_scores = []
    i = target_class_idx
    for j in label_indices:
        if i == j:
            continue

        if method.upper() == "KL":
            # KL-based confidence: S_KL = 1 - 1/(1 + KL)
            dkl = KL_div_gauss(M[i, i], S[i, i], M[i, j], S[i, j])
            score_ij = 1.0 - 1.0 / (1.0 + dkl)

        elif method.upper() == "BC":
            # BC-based confidence: score = 1 - BC (truncated at 0)
            bc = BC_gauss(M[i, i], S[i, i], M[i, j], S[i, j])
            score_ij = np.sqrt(1.0 - bc)

        else:
            raise ValueError(f"Unknown method: {method}")

        all_scores.append(score_ij)

    if VERBOSE > 1:
            if plot_heatmaps:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
                axes[0].set_title("Mean SLAAC (M matrix)")

                sns.heatmap(S, annot=True, fmt=".2f", cmap="Reds", ax=axes[1])
                axes[1].set_title("Std SLAAC (S matrix)")

                # Add numeric labels
                axes[0].set_xlabel("Label from which SLAAC is computed")
                axes[0].set_ylabel("Label on which SLAAC is computed")
                axes[1].set_xlabel("Label from which SLAAC is computed")
                axes[1].set_ylabel("Label on which SLAAC is computed")

                plt.tight_layout()
                plt.savefig(f"graphs/{DATASET_NAME}_SLAAC_M_S_matrices.png")
                plt.show()

            if plot_kde:
                # Extract means and standard deviations for the target class (target_class_idx)
                target_means = M[:, target_class_idx]  # Means for the target class
                target_stds = S[:, target_class_idx]   # Stds for the target class

                # Check the std range to determine scale
                min_std, max_std = np.min(target_stds[target_stds > 0]), np.max(target_stds)
                #use_log_scale = (max_std - min_std) > 10
                use_log_scale = False
                
                # Generate x-range starting from SLAAC = 0
                x_range = np.linspace(0, max(target_means) + 3 * max(target_stds), 1000)

                # Prepare data for KDE plot
                plt.figure(figsize=(12, 8))

                for j, (mean, std) in enumerate(zip(target_means, target_stds)):
                    if std > 0:  # Avoid issues with zero or negative std
                        # Generate the PDF values for the Gaussian
                        pdf_values = norm.pdf(x_range, loc=mean, scale=std)

                        # Plot the Gaussian PDF
                        sns.lineplot(
                            x=x_range,
                            y=pdf_values,
                            label=f"Label {labels[j]}",
                            alpha=0.8
                        )

                        # Fill the area under the curve
                        plt.fill_between(
                            x_range,
                            pdf_values,
                            alpha=0.3,
                        )

                # Use log scale if required
                if use_log_scale:
                    plt.yscale("log")
                    plt.title(f"Gaussian PDFs (Log Scale) of SLAAC Values for Target Class {target_label}", fontsize=16)
                else:
                    plt.title(f"Gaussian PDFs of SLAAC Values for Target Class {target_label}", fontsize=16)

                # Add labels and legend
                plt.xlabel("SLAAC Value", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                plt.legend(title="Computed From Label", fontsize=12, title_fontsize=13)
                plt.tight_layout()
                plt.savefig(f"graphs/{DATASET_NAME}_SLAAC_KDE.png")
                plt.show()

    # 7) Confidence score = min(all_scores); threshold for acceptance is 0.5
    if len(all_scores) == 0:
        # No valid scores => can't compute confidence
        return False, 0.0, slarc_params

    confidence_score = min(all_scores)
    # Hypotheses considered met if confidence >= 0.5
    return (confidence_score >= 0.5), confidence_score, slarc_params

###########################################################
#   COMPLEMENTARY LIKELIHOOD COST
###########################################################

def likelihood_true_approach(x_vec, slarc_params):
    """
    Implements eqs. (5.12)–(5.15) using the actual M, S matrices from check_SLARC_hypotheses.

    - For each label k in slarc_params["labels_order"]:
      1) Compute slaac_val = SLAAC_lk(x_vec).
      2) Let M_kk = slarc_params["M"][k_idx, k_idx], S_kk = slarc_params["S"][k_idx, k_idx],
         where k_idx is the index of label k in labels_order.
      3) C_k(x) = 1 - NormalCDF(slaac_val, mean=M_kk, std=S_kk).
      4) L_k(x) = C_k(x) / (1 + sum_{s != k} C_s(x)).

    :param x_vec: pd.Series (1 row) or numpy array representing the sample.
    :param slarc_params: dict containing
         - "labels_order": array/list of all labels in sorted order
         - "M": 2D array, M[i, j] = E[SLAAC_{li}(X_{lj})]
         - "S": 2D array, S[i, j] = Std[SLAAC_{li}(X_{lj})]
         - plus the usual "mu", "sigma", "min", "max", "omega" for each label key if needed by SLAAC.
    :return: dict { label: L_k(x_vec) } for each label in "labels_order".
    """
    labels_order = slarc_params["labels_order"]  # This is already a sorted list/array of label IDs

    M = slarc_params["M"]
    S = slarc_params["S"]

    # Now loop over labels_order instead of sorted(slarc_params.keys())
    C_vals = {}
    for k_idx, lbl in enumerate(labels_order):
        slaac_val = SLAAC(x_vec, lbl, slarc_params)
        m_kk = M[k_idx, k_idx]
        s_kk = S[k_idx, k_idx]
        c_k = survival_normal(slaac_val, m_kk, s_kk)
        C_vals[lbl] = c_k

    # Next compute L_k(x)
    L_vals = {}
    for k_idx, lbl in enumerate(labels_order):
        c_k = C_vals[lbl]
        sum_others = sum(C_vals[o_lbl] for o_lbl in labels_order if o_lbl != lbl)
        L_vals[lbl] = c_k / (1.0 + sum_others)

    return L_vals



def survival_normal(x, mean, std):
    """
    Return the survival function 1 - CDF for a normal( mean, std^2 ) at value x:
      survival_normal(x) = 1 - Phi((x - mean)/std)
    """
    if std < 1e-15:
        # degenerate case: if std=0, interpret as a delta at mean
        # If x < mean => survival=1, else 0. Or do a small fallback.
        return 1.0 if x < mean else 0.0

    # Convert x -> z
    z = (x - mean)/(std * sqrt(2.0))
    # Normal CDF can be computed from erf:
    cdf_val = 0.5*(1.0 + erf(z))
    return 1.0 - cdf_val

def complementary_likelihood_cost_function(x_candidate, target_label, slarc_params):
    """
    The "true" cost function = 1 - L_{target_label}(x_candidate).

    L_{target_label}(x_candidate) is computed by the new, thorough approach 
    with eq. (5.12)–(5.15). 
    """
    L_vals = likelihood_true_approach(x_candidate, slarc_params)
    return 1.0 - L_vals[target_label]

def SLAAC_cost(x0, x1, target_label, slarc_params):
    return SLAAC(x1, target_label, slarc_params)


def final_cost_function(x0, x1, target_label, slarc_params):
    """
    Must have signature cost(x0, x1) for the attacks.
    Uses global TARGET_LABEL, SLARC_PARAMS.
    """
    return complementary_likelihood_cost_function(x1, target_label, slarc_params)



###########################################################
#   HAR PROTOCOL
###########################################################
def HAR_protocol(
    X,
    y,
    target_class=0,
    estimator=None,
    random_state=42,
    rigid=False,
    test_size=STANDARD_TEST_SIZE,  # e.g. 0.2
    adv_samples=100,
    constraints=None,
    neighborhood=None,
    attacks=None,
    omega_k_dict=None,
    mutual_info_max_samples=100_000,
    use_joblib=True,
    heuristic_max_iter=10_000,
    heuristic_patience=1000,
    neighborhood_default_max_iter=10_000,
    ignore_SLARC_check=False,
    SLARC_check_technique="BC", # "BC" or "KL"
    SLARC_check_max_samples=100_000,
    SLARC_check_labels_to_check="all",
    precomputed_confidence_score=None,
    precomputed_M=None,
    precomputed_S=None,
    final_cost = "Likelihood",  # "Likelihood" or "SLAAC"
    verbose=2,
    distribution_transform=None,  # (None, "yeo-johnson", or "quantile")
    debug_mode=False,
    stop_after_computing_slarc_params=False,
    dataset_name="Unknown",
):
    """
    HAR protocol:
      1) Validates dataset, splits train/test.
      2) Validates / (optionally) trains 'estimator'.
      3) Excludes any sample from X_test that already has true_label == target_class
         (because we want an attack TOWARDS 'target_class' from a different label).
      4) Picks up to 'adv_samples' from the remaining subset.
      5) For each sample, runs each attack (in parallel if requested),
         recording the adversarial example with the highest L_{target_class}.
      6) Computes the updated HAR formula:
         HAR = (AAcc / n) * Σ [1 - L_{target_class}(x_s)]
         where n = number of attacked samples, AAcc = (# flips) / n.

      Returns a dictionary with:
         - HAR
         - confidence_score
         - adversarial_accuracy
         - avg_likelihood
         - max_likelihood
         - etc.
    """
    global DATASET_NAME
    DATASET_NAME = dataset_name

    # Set module-wide verbosity if needed
    set_verbose(verbose)

    # Decuce target_class_idx from target_class and y
    # (e.g. if target class is "2" but y_train only has labels "0" and "2", the idx should be 1, not 2)
    target_class_idx = y.unique().tolist().index(target_class)

    # -----------------------------
    # 1) ADJUST AND VALIDATE DATASET
    # -----------------------------
    np.random.seed(random_state)  # set random seed

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Possibly transform distribution on X_train, X_test (Yeo-Johnson or Quantile)
    if distribution_transform is not None:

        if constraints is not None and VERBOSE > 0:
            print(
                "[WARNING] A distribution transform is being applied, but "
                "the constraints have not been updated accordingly. They "
                "might no longer match the new data scale or domain."
            )

        if estimator is not None and VERBOSE > 0:
            print(
                "[WARNING] A distribution transform is being applied, but "
                "the estimator has already been trained. Ensure that "
                "the estimator can handle the transformed data."
            )

        if distribution_transform.lower() == "yeo-johnson":
            if VERBOSE > 1:
                print("Applying Yeo-Johnson transform to X_train and X_test.")
            # Yeo-Johnson can handle negative values
            transformer = PowerTransformer(method='yeo-johnson')
            X_train = pd.DataFrame(transformer.fit_transform(X_train), columns=X_train.columns)
            X_test  = pd.DataFrame(transformer.transform(X_test), columns=X_test.columns)

        elif distribution_transform.lower() == "quantile":
            if VERBOSE > 1:
                print("Applying Quantile transform (to normal) to X_train and X_test.")
            # output_distribution='normal' tries to map data -> N(0, 1)
            transformer = QuantileTransformer(
                output_distribution='normal', random_state=random_state
            )
            X_train = pd.DataFrame(transformer.fit_transform(X_train), columns=X_train.columns)
            X_test  = pd.DataFrame(transformer.transform(X_test), columns=X_test.columns)

        else:
            if VERBOSE > 0:
                print(f"[WARNING] Unrecognized distribution transform: {distribution_transform}")

    check1, X, y = check_data(X, y, rigid=rigid)
    if not check1:
        raise HarError("[Stage 1] Invalid dataset for the HAR protocol.")

    # -----------------------------
    # 2) VALIDATE / PREPARE ESTIMATOR
    # -----------------------------
    if estimator is None:
        if VERBOSE > 1:
            print("No estimator provided. Using LGBMClassifier as default.")
        estimator = LGBMClassifier(random_state=random_state, verbose=-1)
        estimator.fit(X_train, y_train)

    check2, estimator = check_estimator(X_train, X_test, y_train, y_test, estimator, rigid=rigid)
    if not check2:
        raise HarError("[Stage 2] Invalid estimator for the HAR protocol.")

    # -----------------------------
    # 3) BUILD ATTACK CANDIDATES
    # -----------------------------
    # Ensure the indices align between X_test and y_test
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    # Exclude any sample from X_test whose true label == target_class
    subset_mask = (y_test != target_class)
    X_candidates = X_test[subset_mask]
    y_candidates = y_test[subset_mask]

    # Possibly pick up to adv_samples from that subset
    if len(X_candidates) == 0:
        raise HarError(
            f"No test samples have a label different from the target_class={target_class}. "
            "Cannot perform a targeted attack."
        )

    # If adv_samples is larger than the subset size, reduce it
    final_num_samples = min(adv_samples, len(X_candidates))
    if final_num_samples < adv_samples and VERBOSE > 1:
        print(f"Number of attack samples reduced to {final_num_samples} due to limited subset size.")

    # Randomly pick final_num_samples from X_candidates
    idx_candidates = np.random.choice(X_candidates.shape[0], final_num_samples, replace=False)
    X_attack = X_candidates.iloc[idx_candidates].copy()
    y_true   = y_candidates.iloc[idx_candidates].copy()

    # -----------------------------
    # 4) CHECK CONSTRAINTS, NEIGHBORHOOD, ATTACKS
    # -----------------------------
    if constraints is None:
        clip_min = X_train.quantile(0.01)
        clip_max = X_train.quantile(0.99)
        constraints = {
            "equality": [],
            "inequality": [],
            "clip_min": list(clip_min),
            "clip_max": list(clip_max),
            "categorical": [None] * X_train.shape[1]
        }
    check3 = simple_check_constraints(constraints)
    if not check3:
        raise HarError("[Stage 3] Invalid constraints for the HAR protocol.")

    if neighborhood is None:
        neighborhood = Radar(constraints, max_iter_generation=neighborhood_default_max_iter, enable_warning_message=(VERBOSE>1))

    if attacks is None:
        attacks = [
            HillClimbing(estimator=estimator, verbose=(VERBOSE>1)),
            SimulatedAnnealing(estimator=estimator, verbose=(VERBOSE>1)),
            TabuSearch(estimator=estimator, verbose=(VERBOSE>1))
        ]
    check4 = check_neighborhood(neighborhood) and check_attacks(attacks)
    if not check4:
        raise HarError("[Stage 4] Invalid neighborhood or attacks for the HAR protocol.")

    # -----------------------------
    # 5) SLARC WEIGHTS & CHECKS
    # -----------------------------
    if omega_k_dict is None:
        N = mutual_info_max_samples
        if X_train.shape[0] > N:
            if VERBOSE > 1:
                print(f"Training set too large for mutual_info. Using {N} random samples.")
            idx_small = np.random.choice(X_train.shape[0], N, replace=False)
            X_train_small = X_train.iloc[idx_small]
            y_train_small = y_train.iloc[idx_small]
        else:
            X_train_small = X_train
            y_train_small = y_train

        mi = mutual_info_classif(X_train_small, y_train_small)
        mi = pd.Series(mi, index=X_train_small.columns)
        mi = mi / mi.sum()
        omega_k_dict = mi.to_dict()

    check5, omega_k_dict = check_SLARC_weights(omega_k_dict, X_train, rigid=rigid)
    if not check5:
        raise HarError("[Stage 5] Invalid feature weights for the HAR protocol.")

    # -----------------------------
    # 6) SLARC Hypotheses Check
    # -----------------------------
    if precomputed_confidence_score is None and not ignore_SLARC_check:
        check6, confidence_score, slarc_params = check_SLARC_hypotheses(X_train, y_train,
                                                          omega_k_dict,
                                                          target_class_idx,
                                                          target_class,
                                                          SLARC_check_technique,
                                                          SLARC_check_max_samples,
                                                          SLARC_check_labels_to_check,
                                                          precomputed_M=precomputed_M,
                                                          precomputed_S=precomputed_S)
        if VERBOSE > 0:
            print(f"\n--- Confidence score: {confidence_score:.4f} ---")
        if not check6:
            raise HarError("[Stage 6] SLARC hypotheses are not met for this dataset." +
                           " You can set a distribution transform among 'yeo-johnson' or 'quantile'" +
                           " to try to meet the hypotheses, or set 'ignore_SLARC_check=True' to skip this check.")
    
    else:
        _, _, slarc_params = check_SLARC_hypotheses(X_train, y_train,
                                        omega_k_dict,
                                        target_class_idx,
                                        target_class,
                                        SLARC_check_technique,
                                        SLARC_check_max_samples,
                                        SLARC_check_labels_to_check,
                                        precomputed_M=precomputed_M,
                                        precomputed_S=precomputed_S,
                                        stop_after_computing_slarc_params=True) # SLARC_PARAMS is set here
        
        if precomputed_confidence_score is not None:
            confidence_score = precomputed_confidence_score
            print(f"\n--- Precomputed confidence score: {precomputed_confidence_score:.4f} ---")
        else:
            confidence_score = None
            if VERBOSE > 0:
                print("\n--- SLARC check ignored ---")
        
    if stop_after_computing_slarc_params:
        return {
            "HAR": None,
            "confidence_score": confidence_score,
            "adversarial_accuracy": None,
            "avg_likelihood": None,
            "max_likelihood": None,
            "slarc_params": slarc_params
        }

    # ------------------------------------------------------
    #    All checks passed -> Perform targeted attacks
    # ------------------------------------------------------

    if final_cost == "SLAAC":
        cost_function = SLAAC_cost
    elif final_cost == "Likelihood":
        cost_function = final_cost_function
    else:
        raise HarError(f"Unknown cost function: {final_cost}. Use 'SLAAC' or 'Likelihood'.")

    # We'll collect (did_flip, best_L_val) for each attacked sample

    def process_one_sample(i):
        """
        For sample i:
          - Run each attack trying to flip it to 'target_class'.
          - Among all successful attacks, pick the one yielding the highest L_{target_class}.
          - Return (did_flip, best_L_val).
            * did_flip=1 if we got adv_pred == target_class, else 0
            * best_L_val= L_{target_class}(best_sample) if flip succeeded, else 0.0
        """
        x0 = X_attack.iloc[i].copy()
        y0 = y_true.iloc[i]

        # The initial prediction might or might not be target_class,
        # but we know y0 != target_class from above filtering.
        best_adv_sample = None
        best_L_val = -float("inf")

        if VERBOSE > 1:
            print(f"\n[Sample index={i}] True label={y0}")

        for attack in attacks:
            if VERBOSE > 1:
                print(f"  Running attack: {attack.__class__.__name__}")
            try:
                adv_x, best_cost, total_queries = attack.run(
                    input=x0,
                    cost_function=lambda xx1, xx0: final_cost_function(xx0, xx1, target_class, slarc_params),  # must have signature: cost(x0, x_candidate)
                    targeted=True,
                    specific_class=target_class,
                    neighborhood=neighborhood,
                    max_iter=heuristic_max_iter,
                    patience=heuristic_patience,
                )
                adv_pred = estimator.predict(adv_x.values.reshape(1, -1))[0]

                if adv_pred == target_class:
                    # Attack succeeded => check the target_class likelihood
                    L_vals = likelihood_true_approach(adv_x, slarc_params)
                    L_tgt  = L_vals[target_class]
                    slaac = SLAAC(adv_x, target_class, slarc_params)
                    if VERBOSE > 1:
                        print(f"    Attack success: L_target={L_tgt:.4f} SLAAC={slaac:.4f}")
                    if L_tgt > best_L_val:
                        best_L_val = L_tgt
                        best_adv_sample = adv_x.copy()
            except Exception as e:
                if debug_mode:
                    raise(e)
                if VERBOSE > 1:
                    print(f"    Attack error: {e}")

        # Also check if x0 is already predicted as target_class
        # Possibly it yields a bigger L_{target_class} than the attacks
        x0_pred = estimator.predict(x0.values.reshape(1, -1))[0]
        if x0_pred == target_class:
            L_vals_x0 = likelihood_true_approach(x0, slarc_params)
            L_x0 = L_vals_x0[target_class]
            if L_x0 > best_L_val:
                best_L_val = L_x0
                best_adv_sample = x0.copy()
                if VERBOSE > 1:
                    print(f"    x0 was already target_class with L_target={L_x0:.4f}")

        if best_adv_sample is not None:
            return (1, best_L_val)  # flipped
        else:
            return (0, 0.0)         # no success

    # Decide whether to run in parallel or not
    if use_joblib:
        if VERBOSE > 1:
            print("\n== Parallel attack processing with joblib ==\n")
        results = Parallel(n_jobs=-1)(
            delayed(process_one_sample)(i) for i in range(final_num_samples)
        )
    else:
        if VERBOSE > 1:
            print("\n== Single-threaded attack processing ==\n")
        results = []
        for i in range(final_num_samples):
            results.append(process_one_sample(i))

    # Now compute new HAR formula:
    #   HAR = (AAcc / n) * Σ [1 - L_{target_class}(x_s)],
    #   where n = final_num_samples, AAcc = (#flips / n).
    num_flips = sum(did_flip for (did_flip, _) in results)
    success_rate = num_flips / final_num_samples
    AAcc = 1 - success_rate
    sum_1_minus_L = 0.0
    all_likelihoods = []

    for (did_flip, best_L_val) in results:
        # best_L_val = L_{target_class} of whichever adv sample had max likelihood
        #  or 0.0 if no success
        all_likelihoods.append(best_L_val)
        sum_1_minus_L += (1.0 - best_L_val)

    HAR_value = (AAcc / final_num_samples) * sum_1_minus_L
    adversarial_accuracy = 1.0 - AAcc  # how many remain correct

    if len(all_likelihoods) == 0:
        raise HarError("No attacked samples or no results to compute HAR.")

    avg_likelihood = np.mean(all_likelihoods)
    max_likelihood = np.max(all_likelihoods)

    if VERBOSE > 0:
        print("\n=== HAR protocol results ===")
        print(f"Attacked samples                = {final_num_samples}")
        print(f"Attack success rate (AAcc)      = {AAcc:.4f}")
        print(f"Adversarial accuracy            = {adversarial_accuracy:.4f}")
        #print(f"Sum(1 - L_target)               = {sum_1_minus_L:.4f}")
        print(f"Average likelihood (target)     = {avg_likelihood:.4f}")
        print(f"Maximum likelihood (target)     = {max_likelihood:.4f}")
        print(f"HAR score                       = {HAR_value:.4f}")

    return {
        "HAR": HAR_value,
        "confidence_score": confidence_score,
        "adversarial_accuracy": adversarial_accuracy,
        "avg_likelihood": avg_likelihood,
        "max_likelihood": max_likelihood
    }


from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from joblib import Parallel, delayed

###########################################################
#   CUSTOM COST PROTOCOL
###########################################################
def cost_l2_norm(x0, x1):
    """
    L2 norm between x0 and x1, interpreted as cost for images.
    Minimizing cost => we want x1 to be close to x0 in pixel space.
    Typically used if 'images=True'.
    """
    arr0 = np.array(x0).flatten()
    arr1 = np.array(x1).flatten()
    return np.linalg.norm(arr1 - arr0, ord=2)

def custom_cost_protocol(
    X,
    y,
    cost_profit_func=None,
    targeted=False,
    target_class=0,
    estimator=None,
    images=False,
    image_shape=None,
    adv_samples=10,
    constraints=None,
    neighborhood=None,
    attacks=None,
    random_state=42,
    test_size=STANDARD_TEST_SIZE,
    rigid=False,
    use_joblib=True,
    heuristic_max_iter=100,
    heuristic_patience=100,
    neighborhood_default_max_iter=1000,
    skip_natural_misclass=False,
    trivial_success_if_already_target=True,
    verbose=2,
    dataset_name="Unknown",
):
    """
    A simpler protocol than HAR, using a custom cost function:

    1) If images=True => cost = L2_norm(x0, x1).
       Else (tabular data) => cost = - profit_func(x1), success if profit_func(x1)>0.

    2) 'targeted' indicates whether we want to flip to 'target_class' or just cause
       any misclassification.

    3) Handling natural misclassification:
       - If 'skip_natural_misclass=True', we skip samples that are already misclassified
         (untargeted) or already predicted as 'target_class' (targeted).
       - If 'trivial_success_if_already_target=True' and targeted, we treat samples
         that are already predicted as 'target_class' as trivially attacked => success.

    4) If images=True and 'image_shape' is provided, we use it to reshape images
       for the side-by-side display. Otherwise, we attempt an automatic shape guess.

    5) We do minimal checks on the data and estimator. The user can pass constraints,
       neighborhood, and attack classes. Or we create defaults if none are provided.

    6) Parallelization:
       - If 'use_joblib=True', we process each sample in parallel with joblib.

    Returns a dictionary with:
       {
         "success_rate": ...,
         "avg_cost": ...,
         "min_cost": ...,
         "samples_attacked": ...,
         "results": List of (best_candidate, cost, success)
       }
    """
    global DATASET_NAME
    DATASET_NAME = dataset_name

    set_verbose(verbose)

    sample_number = adv_samples

    # -----------------------------
    # 1) VALIDATE DATASET
    # -----------------------------
    np.random.seed(random_state)
    check1, X, y = check_data(X, y, rigid=rigid)
    if not check1:
        raise HarError("[Stage 1] Invalid dataset for the custom protocol.")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # -----------------------------
    # 2) VALIDATE / PREPARE ESTIMATOR
    # -----------------------------
    if estimator is None:
        if VERBOSE > 1:
            print("[CustomCost] No estimator provided. Using LGBMClassifier as default.")
        estimator = LGBMClassifier(random_state=random_state)

    check2, estimator = check_estimator(X_train, X_test, y_train, y_test, estimator, rigid=rigid)
    if not check2:
        raise HarError("[Stage 2] Invalid estimator for the custom protocol.")

    # -----------------------------
    # 3) BUILD ATTACK CANDIDATES
    # -----------------------------
    # If 'targeted=True', we exclude samples that already have label == target_class
    # because we want to flip from different class => target_class.
    # If 'untargeted', we can keep all samples. But let's mimic HAR logic:
    if targeted:
        subset_mask = (y_test != target_class)
    else:
        # we can just use everything
        subset_mask = np.ones(len(y_test), dtype=bool)  # all True
    X_candidates = X_test[subset_mask]
    y_candidates = y_test[subset_mask]

    if len(X_candidates) == 0:
        raise HarError(
            f"No test samples available for attack. Possibly all had label={target_class} while targeted={targeted}."
        )

    if sample_number > len(X_candidates):
        sample_number = len(X_candidates)
        if VERBOSE > 1:
            print(f"[CustomCost] Adjusting sample_number to {sample_number}, since subset is smaller.")

    idx = np.random.choice(X_candidates.shape[0], sample_number, replace=False)
    X_attack = X_candidates.iloc[idx].copy()
    y_true   = y_candidates.iloc[idx].copy()

    # Possibly skip "naturally misclassified" or "naturally = target_class" samples
    # if user wants skip_natural_misclass.
    final_indices = []
    for i in range(sample_number):
        x0_orig = X_attack.iloc[i]
        pred = estimator.predict(x0_orig.values.reshape(1, -1))[0]
        if targeted:
            # If targeted => we want to flip to target_class.
            # If skip_natural_misclass => skip if 'pred==target_class'.
            if pred == target_class:
                if skip_natural_misclass:
                    continue
                # else treat it as trivial success if user wants
                # or skip it. We'll handle in the process
        else:
            # untargeted => skip if 'pred != y_true'?
            if skip_natural_misclass and (pred != y_true.iloc[i]):
                continue
        final_indices.append(i)

    if not final_indices:
        raise HarError("[CustomCost] All chosen samples are naturally misclassified; none left to attack.")

    # -----------------------------
    # 4) CHECK CONSTRAINTS, NEIGHBORHOOD, ATTACKS
    # -----------------------------
    if constraints is None:
        clip_min = X_train.quantile(0.01)
        clip_max = X_train.quantile(0.99)
        constraints = {
            "equality": [],
            "inequality": [],
            "clip_min": list(clip_min),
            "clip_max": list(clip_max),
            "categorical": [None] * X_train.shape[1]
        }
    check3 = simple_check_constraints(constraints)
    if not check3:
        raise HarError("[Stage 3] Invalid constraints for the custom protocol.")

    if neighborhood is None:
        neighborhood = Radar(constraints, max_iter_generation=neighborhood_default_max_iter, enable_warning_message=(VERBOSE>1))

    if attacks is None:
        attacks = [
            HillClimbing(estimator=estimator, verbose=(VERBOSE>1)),
            SimulatedAnnealing(estimator=estimator, verbose=(VERBOSE>1)),
            TabuSearch(estimator=estimator, verbose=(VERBOSE>1))
        ]
    check4 = check_neighborhood(neighborhood) and check_attacks(attacks)
    if not check4:
        raise HarError("[Stage 4] Invalid neighborhood or attacks for the custom protocol.")

    # -----------------------------
    # 5) DEFINE COST FUNCTION
    # -----------------------------
    if images:
        if VERBOSE > 1:
            print("[custom_cost_protocol] Using L2 norm for images.")
        def custom_cost(x0, x1):
            return cost_l2_norm(x0, x1)
    else:
        if cost_profit_func is None:
            raise HarError("[custom_cost_protocol] No 'cost_profit_func' for tabular data.")
        if VERBOSE > 1:
            print("[custom_cost_protocol] Using profit-based cost for tabular data.")
        def custom_cost(x0, x1):
            return cost_profit_func(x0, x1)

    # -----------------------------
    # 6) ATTACK each sample
    # -----------------------------
    def process_one_sample(i):
        """
        For sample index i in X_attack, run all attacks, pick the best solution (lowest cost).
        Return (best_candidate, best_cost, success).
        """

        x0_orig = X_attack.iloc[i]
        y0      = y_true.iloc[i]

        # Check if it's trivially success (targeted & pred==target_class)
        pred_before = estimator.predict(x0_orig.values.reshape(1, -1))[0]
        if trivial_success_if_already_target and ((targeted and pred_before == target_class)
                                                  or (not targeted and pred_before != y0)):
            # We do not need to run anything => success
            return x0_orig, 0.0, True

        best_candidate = None
        best_cost = float("inf")

        # Run all attacks
        for attack in attacks:
            try:
                result = attack.run(
                    input=x0_orig,
                    cost_function=custom_cost,
                    targeted=targeted,
                    specific_class=target_class,
                    neighborhood=neighborhood,
                    max_iter=heuristic_max_iter,
                    patience=heuristic_patience
                )
                adv_x, cst, total_queries = result

                if cst < best_cost:
                    best_cost = cst
                    best_candidate = adv_x.copy()

            except Exception as e:
                if VERBOSE > 1:
                    print(f"[custom_cost_protocol] Attack error: {e}")
                continue

        # Evaluate success
        if best_candidate is None:
            # Attack never returned anything valid
            return None, float("inf"), False

        if images:
            # For images => check if label changed or target label is reached
            pred_after = estimator.predict(best_candidate.values.reshape(1, -1))[0]
            if targeted:
                success = (pred_after == target_class)
            else:
                success = (pred_after != pred_before)
        else:
            # Tabular => success if profit_func(...) > 0
            if cost_profit_func is not None:
                prof = -best_cost
                success = (prof > 0)
            else:
                success = False  # fallback

        return best_candidate, best_cost, success

    # Decide parallel or serial
    if use_joblib:
        if VERBOSE > 1:
            print("[custom_cost_protocol] Parallel processing with joblib.")
        results = Parallel(n_jobs=-1)(
            delayed(process_one_sample)(i) for i in final_indices
        )
    else:
        if VERBOSE > 1:
            print("[custom_cost_protocol] Processing in single thread.")
        results = []
        for i in final_indices:
            results.append(process_one_sample(i))

    # Summarize
    success_count = sum(r[2] for r in results)
    attacked_count = len(final_indices)
    success_rate  = success_count / attacked_count if attacked_count>0 else 0.0
    all_costs     = [r[1] for r in results if r[1] != float("inf")]
    avg_cost      = np.mean(all_costs) if len(all_costs)>0 else float("nan")
    min_cost      = np.min(all_costs) if len(all_costs)>0 else float("nan")

    if VERBOSE > 0:
        print("\n=== Custom Cost Protocol Results ===")
        print(f"Attacked samples: {attacked_count}")
        print(f"Success count:    {success_count}")
        print(f"Success rate:     {success_rate:.4f}")
        print(f"Average cost:     {avg_cost:.4f}")
        print(f"Minimum cost:     {min_cost:.4f}")

    # If images=True and VERBOSE>0, display final images side-by-side
    if images and VERBOSE > 0:
        for idx_in_list, (best_x, cst, success) in enumerate(results):
            if best_x is None:
                continue

            # Original sample from X_attack
            i_attack = final_indices[idx_in_list]
            x0_orig = X_attack.iloc[i_attack]
            arr_orig = np.array(x0_orig)
            arr_adv  = np.array(best_x)

            if arr_orig.size != arr_adv.size:
                if VERBOSE > 1:
                    print(f"Skipping sample {i_attack}: mismatch in size {arr_orig.size} vs {arr_adv.size}.")
                continue

            # Figure out shape
            if image_shape is not None:
                (h, w, c) = image_shape
            else:
                # auto-guess
                n = arr_orig.size
                side = int(sqrt(n))
                if side*side == n:
                    h, w, c = side, side, 1
                elif n == 3072:
                    h, w, c = 32, 32, 3
                else:
                    # fallback
                    h, w, c = 28, 28, 1

            try:
                arr_orig_reshaped = arr_orig.reshape(h, w, c)
                arr_adv_reshaped  = arr_adv.reshape(h, w, c)
            except Exception as e:
                if VERBOSE > 1:
                    print(f"[custom_cost_protocol] reshape error: {e}")
                continue

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(arr_orig_reshaped.squeeze(), cmap="gray" if c == 1 else None)
            axes[0].set_title(f"Original\nTrue={y_true.iloc[idx_in_list]}")
            axes[0].axis("off")

            # Possibly show predicted label
            if estimator is not None and hasattr(estimator, "predict"):
                pred_after = estimator.predict(best_x.values.reshape(1, -1))[0]
                axes[1].set_title(f"Adversarial\nPred={pred_after}, Cost={cst:.2f}, Success={success}")
            else:
                axes[1].set_title(f"Adversarial\nCost={cst:.2f}\nSuccess={success}")

            axes[1].imshow(arr_adv_reshaped.squeeze(), cmap="gray" if c==1 else None)
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(f"graphs/{DATASET_NAME}_success_cost_{cst}.png")
            plt.show()

    return {
        "success_rate": success_rate,
        "avg_cost": avg_cost,
        "min_cost": min_cost,
        "samples_attacked": attacked_count,
        "results": results  # (best_candidate, cost, success) for each attacked sample
    }
