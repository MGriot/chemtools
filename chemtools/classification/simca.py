import numpy as np
import pandas as pd
from chemtools.base.base_models import BaseModel
from chemtools.exploration import PrincipalComponentAnalysis
from chemtools.utils.data import initialize_names_and_counts

class SIMCA(BaseModel):
    """
    Soft Independent Modeling of Class Analogies (SIMCA).

    SIMCA is a supervised classification method that builds a Principal Component
    Analysis (PCA) model for each class in the training set. A new observation is
    classified based on whether its distance to each class model falls within
    a statistical limit.

    This method allows an observation to be assigned to one class, multiple classes,
    or no class at all (outlier).

    Attributes:
        n_components (int): The number of principal components to use for each class model.
        alpha (float): The significance level for calculating statistical distance thresholds.
        class_models (dict): A dictionary to store the fitted PCA model for each class.
    
    References:
        - Wold, S., & Sjöström, M. (1977). SIMCA: A method for analyzing chemical data in terms of similarity and analogy. In ACS symposium series (Vol. 52, pp. 243-282).
    """

    def __init__(self, n_components=2, alpha=0.05):
        super().__init__()
        self.model_name = "SIMCA"
        self.method = "Soft Independent Modeling of Class Analogies"
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.n_components = n_components
        self.alpha = alpha
        self.class_models = {}

    def fit(self, X, y, variables_names=None, objects_names=None):
        """
        Fits a PCA model for each class in the training data.

        Args:
            X (np.ndarray): The training input samples (n_samples, n_features).
            y (np.ndarray): The target values (class labels) (n_samples,).
            variables_names (list, optional): Names of the variables. Defaults to None.
            objects_names (list, optional): Names of the objects. Defaults to None.
        """
        self.X_train = X
        self.y_train = y
        self.variables, self.objects, self.n_variables, self.n_objects = (
            initialize_names_and_counts(X, variables_names, objects_names)
        )
        
        unique_classes = np.unique(y)
        
        for c in unique_classes:
            X_c = X[y == c]
            
            if X_c.shape[0] < 2:
                self.notes.append(f"Warning: Class '{c}' has fewer than 2 samples and will be skipped.")
                continue

            # Ensure n_components is valid for the subset of data
            max_components = min(X_c.shape[0] - 1, X_c.shape[1])
            if self.n_components > max_components:
                self.notes.append(f"Warning: n_components for class '{c}' was reduced to {max_components} due to data shape.")
                n_comp_c = max_components
            else:
                n_comp_c = self.n_components
            
            if n_comp_c <= 0:
                self.notes.append(f"Warning: Class '{c}' cannot be modeled (not enough samples/features).")
                continue

            pca_model = PrincipalComponentAnalysis()
            pca_model.fit(X_c)
            pca_model.reduction(n_components=n_comp_c)
            pca_model.statistics(alpha=self.alpha) # Calculate T2 and Q limits
            
            self.class_models[c] = pca_model
            
        if not self.class_models:
            raise RuntimeError("No valid class models could be built. Check data and class labels.")

        return self

    def predict(self, X_new):
        """
        Classifies new observations based on their distance to each class model.

        Args:
            X_new (np.ndarray): New data to classify (n_new_samples, n_features).

        Returns:
            list: A list of lists, where each inner list contains the class(es)
                  the corresponding sample is assigned to. An empty list means
                  the sample is an outlier.
        """
        if not self.class_models:
            raise RuntimeError("The model has not been fitted yet.")

        predictions = []
        for i in range(X_new.shape[0]):
            sample = X_new[i:i+1, :]
            sample_class_assignments = []
            
            for class_label, pca_model in self.class_models.items():
                # Use the new method in PCA to get statistics for the new sample
                T2_new, Q_new = pca_model.predict_stats(sample)

                # Get the thresholds calculated during fitting
                q_threshold = np.quantile(pca_model.Q, 1 - self.alpha)
                t2_threshold = pca_model.T2_critical_value
                
                if T2_new <= t2_threshold and Q_new <= q_threshold:
                    sample_class_assignments.append(class_label)
            
            predictions.append(sample_class_assignments)
            
        return predictions

    def _get_summary_data(self):
        """Returns a dictionary containing summary data for the model."""
        if not self.class_models:
            return {}
        
        summary = self._create_general_summary(
            self.n_variables,
            self.n_objects,
            No_Classes=f"{len(self.class_models)}",
            n_components_per_class=f"{self.n_components}",
            alpha=f"{self.alpha}"
        )
        
        class_summaries = []
        for class_label, pca_model in self.class_models.items():
            t2_crit = pca_model.T2_critical_value
            q_thresh = np.quantile(pca_model.Q, 1 - self.alpha) if hasattr(pca_model, 'Q') and pca_model.Q is not None and len(pca_model.Q) > 0 else np.nan
            class_summaries.append(
                f"Class '{class_label}': {pca_model.n_component} components, "
                f"T2_crit={t2_crit:.2f}, Q_thresh ({(1-self.alpha)*100:.0f}%)={q_thresh:.2f}"
            )
        
        self.notes = class_summaries
        
        return summary
