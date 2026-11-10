from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def make_logistic_pipeline(C: float = 1.0) -> Pipeline:
    """Build scaled, L2-regularized Logistic Regression."""
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])


def make_random_forest(
    max_depth: int = 8,
    min_samples_leaf: int = 20,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Build a forest with configurable complexity."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def make_pca_logistic_pipeline(
    n_components: int,
    C: float = 1.0,
) -> Pipeline:
    """Build training-fitted scaling, PCA, and Logistic Regression."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="full")),
        (
            "model",
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])