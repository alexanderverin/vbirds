"""Сохранить baseline-модель (Этап 3) как переиспользуемый артефакт для API (Этап 8).

Обучает SVM с probability=True (нужно для confidence в ответе API) на кэшированных признаках
train-сплита и дампит {scaler, clf, classes} в ml/models/baseline_svm.joblib.
Лёгкая модель: инференс требует только numpy/PIL/skimage/opencv/sklearn, БЕЗ torch — контейнер лёгкий.

Запуск: python ml/src/save_baseline.py   (нужен ml/reports/baseline_features.npz из baseline.py)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ml" / "reports" / "baseline_features.npz"
OUT = ROOT / "ml" / "models" / "baseline_svm.joblib"
CLASSES = ["fertile", "infertile", "dead"]


def main() -> int:
    if not CACHE.exists():
        print(f"Нет {CACHE}. Сначала: python ml/src/baseline.py")
        return 1
    d = np.load(CACHE, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    tr = split == "train"
    # CalibratedClassifierCV даёт predict_proba (для confidence) без устаревшего SVC(probability=True)
    svc = CalibratedClassifierCV(SVC(kernel="rbf", C=100, random_state=42), ensemble=False)
    model = make_pipeline(StandardScaler(), svc)
    model.fit(X[tr], y[tr])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES,
                 "feat": {"size": 128, "hist_bins": 32, "glcm_levels": 32}}, OUT)
    # быстрая самопроверка на test
    te = split == "test"
    acc = (model.predict(X[te]) == y[te]).mean()
    print(f"Сохранено: {OUT.relative_to(ROOT)} | self-check test accuracy = {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
