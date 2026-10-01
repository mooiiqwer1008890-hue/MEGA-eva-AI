"""
meta_labeling.py
================

Advanced Meta-Labeling Engine
=============================

الغرض:
-------
هذا الملف يمثل الطبقة الثانوية (Meta-Model) في نظام التداول.

Primary Model:
    يحدد اتجاه الصفقة / الإشارة الأساسية.

Meta Model:
    يتعلم:
        "هل هذه الإشارة تستحق أن نأخذها؟"

المفهوم:
---------
    Primary Signal
          |
          v
    Meta Features
          |
          v
    Meta Classifier
          |
          +---- probability
          |
          +---- confidence
          |
          +---- trade / skip
          |
          +---- position multiplier


المراجع والمنهجية:
------------------
1. Marcos López de Prado
   Advances in Financial Machine Learning (2018)
   Chapter 3:
       - Triple-Barrier Method
       - Learning Side and Size
       - Meta-Labeling

   Chapter 4:
       - Sample Weights
       - Class Imbalance
       - Uniqueness

   Chapter 7:
       - Purged K-Fold
       - Embargo

2. scikit-learn documentation:
   - TimeSeriesSplit
   - Probability Calibration
   - Random Forest

ملاحظات مهمة:
--------------
- النموذج لا يتنبأ بالسوق مباشرة.
- النموذج يتعلم جودة الإشارة الأساسية.
- لا يجوز إدخال أي معلومة لم تكن معروفة وقت فتح الصفقة.
- probability ليست "نسبة ربح مؤكدة".
- يجب اختبار النموذج Out-of-Sample قبل استخدامه بأموال حقيقية.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


# ============================================================
# LOGGING
# ============================================================

log = logging.getLogger("meta_labeling")


# ============================================================
# CONFIG
# ============================================================

TRADES_LOG = os.getenv(
    "TRADES_LOG",
    "trades_log.csv"
)

META_MODEL_FILE = os.getenv(
    "META_MODEL_FILE",
    "meta_model.pkl"
)

META_CALIBRATOR_FILE = os.getenv(
    "META_CALIBRATOR_FILE",
    "meta_calibrator.pkl"
)

META_METADATA_FILE = os.getenv(
    "META_METADATA_FILE",
    "meta_metadata.json"
)


# ------------------------------------------------------------
# Training requirements
# ------------------------------------------------------------

MIN_TRADES_FOR_TRAINING = int(
    os.getenv("META_MIN_TRADES", "80")
)

MIN_CLASS_SAMPLES = int(
    os.getenv("META_MIN_CLASS_SAMPLES", "15")
)

N_SPLITS = int(
    os.getenv("META_N_SPLITS", "5")
)

TIME_GAP = int(
    os.getenv("META_TIME_GAP", "2")
)


# ------------------------------------------------------------
# Decision thresholds
# ------------------------------------------------------------

META_MIN_PROBABILITY = float(
    os.getenv("META_MIN_PROBABILITY", "0.55")
)

META_HIGH_PROBABILITY = float(
    os.getenv("META_HIGH_PROBABILITY", "0.70")
)

META_MAX_PROBABILITY = float(
    os.getenv("META_MAX_PROBABILITY", "0.90")
)


# ------------------------------------------------------------
# Position sizing
# ------------------------------------------------------------

MIN_POSITION_MULTIPLIER = float(
    os.getenv("META_MIN_MULTIPLIER", "0.0")
)

MAX_POSITION_MULTIPLIER = float(
    os.getenv("META_MAX_MULTIPLIER", "1.5")
)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    # Primary signal
    "z_score",
    "abs_z_score",
    "signal_direction",

    # Time
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    # HMM
    "regime_bull",
    "regime_bear",
    "regime_neutral",

    # GARCH / volatility
    "vol_low",
    "vol_normal",
    "vol_high",
    "vol_ratio",

    # Sentiment
    "sentiment_score",

    # Portfolio state
    "active_positions",

    # Historical performance
    "recent_win_rate",
    "recent_avg_pnl",
    "recent_pnl_std",

    # Interaction features
    "z_x_regime",
    "z_x_volatility",
    "sentiment_x_signal",
]


# ============================================================
# UTILITIES
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    تحويل آمن إلى float.
    """

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def clip(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    قص القيمة ضمن المجال.
    """

    return float(
        max(
            low,
            min(
                high,
                value,
            )
        )
    )


def parse_timestamp(
    timestamp: Any
) -> datetime:
    """
    تحويل timestamp إلى datetime.

    في حالة الفشل:
        datetime.utcnow()
    """

    try:

        text = str(timestamp)

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        return datetime.fromisoformat(text)

    except Exception:

        return datetime.utcnow()


def safe_probability(
    value: float
) -> float:
    """
    تنظيف probability.
    """

    return clip(
        safe_float(value, 0.5),
        0.0,
        1.0,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class MetaFeatureEngineer:
    """
    Feature Engineering للـ Meta Model.

    القاعدة الأساسية:

        كل Feature يجب أن تكون معروفة
        عند لحظة فتح الصفقة.

    لذلك لا نستخدم:
        exit price
        future return
        future volatility
        final pnl للصفقة الحالية
        duration للصفقة الحالية
    """

    feature_names = FEATURE_NAMES.copy()

    def __init__(
        self,
        recent_window: int = 20,
    ):

        self.recent_window = int(
            max(
                5,
                recent_window
            )
        )

        log.info(
            "[META] FeatureEngineer initialized "
            f"with {len(self.feature_names)} features"
        )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    @staticmethod
    def _time_features(
        timestamp: Any
    ) -> Tuple[float, float, float, float]:

        dt = parse_timestamp(timestamp)

        hour = (
            dt.hour
            + dt.minute / 60.0
        )

        day = float(
            dt.weekday()
        )

        # Circular encoding.
        # أفضل من استخدام 23 -> 0 كمسافة خطية.

        hour_angle = (
            2.0
            * np.pi
            * hour
            / 24.0
        )

        day_angle = (
            2.0
            * np.pi
            * day
            / 7.0
        )

        return (
            float(np.sin(hour_angle)),
            float(np.cos(hour_angle)),
            float(np.sin(day_angle)),
            float(np.cos(day_angle)),
        )

    # --------------------------------------------------------
    # Recent performance
    # --------------------------------------------------------

    def _recent_statistics(
        self,
        recent_trades: Optional[List[float]],
    ) -> Tuple[float, float, float]:

        if not recent_trades:

            return (
                0.5,
                0.0,
                0.0,
            )

        values = np.asarray(
            [
                safe_float(x)
                for x in recent_trades[-self.recent_window:]
            ],
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:

            return (
                0.5,
                0.0,
                0.0,
            )

        win_rate = float(
            np.mean(values > 0)
        )

        avg_pnl = float(
            np.mean(values)
        )

        pnl_std = float(
            np.std(values)
        )

        return (
            clip(win_rate, 0.0, 1.0),
            avg_pnl,
            pnl_std,
        )

    # --------------------------------------------------------
    # Single observation
    # --------------------------------------------------------

    def extract_features(
        self,
        z_score: float,
        timestamp: Any,
        regime: str = "Neutral",
        vol_regime: str = "NORMAL",
        sentiment_score: float = 0.0,
        active_positions: int = 0,
        recent_trades: Optional[List[float]] = None,
        vol_ratio: float = 1.0,
        signal_direction: int = 0,
    ) -> np.ndarray:
        """
        استخراج Feature Vector واحد.

        signal_direction:
            +1 = Long
            -1 = Short
             0 = غير معروف
        """

        z = safe_float(
            z_score
        )

        abs_z = abs(z)

        direction = int(
            np.sign(
                safe_float(
                    signal_direction
                )
            )
        )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        (
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
        ) = self._time_features(
            timestamp
        )

        # ----------------------------------------------------
        # HMM regime
        # ----------------------------------------------------

        regime_normalized = str(
            regime
        ).strip().lower()

        regime_bull = float(
            regime_normalized == "bull"
        )

        regime_bear = float(
            regime_normalized == "bear"
        )

        regime_neutral = float(
            not (
                regime_bull
                or regime_bear
            )
        )

        # ----------------------------------------------------
        # Volatility regime
        # ----------------------------------------------------

        vol_normalized = str(
            vol_regime
        ).strip().upper()

        vol_low = float(
            vol_normalized == "LOW"
        )

        vol_normal = float(
            vol_normalized == "NORMAL"
        )

        vol_high = float(
            vol_normalized == "HIGH"
        )

        vr = safe_float(
            vol_ratio,
            1.0
        )

        # Prevent extreme values.
        vr = clip(
            vr,
            0.05,
            10.0,
        )

        # ----------------------------------------------------
        # Sentiment
        # ----------------------------------------------------

        sentiment = clip(
            safe_float(
                sentiment_score
            ),
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Portfolio
        # ----------------------------------------------------

        active = clip(
            safe_float(
                active_positions
            ),
            0.0,
            100.0,
        )

        # ----------------------------------------------------
        # Historical performance
        # ----------------------------------------------------

        (
            recent_win_rate,
            recent_avg_pnl,
            recent_pnl_std,
        ) = self._recent_statistics(
            recent_trades
        )

        # ----------------------------------------------------
        # Interaction terms
        # ----------------------------------------------------

        z_x_regime = (
            z
            * (
                regime_bull
                - regime_bear
            )
        )

        z_x_volatility = (
            z
            * vr
        )

        sentiment_x_signal = (
            sentiment
            * direction
        )

        # ----------------------------------------------------
        # Final vector
        # ----------------------------------------------------

        features = np.array(
            [
                z,
                abs_z,
                direction,

                hour_sin,
                hour_cos,
                day_sin,
                day_cos,

                regime_bull,
                regime_bear,
                regime_neutral,

                vol_low,
                vol_normal,
                vol_high,
                vr,

                sentiment,

                active,

                recent_win_rate,
                recent_avg_pnl,
                recent_pnl_std,

                z_x_regime,
                z_x_volatility,
                sentiment_x_signal,
            ],
            dtype=float,
        )

        # Safety
        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return features

    # --------------------------------------------------------
    # Batch extraction
    # --------------------------------------------------------

    def extract_features_batch(
        self,
        trades_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        استخراج Features من trades_df.

        مهم:
        recent_trades يتم أخذها من الصفقات السابقة فقط.

        لا يتم استخدام PnL الصفقة الحالية
        في Features الخاصة بها.
        """

        if trades_df.empty:

            return np.empty(
                (
                    0,
                    len(self.feature_names)
                ),
                dtype=float,
            )

        df = trades_df.reset_index(
            drop=True
        ).copy()

        features = []

        historical_pnls: List[float] = []

        for i, row in df.iterrows():

            recent = historical_pnls[
                -self.recent_window:
            ]

            vector = self.extract_features(
                z_score=row.get(
                    "z_score",
                    0.0
                ),

                timestamp=row.get(
                    "timestamp_open",
                    row.get(
                        "timestamp",
                        datetime.utcnow().isoformat()
                    )
                ),

                regime=row.get(
                    "regime",
                    "Neutral"
                ),

                vol_regime=row.get(
                    "vol_regime",
                    "NORMAL"
                ),

                sentiment_score=row.get(
                    "sentiment_score",
                    0.0
                ),

                active_positions=row.get(
                    "active_positions",
                    0
                ),

                recent_trades=recent,

                vol_ratio=row.get(
                    "vol_ratio",
                    1.0
                ),

                signal_direction=row.get(
                    "signal_direction",
                    row.get(
                        "side",
                        0
                    )
                ),
            )

            features.append(
                vector
            )

            # مهم:
            # أضف PnL بعد بناء Features
            # حتى لا يدخل PnL الحالي في نفسه.

            historical_pnls.append(
                safe_float(
                    row.get(
                        "pnl_usd",
                        0.0
                    )
                )
            )

        return np.asarray(
            features,
            dtype=float,
        )


# ============================================================
# DATA VALIDATION
# ============================================================

def prepare_training_dataframe(
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    تنظيف وتجهيز سجل الصفقات.
    """

    if trades_df is None:

        raise ValueError(
            "trades_df is None"
        )

    df = trades_df.copy()

    if "pnl_usd" not in df.columns:

        raise ValueError(
            "trades_log يجب أن يحتوي على pnl_usd"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    timestamp_column = None

    for candidate in (
        "timestamp_open",
        "timestamp",
        "open_time",
        "entry_time",
    ):

        if candidate in df.columns:

            timestamp_column = candidate
            break

    if timestamp_column:

        parsed = pd.to_datetime(
            df[timestamp_column],
            errors="coerce",
            utc=True,
        )

        df["_parsed_timestamp"] = parsed

        df = df.sort_values(
            "_parsed_timestamp"
        )

    else:

        log.warning(
            "[META] No timestamp column found. "
            "Assuming CSV order is chronological."
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "pnl_usd",
        "z_score",
        "sentiment_score",
        "active_positions",
        "vol_ratio",
        "signal_direction",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Remove invalid PnL
    # --------------------------------------------------------

    df = df[
        np.isfinite(
            df["pnl_usd"].astype(float)
        )
    ]

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# MODEL
# ============================================================

class MetaLabeler:
    """
    Advanced Meta-Labeling classifier.

    Architecture:

        Features
           |
           v
        Random Forest
           |
           v
        OOF probability
           |
           v
        Probability Calibration
           |
           v
        calibrated probability
           |
           +------> decision
           |
           +------> position size


    لماذا Random Forest؟

    - غير خطي
    - يتعامل مع interactions
    - لا يحتاج StandardScaler
    - robust نسبيًا مع Features مختلفة المقاييس
    - يعطي feature importance

    لكن:
    probability الخام من RF ليست بالضرورة معايرة.

    لذلك نستخدم:
        OOF predictions
              ↓
        Platt / Logistic calibration
    """

    MODEL_VERSION = "3.0"

    def __init__(
        self,
        recent_window: int = 20,
    ):

        self.feature_engineer = MetaFeatureEngineer(
            recent_window=recent_window
        )

        self.model: Optional[
            RandomForestClassifier
        ] = None

        self.calibrator: Optional[
            LogisticRegression
        ] = None

        self.is_trained = False

        self.training_stats: Dict[str, Any] = {}

        self._load()

    # ========================================================
    # BASE MODEL
    # ========================================================

    @staticmethod
    def _create_model() -> RandomForestClassifier:
        """
        إنشاء Random Forest.

        class_weight=balanced
        مهم عندما تكون نتائج الصفقات غير متوازنة.
        """

        return RandomForestClassifier(
            n_estimators=400,

            max_depth=7,

            min_samples_leaf=4,

            max_features="sqrt",

            class_weight="balanced_subsample",

            bootstrap=True,

            random_state=42,

            n_jobs=-1,
        )

    # ========================================================
    # OOF VALIDATION
    # ========================================================

    def _walk_forward_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict]]:

        """
        Walk-forward / Time-Series validation.

        لا نخلط المستقبل بالماضي.

        gap يمنع العينات القريبة جدًا من حد التدريب
        من التأثير مباشرة على الاختبار.
        """

        n_samples = len(X)

        if n_samples < 30:

            raise ValueError(
                "بيانات غير كافية للـwalk-forward"
            )

        splits = min(
            N_SPLITS,
            max(
                2,
                n_samples // 20
            ),
        )

        tscv = TimeSeriesSplit(
            n_splits=splits,
            gap=min(
                TIME_GAP,
                max(
                    0,
                    n_samples // 20
                ),
            ),
        )

        oof_probability = np.full(
            n_samples,
            np.nan,
            dtype=float,
        )

        fold_stats = []

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            tscv.split(X),
            start=1,
        ):

            X_train = X[
                train_idx
            ]

            y_train = y[
                train_idx
            ]

            X_test = X[
                test_idx
            ]

            y_test = y[
                test_idx
            ]

            # لا يمكن تدريب classifier
            # إذا كانت عينة التدريب تحتوي على class واحد.

            if len(
                np.unique(y_train)
            ) < 2:

                log.warning(
                    f"[META] Fold {fold} skipped: "
                    "training contains one class."
                )

                continue

            model = self._create_model()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            oof_probability[
                test_idx
            ] = probabilities

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            stats = {
                "fold": fold,

                "n_train": len(
                    train_idx
                ),

                "n_test": len(
                    test_idx
                ),

                "accuracy": float(
                    accuracy_score(
                        y_test,
                        predictions,
                    )
                ),

                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_test,
                        predictions,
                    )
                ),

                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),

                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),

                "f1": float(
                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    )
                ),

                "brier": float(
                    brier_score_loss(
                        y_test,
                        probabilities,
                    )
                ),
            }

            if len(
                np.unique(y_test)
            ) == 2:

                stats["roc_auc"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                    )
                )

            else:

                stats["roc_auc"] = 0.5

            fold_stats.append(
                stats
            )

            log.info(
                "[META] Fold %d | "
                "AUC=%.3f | "
                "F1=%.3f | "
                "Brier=%.4f",
                fold,
                stats["roc_auc"],
                stats["f1"],
                stats["brier"],
            )

        return (
            oof_probability,
            fold_stats,
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        trades_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        تدريب كامل.

        Pipeline:

        1. تنظيف البيانات
        2. بناء Features
        3. بناء Meta Labels
        4. Walk-forward OOF
        5. Calibration
        6. تدريب النموذج النهائي
        7. Feature importance
        8. حفظ النموذج
        """

        try:

            df = prepare_training_dataframe(
                trades_df
            )

        except Exception as exc:

            log.error(
                "[META] Data preparation failed: %s",
                exc,
            )

            return {
                "success": False,
                "reason": "data_error",
                "error": str(exc),
            }

        n = len(df)

        if n < MIN_TRADES_FOR_TRAINING:

            log.warning(
                "[META] Insufficient trades: "
                f"{n}/{MIN_TRADES_FOR_TRAINING}"
            )

            return {
                "success": False,
                "reason": "insufficient_trades",
                "n_trades": n,
            }

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        X = self.feature_engineer.extract_features_batch(
            df
        )

        # ----------------------------------------------------
        # Meta label
        # ----------------------------------------------------

        y = (
            df["pnl_usd"]
            .astype(float)
            > 0.0
        ).astype(int).values

        positive = int(
            np.sum(y == 1)
        )

        negative = int(
            np.sum(y == 0)
        )

        win_rate = float(
            np.mean(y)
        )

        log.info(
            "[META] Dataset: %d trades | "
            "wins=%d | losses=%d | "
            "win_rate=%.2f%%",
            n,
            positive,
            negative,
            win_rate * 100,
        )

        if (
            positive < MIN_CLASS_SAMPLES
            or negative < MIN_CLASS_SAMPLES
        ):

            return {
                "success": False,
                "reason": "class_imbalance_or_too_few_samples",
                "n_trades": n,
                "wins": positive,
                "losses": negative,
            }

        # ----------------------------------------------------
        # OOF predictions
        # ----------------------------------------------------

        (
            oof_prob,
            fold_stats,
        ) = self._walk_forward_predictions(
            X,
            y,
        )

        valid_mask = np.isfinite(
            oof_prob
        )

        valid_prob = oof_prob[
            valid_mask
        ]

        valid_y = y[
            valid_mask
        ]

        # يجب وجود كلا class في calibration.

        if (
            len(valid_prob) < 20
            or len(np.unique(valid_y)) < 2
        ):

            log.warning(
                "[META] Insufficient OOF observations "
                "for calibration."
            )

            self.calibrator = None

        else:

            # ------------------------------------------------
            # Platt scaling
            # ------------------------------------------------

            # logit(raw probability)
            eps = 1e-6

            raw_logit = np.log(
                np.clip(
                    valid_prob,
                    eps,
                    1.0 - eps,
                )
                /
                np.clip(
                    1.0 - valid_prob,
                    eps,
                    1.0 - eps,
                )
            ).reshape(
                -1,
                1,
            )

            calibrator = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )

            calibrator.fit(
                raw_logit,
                valid_y,
            )

            self.calibrator = calibrator

        # ----------------------------------------------------
        # Final model
        # ----------------------------------------------------

        self.model = self._create_model()

        self.model.fit(
            X,
            y,
        )

        self.is_trained = True

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        importances = self.model.feature_importances_

        feature_importance = dict(
            zip(
                self.feature_engineer.feature_names,
                map(float, importances),
            )
        )

        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # ----------------------------------------------------
        # Aggregate CV statistics
        # ----------------------------------------------------

        if fold_stats:

            metric_names = [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier",
            ]

            cv_scores = {}

            for metric in metric_names:

                values = [
                    fold[metric]
                    for fold in fold_stats
                    if metric in fold
                ]

                if values:

                    cv_scores[metric] = float(
                        np.mean(values)
                    )

        else:

            cv_scores = {}

        # ----------------------------------------------------
        # Calibration diagnostics
        # ----------------------------------------------------

        calibration_brier = None

        if (
            self.calibrator is not None
            and len(valid_prob) > 0
        ):

            calibrated_prob = self._calibrate_probability_array(
                valid_prob
            )

            calibration_brier = float(
                brier_score_loss(
                    valid_y,
                    calibrated_prob,
                )
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        self.training_stats = {

            "success": True,

            "model_version": self.MODEL_VERSION,

            "trained_at": datetime.utcnow().isoformat(),

            "n_trades": int(n),

            "n_features": int(
                X.shape[1]
            ),

            "wins": positive,

            "losses": negative,

            "win_rate": win_rate,

            "cv_scores": cv_scores,

            "folds": fold_stats,

            "oof_samples": int(
                np.sum(valid_mask)
            ),

            "calibration_brier": calibration_brier,

            "feature_importance":
                feature_importance,
        }

        self._save()

        log.info(
            "[META] Training completed successfully."
        )

        return self.training_stats

    # ========================================================
    # CALIBRATION
    # ========================================================

    def _calibrate_probability(
        self,
        probability: float,
    ) -> float:
        """
        تحويل raw probability إلى
        calibrated probability.
        """

        p = safe_probability(
            probability
        )

        if self.calibrator is None:

            return p

        eps = 1e-6

        logit = np.log(
            np.clip(
                p,
                eps,
                1.0 - eps,
            )
            /
            np.clip(
                1.0 - p,
                eps,
                1.0 - eps,
            )
        )

        calibrated = self.calibrator.predict_proba(
            np.array(
                [[logit]],
                dtype=float,
            )
        )[0, 1]

        return safe_probability(
            calibrated
        )

    def _calibrate_probability_array(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:

        return np.asarray(
            [
                self._calibrate_probability(
                    p
                )
                for p in probabilities
            ],
            dtype=float,
        )

    # ========================================================
    # POSITION SIZE
    # ========================================================

    @staticmethod
    def _probability_to_multiplier(
        probability: float,
    ) -> float:
        """
        تحويل الاحتمال إلى حجم تدريجي.

        لا نستخدم:
            0.70 -> 1.5
            0.69 -> 1.0

        بشكل قاطع.

        بدلاً من ذلك:
            probability
                    |
                    v
            continuous sizing

        مع الحفاظ على حدود صارمة.
        """

        p = safe_probability(
            probability
        )

        threshold = META_MIN_PROBABILITY

        if p < threshold:

            return 0.0

        # Normalization:
        #
        # threshold -> 0
        # 0.90      -> 1

        upper = META_MAX_PROBABILITY

        if upper <= threshold:

            return 1.0

        normalized = (
            p - threshold
        ) / (
            upper - threshold
        )

        normalized = clip(
            normalized,
            0.0,
            1.0,
        )

        # Smoothstep
        smooth = (
            normalized
            * normalized
            * (
                3.0
                - 2.0
                * normalized
            )
        )

        multiplier = (
            MIN_POSITION_MULTIPLIER
            +
            smooth
            * (
                MAX_POSITION_MULTIPLIER
                - MIN_POSITION_MULTIPLIER
            )
        )

        return round(
            clip(
                multiplier,
                MIN_POSITION_MULTIPLIER,
                MAX_POSITION_MULTIPLIER,
            ),
            4,
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        z_score: float,
        timestamp: Any,
        regime: str = "Neutral",
        vol_regime: str = "NORMAL",
        sentiment_score: float = 0.0,
        active_positions: int = 0,
        recent_trades: Optional[List[float]] = None,
        vol_ratio: float = 1.0,
        signal_direction: int = 0,
    ) -> Dict[str, Any]:
        """
        التنبؤ بجودة الإشارة.

        Returns:
            probability
            raw_probability
            confidence
            should_trade
            multiplier
            model_status
        """

        # ----------------------------------------------------
        # Model unavailable
        # ----------------------------------------------------

        if (
            not self.is_trained
            or self.model is None
        ):

            return {
                "probability": 0.5,
                "raw_probability": 0.5,
                "confidence": "UNAVAILABLE",
                "should_trade": False,
                "multiplier": 0.0,
                "model_status": "not_trained",
                "reason":
                    "Meta model is not trained.",
            }

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = self.feature_engineer.extract_features(
            z_score=z_score,
            timestamp=timestamp,
            regime=regime,
            vol_regime=vol_regime,
            sentiment_score=sentiment_score,
            active_positions=active_positions,
            recent_trades=recent_trades,
            vol_ratio=vol_ratio,
            signal_direction=signal_direction,
        ).reshape(
            1,
            -1,
        )

        # ----------------------------------------------------
        # Raw prediction
        # ----------------------------------------------------

        raw_probability = float(
            self.model.predict_proba(
                features
            )[0, 1]
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        probability = self._calibrate_probability(
            raw_probability
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if probability >= META_HIGH_PROBABILITY:

            confidence = "HIGH"

        elif probability >= META_MIN_PROBABILITY:

            confidence = "MEDIUM"

        elif probability >= 0.45:

            confidence = "LOW"

        else:

            confidence = "VERY_LOW"

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        should_trade = bool(
            probability
            >= META_MIN_PROBABILITY
        )

        # ----------------------------------------------------
        # Position multiplier
        # ----------------------------------------------------

        multiplier = (
            self._probability_to_multiplier(
                probability
            )
            if should_trade
            else 0.0
        )

        return {

            "probability":
                round(
                    probability,
                    4,
                ),

            "raw_probability":
                round(
                    raw_probability,
                    4,
                ),

            "confidence":
                confidence,

            "should_trade":
                should_trade,

            "multiplier":
                multiplier,

            "model_status":
                "trained",

            "model_version":
                self.MODEL_VERSION,
        }

    # ========================================================
    # SAVE
    # ========================================================

    def _save(self) -> None:
        """
        حفظ model + calibrator + metadata.
        """

        try:

            if self.model is not None:

                joblib.dump(
                    self.model,
                    META_MODEL_FILE,
                )

            if self.calibrator is not None:

                joblib.dump(
                    self.calibrator,
                    META_CALIBRATOR_FILE,
                )

            metadata = {
                "model_version":
                    self.MODEL_VERSION,

                "feature_names":
                    self.feature_engineer.feature_names,

                "training_stats":
                    self.training_stats,

                "config": {
                    "min_trades":
                        MIN_TRADES_FOR_TRAINING,

                    "min_class_samples":
                        MIN_CLASS_SAMPLES,

                    "n_splits":
                        N_SPLITS,

                    "time_gap":
                        TIME_GAP,

                    "min_probability":
                        META_MIN_PROBABILITY,

                    "high_probability":
                        META_HIGH_PROBABILITY,

                    "max_probability":
                        META_MAX_PROBABILITY,
                },
            }

            with open(
                META_METADATA_FILE,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            log.info(
                "[META] Model saved successfully."
            )

        except Exception as exc:

            log.error(
                "[META] Save failed: %s",
                exc,
            )

    # ========================================================
    # LOAD
    # ========================================================

    def _load(self) -> None:
        """
        تحميل النموذج.

        يتم التحقق من:
            model
            metadata
            feature names
        """

        try:

            if not os.path.exists(
                META_MODEL_FILE
            ):

                return

            self.model = joblib.load(
                META_MODEL_FILE
            )

            if os.path.exists(
                META_CALIBRATOR_FILE
            ):

                self.calibrator = joblib.load(
                    META_CALIBRATOR_FILE
                )

            if os.path.exists(
                META_METADATA_FILE
            ):

                with open(
                    META_METADATA_FILE,
                    "r",
                    encoding="utf-8",
                ) as file:

                    metadata = json.load(
                        file
                    )

                saved_features = metadata.get(
                    "feature_names"
                )

                if (
                    saved_features
                    and saved_features
                    != self.feature_engineer.feature_names
                ):

                    log.warning(
                        "[META] Feature schema changed. "
                        "Ignoring old model."
                    )

                    self.model = None
                    self.calibrator = None
                    self.is_trained = False

                    return

                self.training_stats = metadata.get(
                    "training_stats",
                    {},
                )

            self.is_trained = (
                self.model is not None
            )

            if self.is_trained:

                log.info(
                    "[META] Model loaded successfully."
                )

        except Exception as exc:

            log.warning(
                "[META] Model loading failed: %s",
                exc,
            )

            self.model = None
            self.calibrator = None
            self.is_trained = False

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> str:
        """
        ملخص النموذج.
        """

        if not self.is_trained:

            return (
                "⚪ Meta-Labeling\n"
                "   Status: NOT TRAINED"
            )

        stats = (
            self.training_stats
            or {}
        )

        cv = stats.get(
            "cv_scores",
            {},
        )

        importance = stats.get(
            "feature_importance",
            {},
        )

        top_features = list(
            importance.items()
        )[:5]

        features_text = "\n".join(
            [
                f"   • {name}: {value:.4f}"
                for name, value
                in top_features
            ]
        )

        return (
            "🧠 Advanced Meta-Labeling\n"
            "────────────────────────\n"
            f"Version: {self.MODEL_VERSION}\n"
            f"Trades: {stats.get('n_trades', 0)}\n"
            f"Wins: {stats.get('wins', 0)}\n"
            f"Losses: {stats.get('losses', 0)}\n"
            f"Win Rate: "
            f"{stats.get('win_rate', 0):.2%}\n\n"

            f"Walk-Forward AUC: "
            f"{cv.get('roc_auc', 0.5):.3f}\n"

            f"Walk-Forward F1: "
            f"{cv.get('f1', 0):.3f}\n"

            f"Balanced Accuracy: "
            f"{cv.get('balanced_accuracy', 0):.3f}\n"

            f"Raw Brier: "
            f"{cv.get('brier', 0):.4f}\n"

            f"Calibrated Brier: "
            f"{stats.get('calibration_brier', 0):.4f}\n\n"

            "Top Features:\n"
            f"{features_text}"
        )


# ============================================================
# LOAD TRADES
# ============================================================

def load_trades_log(
    path: str = TRADES_LOG,
) -> pd.DataFrame:
    """
    تحميل trades_log.csv.
    """

    if not os.path.exists(path):

        log.warning(
            "[META] trades log not found: %s",
            path,
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            path
        )

        log.info(
            "[META] Loaded %d trades.",
            len(df),
        )

        return df

    except Exception as exc:

        log.error(
            "[META] Failed to load trades: %s",
            exc,
        )

        return pd.DataFrame()


# ============================================================
# SINGLETON
# ============================================================

_meta_labeler: Optional[
    MetaLabeler
] = None


def get_meta_labeler() -> MetaLabeler:
    """
    الحصول على MetaLabeler واحد مشترك.
    """

    global _meta_labeler

    if _meta_labeler is None:

        _meta_labeler = MetaLabeler()

    return _meta_labeler


# ============================================================
# TEST
# ============================================================

def run_test() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    print()
    print("=" * 75)
    print(
        "ADVANCED META-LABELING ENGINE"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_trades_log()

    print(
        f"\n📊 Trades loaded: {len(df)}"
    )

    if df.empty:

        print(
            "\n❌ trades_log.csv غير موجود "
            "أو فارغ."
        )

        return

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    labeler = MetaLabeler()

    result = labeler.train(
        df
    )

    if not result.get(
        "success",
        False,
    ):

        print(
            "\n❌ Training failed:"
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        labeler.summary()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "LIVE PREDICTION TEST"
    )
    print("=" * 75)

    recent_pnl = []

    if "pnl_usd" in df.columns:

        recent_pnl = (
            pd.to_numeric(
                df["pnl_usd"],
                errors="coerce",
            )
            .dropna()
            .tail(20)
            .tolist()
        )

    prediction = labeler.predict(

        z_score=-2.5,

        timestamp=datetime.utcnow().isoformat(),

        regime="Bull",

        vol_regime="NORMAL",

        sentiment_score=0.30,

        active_positions=1,

        recent_trades=recent_pnl,

        vol_ratio=1.0,

        signal_direction=1,
    )

    print(
        json.dumps(
            prediction,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 75)
    print(
        "END"
    )
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test()
