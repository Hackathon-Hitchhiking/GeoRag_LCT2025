"""Сопоставление локальных признаков с геометрической проверкой."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import LightGlue

from .features import LocalFeatureSet


@dataclass(slots=True)
class MatchScore:
    """Статистика сопоставления между двумя наборами признаков."""

    matches: int
    total_query: int
    total_candidate: int
    mean_score: float
    inliers: int
    inlier_ratio: float
    geometric_score: float

    @property
    def match_ratio(self) -> float:
        """Доля сопоставленных ключевых точек."""

        if self.matches == 0:
            return 0.0
        denom = max(1, min(self.total_query, self.total_candidate))
        return float(self.matches) / float(denom)

    @property
    def local_score(self) -> float:
        """Агрегированный показатель качества матчинга."""

        return self.match_ratio * self.mean_score

    @property
    def geometric_strength(self) -> float:
        """Интегральная геометрическая оценка (инлайеры × качество)."""

        return self.geometric_score


class LightGlueMatcher:
    """Высокоуровневый интерфейс для LightGlue с верификацией геометрии."""

    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._matcher = LightGlue(features="superpoint").to(device).eval()

    def match(self, query: LocalFeatureSet, candidate: LocalFeatureSet) -> MatchScore:
        """Вычислить сопоставление двух наборов признаков."""

        if query.keypoints_count == 0 or candidate.keypoints_count == 0:
            return MatchScore(
                matches=0,
                total_query=query.keypoints_count,
                total_candidate=candidate.keypoints_count,
                mean_score=0.0,
                inliers=0,
                inlier_ratio=0.0,
                geometric_score=0.0,
            )

        inputs = {
            "image0": query.to_lightglue_inputs(self._device),
            "image1": candidate.to_lightglue_inputs(self._device),
        }
        with torch.inference_mode():
            matches = self._matcher(inputs)

        matches0 = matches["matches0"][0]
        scores0 = matches["matching_scores0"][0]
        valid = matches0 > -1
        matched = int(valid.sum().item())
        if matched == 0:
            return MatchScore(
                matches=0,
                total_query=query.keypoints_count,
                total_candidate=candidate.keypoints_count,
                mean_score=0.0,
                inliers=0,
                inlier_ratio=0.0,
                geometric_score=0.0,
            )

        matched_scores = scores0[valid].detach().cpu().numpy().astype(np.float32)
        mean_score = float(np.clip(np.mean(matched_scores), 0.0, 1.0)) if matched_scores.size else 0.0

        query_indices = torch.arange(matches0.shape[0], device=matches0.device)[valid]
        candidate_indices = matches0[valid].detach().cpu().numpy().astype(np.int32)
        query_points = query.keypoints[query_indices.detach().cpu().numpy().astype(np.int32)]
        candidate_points = candidate.keypoints[candidate_indices]

        inliers = 0
        inlier_ratio = 0.0
        geometric_score = 0.0
        if matched >= 8:
            fundamental, mask = cv2.findFundamentalMat(
                query_points.astype(np.float32),
                candidate_points.astype(np.float32),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
                confidence=0.9999,
                maxIters=10000,
            )
            if fundamental is not None and mask is not None:
                inlier_mask = mask.reshape(-1).astype(bool)
                inliers = int(inlier_mask.sum())
                if inliers:
                    inlier_ratio = float(inliers) / float(matched)
                    inlier_scores = matched_scores[inlier_mask]
                    if inlier_scores.size:
                        geometric_score = float(
                            np.clip(float(np.mean(inlier_scores)), 0.0, 1.0)
                        ) * inlier_ratio

        return MatchScore(
            matches=matched,
            total_query=query.keypoints_count,
            total_candidate=candidate.keypoints_count,
            mean_score=mean_score,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            geometric_score=geometric_score,
        )

    async def amatch(self, query: LocalFeatureSet, candidate: LocalFeatureSet) -> MatchScore:
        """Асинхронный подсчёт сопоставлений."""

        return await asyncio.to_thread(self.match, query, candidate)
