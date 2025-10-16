"""Сопоставление локальных признаков с геометрической проверкой."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import LightGlue

from ..logging import get_logger
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
    query_indices: np.ndarray
    candidate_indices: np.ndarray
    matching_scores: np.ndarray
    relative_rotation: np.ndarray | None
    relative_translation: np.ndarray | None
    triangulated_points: np.ndarray
    triangulated_query_indices: np.ndarray
    triangulated_candidate_indices: np.ndarray
    triangulated_scores: np.ndarray

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

    @property
    def triangulated_count(self) -> int:
        """Количество точек, восстановленных в 3D."""

        return int(self.triangulated_points.shape[0]) if self.triangulated_points.size else 0


class LightGlueMatcher:
    """Высокоуровневый интерфейс для LightGlue с верификацией геометрии."""

    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._matcher = LightGlue(features="superpoint").to(device).eval()
        self._log = get_logger("georag.matcher")
        self._log.info("event=lightglue_init device=%s", device)

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
                query_indices=np.empty(0, dtype=np.int32),
                candidate_indices=np.empty(0, dtype=np.int32),
                matching_scores=np.empty(0, dtype=np.float32),
                relative_rotation=None,
                relative_translation=None,
                triangulated_points=np.empty((0, 3), dtype=np.float32),
                triangulated_query_indices=np.empty(0, dtype=np.int32),
                triangulated_candidate_indices=np.empty(0, dtype=np.int32),
                triangulated_scores=np.empty(0, dtype=np.float32),
            )

        inputs = {
            "image0": query.to_lightglue_inputs(self._device),
            "image1": candidate.to_lightglue_inputs(self._device),
        }
        self._log.debug(
            "event=lightglue_match_start query_points=%s candidate_points=%s",
            query.keypoints_count,
            candidate.keypoints_count,
        )
        with torch.inference_mode():
            matches = self._matcher(inputs)

        matches0 = matches["matches0"][0]
        scores0 = matches["matching_scores0"][0]
        valid = matches0 > -1
        matched = int(valid.sum().item())
        self._log.debug("event=lightglue_match_result matches=%s", matched)
        if matched == 0:
            return MatchScore(
                matches=0,
                total_query=query.keypoints_count,
                total_candidate=candidate.keypoints_count,
                mean_score=0.0,
                inliers=0,
                inlier_ratio=0.0,
                geometric_score=0.0,
                query_indices=np.empty(0, dtype=np.int32),
                candidate_indices=np.empty(0, dtype=np.int32),
                matching_scores=np.empty(0, dtype=np.float32),
                relative_rotation=None,
                relative_translation=None,
                triangulated_points=np.empty((0, 3), dtype=np.float32),
                triangulated_query_indices=np.empty(0, dtype=np.int32),
                triangulated_candidate_indices=np.empty(0, dtype=np.int32),
                triangulated_scores=np.empty(0, dtype=np.float32),
            )

        matched_scores = scores0[valid].detach().cpu().numpy().astype(np.float32)
        mean_score = float(np.clip(np.mean(matched_scores), 0.0, 1.0)) if matched_scores.size else 0.0

        query_indices_tensor = torch.arange(matches0.shape[0], device=matches0.device)[
            valid
        ]
        query_indices = query_indices_tensor.detach().cpu().numpy().astype(np.int32)
        candidate_indices = matches0[valid].detach().cpu().numpy().astype(np.int32)
        inliers = 0
        inlier_ratio = 0.0
        geometric_score = 0.0
        rotation: np.ndarray | None = None
        translation: np.ndarray | None = None
        triangulated_points = np.empty((0, 3), dtype=np.float32)
        triangulated_query_idx = np.empty(0, dtype=np.int32)
        triangulated_candidate_idx = np.empty(0, dtype=np.int32)
        triangulated_scores = np.empty(0, dtype=np.float32)

        if matched >= 8:
            try:
                query_norm = query.normalized_keypoints(query_indices)
                candidate_norm = candidate.normalized_keypoints(candidate_indices)
                essential, mask_e = cv2.findEssentialMat(
                    query_norm.astype(np.float64),
                    candidate_norm.astype(np.float64),
                    focal=1.0,
                    pp=(0.0, 0.0),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1e-3,
                )
            except cv2.error as exc:  # pragma: no cover - OpenCV failures
                self._log.debug("event=essential_failed error=%s", exc)
                essential = None
                mask_e = None

            if essential is not None and mask_e is not None:
                mask_e = mask_e.reshape(-1).astype(bool)
                try:
                    retval, R, t, mask_pose = cv2.recoverPose(
                        essential,
                        query_norm.astype(np.float64),
                        candidate_norm.astype(np.float64),
                        mask=mask_e.astype(np.uint8),
                    )
                except cv2.error as exc:  # pragma: no cover - OpenCV failures
                    self._log.debug("event=recover_pose_failed error=%s", exc)
                else:
                    if retval > 0 and mask_pose is not None:
                        pose_mask = mask_pose.reshape(-1).astype(bool)
                        combined = mask_e & pose_mask
                        inliers = int(np.count_nonzero(combined))
                        if inliers:
                            inlier_ratio = float(inliers) / float(matched)
                            rotation = R.astype(np.float32)
                            translation = t.reshape(3).astype(np.float32)
                            selected_q = query_norm[combined]
                            selected_c = candidate_norm[combined]
                            proj_q = np.hstack([
                                np.eye(3, dtype=np.float64),
                                np.zeros((3, 1), dtype=np.float64),
                            ])
                            proj_c = np.hstack([
                                R.astype(np.float64),
                                t.astype(np.float64),
                            ])
                            try:
                                points_h = cv2.triangulatePoints(
                                    proj_q, proj_c, selected_q.T, selected_c.T
                                )
                                points = cv2.convertPointsFromHomogeneous(points_h.T).reshape(-1, 3)
                            except cv2.error as exc:  # pragma: no cover - OpenCV failures
                                self._log.debug("event=triangulation_failed error=%s", exc)
                            else:
                                depths_q = points[:, 2]
                                points_in_c = (R @ points.T + t).T
                                depths_c = points_in_c[:, 2]
                                positive = (depths_q > 1e-4) & (depths_c > 1e-4)
                                if np.any(positive):
                                    point_indices = np.nonzero(combined)[0][positive]
                                    triangulated_points = points[positive].astype(np.float32)
                                    triangulated_query_idx = query_indices[point_indices]
                                    triangulated_candidate_idx = candidate_indices[point_indices]
                                    triangulated_scores = matched_scores[point_indices]
                                    geometric_score = float(
                                        np.clip(
                                            float(
                                                np.mean(
                                                    triangulated_scores
                                                    if triangulated_scores.size
                                                    else 0.0
                                                )
                                            ),
                                            0.0,
                                            1.0,
                                        )
                                    ) * inlier_ratio
                                    self._log.debug(
                                        "event=lightglue_geometry inliers=%s ratio=%.4f score=%.4f tri_points=%s",
                                        inliers,
                                        inlier_ratio,
                                        geometric_score,
                                        triangulated_points.shape[0],
                                    )

        return MatchScore(
            matches=matched,
            total_query=query.keypoints_count,
            total_candidate=candidate.keypoints_count,
            mean_score=mean_score,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            geometric_score=geometric_score,
            query_indices=query_indices,
            candidate_indices=candidate_indices,
            matching_scores=matched_scores,
            relative_rotation=rotation,
            relative_translation=translation,
            triangulated_points=triangulated_points,
            triangulated_query_indices=triangulated_query_idx,
            triangulated_candidate_indices=triangulated_candidate_idx,
            triangulated_scores=triangulated_scores,
        )

    async def amatch(self, query: LocalFeatureSet, candidate: LocalFeatureSet) -> MatchScore:
        """Асинхронный подсчёт сопоставлений."""

        return await asyncio.to_thread(self.match, query, candidate)
