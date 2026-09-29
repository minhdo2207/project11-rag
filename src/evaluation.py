"""P5 — Cac chi so danh gia.

STUB: chi dinh nghia chu ky ham + mo ta viec can lam. P5 se hien thuc.

Can co (toi thieu):
  - accuracy(y_true, y_pred)
  - balanced_accuracy(y_true, y_pred)   # quan trong vi du lieu lech (lanh >> doc)
  - precision / recall / f1 cho lop "malicious"
  - false_negative_rate               # bo sot ma doc - nguy hiem nhat trong bao mat
  - confusion_matrix
Mo rong (theo huong nghien cuu):
  - context-sufficiency: tach loi retrieval khoi loi generation
  - evidence-localization: model co chi dung dong code doc khong (chong hallucination)
"""

from __future__ import annotations


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    raise NotImplementedError("P5 hien thuc")


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    raise NotImplementedError("P5 hien thuc")
