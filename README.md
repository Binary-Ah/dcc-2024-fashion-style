# 2024 데이터 크리에이터 캠프 (K-ICT 빅데이터센터) — 패션 이미지 스타일 분류 & 선호 스타일 추천

2024.08 ~ 2024.10 · 대학부 · 4인 팀 · 패션 스타일 이미지 + 응답자 선호 설문 JSON 데이터셋

## Missions
| 미션 | 내용 | 방법 |
|---|---|---|
| 1-1 | 파일명 `{W/T}_{이미지ID}_{시대}_{스타일}_{성별}.jpg` 파싱 → 성별×스타일 통계표 | pandas |
| 1-2 | "성별 & 스타일" 클래스 분류, validation 정확도 | **DeepLabV3(ResNet-101) 로 인물 영역 segmentation 후 ResNet-18 학습**, seed 고정, 데이터셋 mean/std 직접 계산, good/bad 예측 시각화 |
| 2-1 / 2-2 | 응답자 설문 JSON 분석, 유효 라벨 필터링 | 선호/비선호 스타일 피처 생성 (`scripts/style_preference_features.py`) |
| 3-1 / 3-2 | 사용자별 선호 스타일 추천 및 평가 | 사용자 스타일 벡터(r_style1~5) 코사인 유사도 기반 **user-based CF** (`user_based_cf_recommendation.ipynb`), train/valid 무결성 확인 후 정확도 산출 |

## Layout
```
notebooks/
  mission_1_2_resnet18_pipeline.ipynb     # Mission 1~3 전체 파이프라인 (출력 비움)
  user_based_cf_recommendation.ipynb      # 사용자 기반 협업 필터링 추천
scripts/style_preference_features.py      # 설문 JSON → 선호/비선호 스타일 데이터프레임
docs/notebook_export_with_outputs.pdf     # 출력 포함 노트북 PDF (결과 확인용)
requirements.txt
```
데이터셋(이미지 ~6.7만 장·라벨 JSON)과 `best_model.pth`는 캠프 제공 자료라 포함하지 않았다.

## My role
팀 4인 중 이미지 전처리(segmentation)·ResNet-18 분류 파이프라인과 사용자 기반 추천 코드 작성, Git-flow 로 브랜치 운영.
(캠프 2025 — 위성/산업 영상 굴뚝 탐지는 별도 작업, 코드 미확보)
