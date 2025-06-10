# 가짜 뉴스 판별 AI 모델 개발

## 목차
  - [프로젝트 기본 정보](#프로젝트-기본-정보)
  - [프로젝트 개요](#프로젝트-개요)
  - [프로젝트 설명](#프로젝트-설명)
  - [분석 결과](#분석-결과)
  - [기대 효과](#기대-효과)

## 프로젝트 기본 정보
- 프로젝트 이름: 가짜 뉴스 판별 AI 모델 개발
- 프로젝트 기간: 2025.04
- 프로젝트 참여인원: 4명
- 사용 언어: Python

## 프로젝트 개요
- 미디어의 발달로 인해 거짓 뉴스가 빠르게 확산되어 잘못된 정보 전달
- 가짜 뉴스 판별 모델을 통해 낮은 품질의 기사로  인한 시간 비용 낭비 사실 왜곡 등 해소에 도움을 주어 국내 저널리즘 신뢰성 향상에 기여하도록 함.
- <img width="398" alt="낚시성 ㅐ요" src="https://github.com/user-attachments/assets/6bcc580c-7488-4037-bc57-1a5668fc1f45" />

## 프로젝트 설명
- 뉴스 기사 본문을 기반으로 낚시성 여부를 이진 분류하는 모델 개발

## 분석 결과
### OKT 형태소 분석기를 통한 토큰화 및 불용어 제거
<img width="458" alt="데이터 전처리1" src="https://github.com/user-attachments/assets/f4279731-125c-42f7-925f-267f2a75f9c6" />
<img width="386" alt="데이터 전처리2" src="https://github.com/user-attachments/assets/e2062c3e-042a-4ba6-a89c-e02722949242" />
<img width="330" alt="낚시성_학습결과" src="https://github.com/user-attachments/assets/d5c2be57-6fe5-414c-9929-7a4cae0c8c4c" />
<img width="567" alt="낚시성 모델 테스트_진짜" src="https://github.com/user-attachments/assets/d374f07a-38ff-48a8-8b71-955d1a071295" />
<img width="557" alt="낚시성 모델 테스트_가짜" src="https://github.com/user-attachments/assets/d6593d14-bba4-4c07-b4d9-3322a137c25b" />

## 기대 효과
- 가짜뉴스 판별 모델을 통해 낮은 품질의 기사로  인한 시간 비용 낭비 사실 왜곡 등 해소
- 국내 저널리즘 신뢰성 향상에 기여

## Lesson & Learned
- 이틀이라는 짧은 시간으로 인해 모델 구조의 다양성 부족한 것과 모델 성능 향상에 대한 아쉬움이 있음.
- 불용어 처리 기준을 선정하는 데에 어려움이 있었음.
- 자연어 처리 과정과 딥러닝 모델 설계에 대한 학습의 필요성을 느낌
