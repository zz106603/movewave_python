# 프로젝트 이름

### MoveWave

사용자가 기분을 입력하면 AI 모델을 활용하여 감정 분석 후, 해당 감정에 어울리는 유튜브 플레이리스트를 추천해주는 서비스


## 주요 기능

- 감정 텍스트 분석
  - 기쁨(행복한)
  - 고마운
  - 설레는(기대하는)
  - 사랑하는
  - 즐거운(신나는)
  - 일상적인
  - 생각이 많은
  - 슬픔(우울한)
  - 힘듦(지침)
  - 짜증남
  - 걱정스러운(불안한)
- 감정에 따른 음악 추천 (유튜브 연동 - 썸네일 및 링크 이동)

## 화면

> ![Image](https://github.com/user-attachments/assets/ea9438e1-a1e9-4baa-ba98-435ec1448d79)

## 아키텍처

> ![Image](https://github.com/user-attachments/assets/de0814cd-770a-4054-a702-3e188c9d6c84)

## 사용 기술 스택

- **Frontend**: React
- **Backend**: Spring Boot
- **Database**: MySQL
- **AI/ML**: Python (Hugging Face 기반 감정 분석 모델)
- **Infra**: Docker, Redis
- **Etc**: GitHub 