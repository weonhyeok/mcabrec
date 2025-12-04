# A/B Test Monte Carlo Simulator for AI App

AI 앱의 **멤버십 상품 구성 변경**이 매출과 전환율에 미치는 영향을 분석하기 위한  
**가상의 사용자 데이터 생성 + A/B 테스트 + 회귀 분석** 프로젝트입니다.

현재 스크립트(`mcmc.py`)는 실행할 때마다 하나의 synthetic 데이터셋을 생성하고,  
그 위에서 **CVR/ARPU/ARPPU 계산 및 회귀 분석(LPM 포함)**을 자동으로 수행합니다.

---

## 📋 프로젝트 개요

이 프로젝트는 아래 네 가지 기능을 제공합니다.

1. 👥 사용자 데이터 생성 (1,700명, 2025년 5~10월)
2. 🧪 A/B 테스트 설정 (6월 가입자 600명 → Type 20/21 무작위 배정)
3. 💳 결제/설문 시뮬레이션 (멤버십/AI추천권/설문 5문항)
4. 📈 주요 지표 및 회귀 분석 출력

---

## 🎯 A/B 테스트 설계

A/B 테스트는 **2025년 6월 가입자 600명**만을 대상으로 합니다.

---

### **Type 20 (대조군 - Control)**

#### 상품 구성
- 일주일 멤버십: 30,000원
- 한달 멤버십: 100,000원
- AI 추천권: 5,000원  
- 일주일 멤버십 구매자의 **75%**가 AI 추천권 추가 구매

#### 행동 파라미터
- 전환율(CVR): **30%**
- 멤버십 구성 비율  
  - weekly: 70%  
  - monthly: 30%
- 재구매율: **50%**

#### 전략
저가 상품 + AI 추천권 번들로 **CVR 최대화**.

---

### **Type 21 (실험군 - Treatment)**

#### 상품 구성
- 일주일 멤버십: 30,000원
- 한달 멤버십: 100,000원
- **AI 추천권 없음**

#### 행동 파라미터
- 전환율(CVR): **20%**
- 멤버십 구성 비율  
  - weekly: 25%  
  - monthly: 75%
- 재구매율: **40%**

#### 전략
프리미엄 구조로 **ARPPU 극대화**.

---

## 📊 측정 지표

| 지표 | 설명 | 계산식 |
|------|------|--------|
| **CVR** | 전환율 | 결제 유저 / 전체 유저 |
| **ARPU** | 평균 매출(전체) | 총 매출 / 전체 유저 |
| **ARPPU** | 평균 매출(결제 유저) | 총 매출 / 결제 유저 |
| **Total Revenue** | 총 매출 | 모든 결제 금액 합 |

---

## 🧮 시뮬레이션 핵심 파라미터

```python
# 전환율
CONVERSION_RATE_TYPE20 = 0.30  
CONVERSION_RATE_TYPE21 = 0.20  

# 멤버십 구성
MEMBERSHIP_DIST_TYPE20 = {'weekly': 0.7, 'monthly': 0.3}
MEMBERSHIP_DIST_TYPE21 = {'weekly': 0.25, 'monthly': 0.75}

# AI 추천권 (Type 20 전용)
AI_TICKET_RATE = 0.75

# 재구매율
REPURCHASE_RATE_TYPE20 = 0.50
REPURCHASE_RATE_TYPE21 = 0.40
```

## 👥 사용자 생성 규칙

- **전체 사용자:** 1,700명  
- **월별 구성**
  - 5월: 300명  
  - 6월: 600명 (A/B 대상)  
  - 7~10월: 각 200명  
- **성별 비율**
  - 남성: 66.7%  
  - 여성: 33.3%  
- **출생년도 분포**
  - 남성 평균: 1995년  
  - 여성 평균: 1998년  

---

## 📁 출력되는 데이터

`mcmc.py` 실행 시 생성되는 파일 구조:


```
dataSave/
├── users.csv
├── ab_tests.csv
├── answers.csv
└── payments.csv
```

---

## 🧪 실행 방법

### 1) 라이브러리 설치
```bash
pip install pandas numpy matplotlib scipy statsmodels
```
2) 실행
```bash
python mcmc.py
```

📈 실행 결과 예시 (2025-12-04 기준)
1️⃣ CVR EDA 결과

```bash
      conversions  non_conversions  total       cvr
type
20.0         89.0            208.0    297  0.299663
21.0         65.0            238.0    303  0.214521
```

Type 20 CVR ≈ 30%
Type 21 CVR ≈ 21%

2️⃣ LPM (선형확률모형)

`converted ~ C(type)`

```scss
Intercept                 0.2997
C(type)[T.21.0]          -0.0851   (p = 0.017)
```

Type 21은 Type 20 대비 CVR이 약 8.5%p 감소

통계적으로 유의 (p < 0.05)

3️⃣ ARPU 회귀 (전체 유저 기준)

`arpu ~ C(type)`

```scss
Intercept              32,490원
C(type)[T.21.0]       -3,185원   (p = 0.566)
```

ARPU 차이는 유의하지 않음

4️⃣ ARPPU 회귀 (결제 유저 기준)

`arppu ~ C(type)`

```scss
Intercept              108,400원
C(type)[T.21.0]        28,190원   (p = 0.041)
```

Type 21의 결제 고객이 1인당 약 2.8만 원 더 지출

통계적으로 유의 (p < 0.05)

5️⃣ Before/After 설문 응답률
```nginx
before_response_rate   0.83~0.99
after_response_rate    0.88~0.98
```

전반적으로 후기 그룹(8~10월)의 응답률이 더 높음.

🎓 분석 방법론 요약
Frequentist 접근

전환율: LPM

ARPU/ARPPU: OLS 회귀

효과 해석은 **계수(절대값, 원화 단위)**로 이해

Monte Carlo 관점

현재는 1회 실행 구조지만,
반복문으로 확장하여 1,000회 반복 시뮬레이션도 쉽게 구현 가능.

🛠️ 커스터마이징 포인트

전환율, 재구매율

멤버십 가격/비율

AI 추천권 구매율

출생년도/성별 분포

설문 응답률

📝 라이선스

교육 및 내부 분석 목적.
실제 비즈니스 적용 시 추가 검증 필요.

📅 마지막 업데이트: 2025-12-04

버전: 2.1.0