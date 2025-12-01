# A/B Test Simulator for AI App

AI 앱의 멤버십 전환율 A/B 테스트를 위한 Monte Carlo 시뮬레이션 도구입니다.

## 📋 프로젝트 개요

이 프로젝트는 AI 앱에서 멤버십 상품 구성 변경이 매출에 미치는 영향을 분석하기 위한 A/B 테스트 시뮬레이터입니다. Monte Carlo 시뮬레이션을 통해 통계적으로 유의미한 결과를 도출합니다.

### 주요 기능

- 📊 사용자 행동 데이터 시뮬레이션 (가입, 설문 응답, 결제)
- 🔬 A/B 테스트 그룹 자동 배정
- 💰 멤버십 전환율 및 매출 분석
- 📈 Monte Carlo 시뮬레이션 (1,000회 반복)
- 📁 CSV 형태의 테스트 데이터 생성
- ✅ 데이터 검증 스크립트 포함

## 🎯 A/B 테스트 설계

### 대조군 (Type 20)
- 일주일 멤버십 비중: 60%
- 한달 멤버십 비중: 40%
- 전환율: 30%
- AI 추천권 추가 판매 (일주일 멤버십 구매자의 70%)

### 실험군 (Type 21)
- 일주일 멤버십 비중: 20%
- 한달 멤버십 비중: 80%
- 전환율: 20%
- AI 추천권 없음

## 📊 측정 지표

- **CVR (Conversion Rate)**: 가입 유저 중 결제 유저 비율
- **ARPU (Average Revenue Per User)**: 유저당 평균 매출
- **ARPPU (Average Revenue Per Paying User)**: 결제 유저당 평균 매출
- **총 매출 (Total Revenue)**: 그룹별 총 매출
- **총 주문 (Total Orders)**: 그룹별 총 주문 건수

## 🚀 시작하기

### 필요 라이브러리
```bash
pip install pandas numpy matplotlib
```

### 실행 방법
```bash
# 시뮬레이션 실행
python ab_test_simulation_monte_carlo.py

# 데이터 검증
python check.py
```

### 출력 데이터

스크립트 실행 시 다음 파일들이 생성됩니다:
```
gdrive/My Drive/DataScience/ABT/dataRaw/
├── users.csv                          # 사용자 정보
├── ab_tests.csv                       # A/B 테스트 배정
├── answers.csv                        # 설문 응답 데이터
├── payments.csv                       # 결제 데이터
├── monte_carlo_results.csv            # 시뮬레이션 결과
└── monte_carlo_visualization.png      # 시각화 결과
```

## 📈 시뮬레이션 파라미터

### 사용자 설정
- 총 사용자 수: 900명 (6개월간)
- 남성 비율: 66.7% (평균 출생년도: 1995년)
- 여성 비율: 33.3% (평균 출생년도: 1998년)

### 가격 정책
- AI 추천권: 5,000원
- 일주일 멤버십: 20,000원
- 한달 멤버십: 50,000원

### 재구매율
- 월별 재구매율: 50%

## 📊 시뮬레이션 결과 예시

### 실제 시뮬레이션 데이터 (단일 실행)
```
[Type 20 - 대조군]
  총 유저: 451명
  결제 유저: 142명
  CVR: 31.49%
  총 매출: 12,955,000원
  ARPU: 28,725원
  ARPPU: 91,232원
  총 주문: 353건
  
  주문 타입별:
    - AI 추천권: 121건
    - 일주일 멤버십: 155건
    - 한달 멤버십: 77건

[Type 21 - 실험군]
  총 유저: 449명
  결제 유저: 93명
  CVR: 20.71%
  총 매출: 10,190,000원
  ARPU: 22,695원
  ARPPU: 109,570원
  총 주문: 132건
  
  주문 타입별:
    - 일주일 멤버십: 43건
    - 한달 멤버십: 89건

[주요 차이]
  매출 차이: 2,765,000원 (Type 20 > Type 21)
  CVR 차이: 10.78%p (Type 20 > Type 21)
  ARPPU 차이: 18,338원 (Type 21 > Type 20)
```

### Monte Carlo 시뮬레이션 평균 결과 (1,000회)
```
[평균 매출]
Type 20 (대조군): 14,500,000원
Type 21 (실험군): 12,800,000원
매출 차이: 1,700,000원

[평균 전환율]
Type 20 (대조군): 30.2%
Type 21 (실험군): 20.1%

[평균 ARPPU]
Type 20 (대조군): 107,500원
Type 21 (실험군): 142,000원

[95% 신뢰구간 - 매출 차이]
95% CI: [1,200,000원, 2,200,000원]
Type 20이 더 높을 확률: 99.8%
```

## 🔍 주요 발견

1. **전환율 트레이드오프**: Type 21은 전환율이 낮지만 ARPPU가 높음
2. **총 매출**: Type 20이 통계적으로 유의미하게 높은 매출 생성
3. **재구매 패턴**: 재구매율 50% 가정 시 6개월간의 누적 효과 분석
4. **상품 구성**: Type 20의 AI 추천권이 추가 매출 창출에 기여
5. **멤버십 선호도**: Type 21은 한달 멤버십 비중이 높아 ARPPU 상승

## 📁 데이터 스키마

### users.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| timestamp | datetime | 가입 시간 |
| id | string | 사용자 ID |
| gender | string | 성별 (M/F) |
| birth | int | 출생년도 |

### ab_tests.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| timestamp | datetime | 배정 시간 |
| user_id | string | 사용자 ID |
| type | int | A/B 테스트 타입 (20/21) |

### answers.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| created | datetime | 응답 시간 |
| answer_id | int | 질문 번호 (1-5) |
| user_id | string | 사용자 ID |
| answer | int | 응답 값 (1/2) |

### payments.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| created | datetime | 결제 시간 |
| user_id | string | 사용자 ID |
| order_name | string | 상품명 (ai 추천권/일주일 멤버십/한달 멤버십) |
| amount | int | 결제 금액 |
| cancel | int | 취소 여부 (0/1) |

## 🛠️ 커스터마이징

시뮬레이션 파라미터는 스크립트 상단에서 수정 가능합니다:
```python
# 시뮬레이션 횟수
N_SIMULATIONS = 1000

# 전환율 설정
CONVERSION_RATE_TYPE20 = 0.30
CONVERSION_RATE_TYPE21 = 0.20

# 멤버십 분포
MEMBERSHIP_DIST_TYPE21 = {
    'weekly': 0.20,
    'monthly': 0.80
}

# 재구매율
REPURCHASE_RATE = 0.50
```

## ✅ 데이터 검증

`check.py` 스크립트를 사용하여 생성된 CSV 데이터의 무결성을 확인할 수 있습니다:
```bash
python check.py
```

검증 항목:
- Type별 총 유저 수 및 결제 유저 수
- CVR, ARPU, ARPPU 계산 검증
- 주문 타입별 분포 확인
- 총 매출 및 주문 건수 확인

## 👤 작성자

Marvin

## 🙏 기여

이슈나 풀 리퀘스트는 언제나 환영합니다!

---

**Note**: 이 시뮬레이터는 교육 및 분석 목적으로 제작되었습니다.