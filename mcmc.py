import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from scipy import stats
import matplotlib.pyplot as plt

# 시드 설정 (재현성을 위해)
np.random.seed(42)
random.seed(42)

# 시뮬레이션 파라미터
N_SIMULATIONS = 1000  # Monte Carlo 시뮬레이션 횟수

# 월별 설정
MONTHS = ['2025-05', '2025-06', '2025-07', '2025-08', '2025-09', '2025-10']
EARLY_MONTHS = ['2025-05', '2025-06', '2025-07']
LATE_MONTHS = ['2025-08', '2025-09', '2025-10']

# 유저 설정
TOTAL_USERS = 900
MALE_RATIO = 600 / 900
MALE_BIRTH_MEAN = 1995
MALE_BIRTH_STD = 3
FEMALE_BIRTH_MEAN = 1998
FEMALE_BIRTH_STD = 3

# 응답률 설정
ANSWER_RATES_EARLY = [0.99, 0.98, 0.95, 0.85, 0.83]
ANSWER_RATES_LATE = [0.99, 0.98, 0.95, 0.90, 0.88]

# 전환율 설정
CONVERSION_RATE_TYPE20 = 0.3
CONVERSION_RATE_TYPE21 = 0.2

# 멤버십 분포 (type별)
MEMBERSHIP_DIST_TYPE20 = {
    'weekly': 0.7,
    'monthly': 0.3
}

MEMBERSHIP_DIST_TYPE21 = {
    'weekly': 0.25,
    'monthly': 0.75
}

# AI 추천권 구매율 (일주일 멤버십 구매자 중)
AI_TICKET_RATE = 0.75

# 가격 설정
PRICES = {
    'ai 추천권': 5000,
    '일주일 멤버십': 30000,
    '한달 멤버십': 100000
}

# 재구매율
REPURCHASE_RATE_TYPE20 = 0.5
REPURCHASE_RATE_TYPE21 = 0.4


def generate_users(month, n_users_per_month):
    """월별 신규 유저 생성"""
    users = []

    start_date = datetime.strptime(f"{month}-01", "%Y-%m-%d")
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    for i in range(n_users_per_month):
        # 랜덤 timestamp 생성
        random_day = random.randint(0, (end_date - start_date).days)
        timestamp = start_date + timedelta(days=random_day,
                                          hours=random.randint(0, 23),
                                          minutes=random.randint(0, 59))

        # 성별 결정
        gender = 'M' if random.random() < MALE_RATIO else 'F'

        # 출생연도 생성
        if gender == 'M':
            birth = int(np.random.normal(MALE_BIRTH_MEAN, MALE_BIRTH_STD))
        else:
            birth = int(np.random.normal(FEMALE_BIRTH_MEAN, FEMALE_BIRTH_STD))

        users.append({
            'timestamp': timestamp,
            'id': f"{month}_user_{i+1}",
            'gender': gender,
            'birth': birth,
            'signup_month': month
        })

    return users


def generate_ab_tests(users_df):
    """AB 테스트 배정"""
    ab_tests = []

    for _, user in users_df.iterrows():
        ab_type = 20 if random.random() < 0.5 else 21

        ab_tests.append({
            'timestamp': user['timestamp'],
            'user_id': user['id'],
            'type': ab_type
        })

    return ab_tests


def generate_answers(users_df):
    """답변 생성"""
    answers = []

    for _, user in users_df.iterrows():
        signup_month = user['signup_month']

        # 월별 응답률 선택
        if signup_month in EARLY_MONTHS:
            answer_rates = ANSWER_RATES_EARLY
        else:
            answer_rates = ANSWER_RATES_LATE

        # 각 질문에 대해 답변 생성
        for answer_id in range(1, 6):
            rate = answer_rates[answer_id - 1]

            if random.random() < rate:
                # 답변 시간 (가입 후 몇 시간 내)
                created = user['timestamp'] + timedelta(hours=random.randint(0, 24))

                answers.append({
                    'created': created,
                    'answer_id': answer_id,
                    'user_id': user['id'],
                    'answer': random.choice([1, 2])
                })

    return answers


def generate_payments(users_df, ab_tests_df, month_index):
    """결제 데이터 생성 (재구매 포함)"""
    payments = []
    user_purchase_history = {}
    
    merged_df = users_df.merge(ab_tests_df, left_on='id', right_on='user_id')
    
    for _, user in merged_df.iterrows():
        user_id = user['id']
        ab_type = user['type']
        signup_month_idx = MONTHS.index(user['signup_month'])
        
        # Type별 재구매율 설정
        repurchase_rate = REPURCHASE_RATE_TYPE20 if ab_type == 20 else REPURCHASE_RATE_TYPE21
        
        for current_month_idx in range(signup_month_idx, len(MONTHS)):
            current_month = MONTHS[current_month_idx]
            
            if current_month_idx == signup_month_idx:
                conversion_rate = CONVERSION_RATE_TYPE20 if ab_type == 20 else CONVERSION_RATE_TYPE21
                will_purchase = random.random() < conversion_rate
            else:
                if user_id in user_purchase_history and (current_month_idx - 1) in user_purchase_history[user_id]:
                    will_purchase = random.random() < repurchase_rate
                else:
                    will_purchase = False

            if will_purchase:
                # 구매 이력 기록
                if user_id not in user_purchase_history:
                    user_purchase_history[user_id] = []
                user_purchase_history[user_id].append(current_month_idx)

                # 멤버십 타입 결정
                if ab_type == 20:
                    membership_type = 'weekly' if random.random() < MEMBERSHIP_DIST_TYPE20['weekly'] else 'monthly'
                else:
                    membership_type = 'weekly' if random.random() < MEMBERSHIP_DIST_TYPE21['weekly'] else 'monthly'

                order_name = '일주일 멤버십' if membership_type == 'weekly' else '한달 멤버십'

                # 결제 시간 (해당 월 내 랜덤)
                start_date = datetime.strptime(f"{current_month}-01", "%Y-%m-%d")
                end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                random_day = random.randint(0, (end_date - start_date).days)
                created = start_date + timedelta(days=random_day,
                                                hours=random.randint(0, 23),
                                                minutes=random.randint(0, 59))

                payments.append({
                    'created': created,
                    'user_id': user_id,
                    'order_name': order_name,
                    'amount': PRICES[order_name],
                    'cancel': 0
                })

                # type=20이고 일주일 멤버십이면 AI 추천권 구매 가능
                if ab_type == 20 and membership_type == 'weekly':
                    if random.random() < AI_TICKET_RATE:
                        payments.append({
                            'created': created + timedelta(seconds=random.randint(1, 300)),
                            'user_id': user_id,
                            'order_name': 'ai 추천권',
                            'amount': PRICES['ai 추천권'],
                            'cancel': 0
                        })

    return payments


def run_single_simulation():
    """단일 시뮬레이션 실행"""
    # 월별 유저 수 배분
    users_per_month = TOTAL_USERS // len(MONTHS)

    all_users = []
    for month in MONTHS:
        all_users.extend(generate_users(month, users_per_month))

    users_df = pd.DataFrame(all_users)
    ab_tests_df = pd.DataFrame(generate_ab_tests(users_df))
    answers_df = pd.DataFrame(generate_answers(users_df))
    payments_df = pd.DataFrame(generate_payments(users_df, ab_tests_df, 0))

    # 결과 계산
    merged_payments = payments_df.merge(ab_tests_df, on='user_id')

    results = {
        'total_revenue_type20': merged_payments[merged_payments['type'] == 20]['amount'].sum(),
        'total_revenue_type21': merged_payments[merged_payments['type'] == 21]['amount'].sum(),
        'n_purchasers_type20': merged_payments[merged_payments['type'] == 20]['user_id'].nunique(),
        'n_purchasers_type21': merged_payments[merged_payments['type'] == 21]['user_id'].nunique(),
        'n_users_type20': len(ab_tests_df[ab_tests_df['type'] == 20]),
        'n_users_type21': len(ab_tests_df[ab_tests_df['type'] == 21]),
    }

    return results, users_df, ab_tests_df, answers_df, payments_df


def run_monte_carlo_simulation(n_simulations=N_SIMULATIONS):
    """Monte Carlo 시뮬레이션 실행"""
    simulation_results = []

    print(f"Running {n_simulations} Monte Carlo simulations...")

    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{n_simulations}")

        results, _, _, _, _ = run_single_simulation()
        simulation_results.append(results)

    return pd.DataFrame(simulation_results)


def save_data_to_gdrive(users_df, ab_tests_df, answers_df, payments_df):
    """데이터를 폴더에 저장"""

    # 폴더 경로 설정
    base_path = 'dataSave/'

    # 디렉토리가 없으면 생성
    os.makedirs(base_path, exist_ok=True)

    print("\n" + "=" * 60)
    print("폴더에 데이터 저장 중...")
    print("=" * 60)

    # users 테이블 저장 (signup_month 컬럼 제거)
    users_save = users_df.drop(columns=['signup_month'])
    users_save.to_csv(f'{base_path}/users.csv', index=False, encoding='utf-8-sig')
    print(f"✓ users.csv 저장 완료 ({len(users_save)} rows)")

    # ab_tests 테이블 저장
    ab_tests_df.to_csv(f'{base_path}/ab_tests.csv', index=False, encoding='utf-8-sig')
    print(f"✓ ab_tests.csv 저장 완료 ({len(ab_tests_df)} rows)")

    # answers 테이블 저장
    answers_df.to_csv(f'{base_path}/answers.csv', index=False, encoding='utf-8-sig')
    print(f"✓ answers.csv 저장 완료 ({len(answers_df)} rows)")

    # payments 테이블 저장
    payments_df.to_csv(f'{base_path}/payments.csv', index=False, encoding='utf-8-sig')
    print(f"✓ payments.csv 저장 완료 ({len(payments_df)} rows)")

    print("\n모든 데이터가 성공적으로 저장되었습니다!")
    print(f"저장 경로: {base_path}")

    return base_path


# ============================================================================
# 메인 실행 코드
# ============================================================================

print("=" * 60)
print("Monte Carlo Simulation 시작")
print("=" * 60)

# 단일 시뮬레이션 예제 (데이터 확인용)
print("\n1. 단일 시뮬레이션 예제 실행 중...")
single_results, users_sample, ab_tests_sample, answers_sample, payments_sample = run_single_simulation()

print("\n[샘플 데이터 확인]")
print(f"\n- Users 테이블: {len(users_sample)} rows")
print(users_sample.head())

print(f"\n- AB Tests 테이블: {len(ab_tests_sample)} rows")
print(ab_tests_sample.head())
print(f"  Type 20: {len(ab_tests_sample[ab_tests_sample['type']==20])} users")
print(f"  Type 21: {len(ab_tests_sample[ab_tests_sample['type']==21])} users")

print(f"\n- Answers 테이블: {len(answers_sample)} rows")
print(answers_sample.head())

print(f"\n- Payments 테이블: {len(payments_sample)} rows")
print(payments_sample.head())
print(f"\n  Order name 분포:")
merged_payments_sample = payments_sample.merge(ab_tests_sample, on='user_id')
print("\nType 20:")
print(merged_payments_sample[merged_payments_sample['type']==20]['order_name'].value_counts())
print("\nType 21:")
print(merged_payments_sample[merged_payments_sample['type']==21]['order_name'].value_counts())

# 폴더에 데이터 저장
saved_path = save_data_to_gdrive(users_sample, ab_tests_sample, answers_sample, payments_sample)

# Monte Carlo 시뮬레이션 실행
print("\n" + "=" * 60)
print("2. Monte Carlo Simulation 실행 중...")
print("=" * 60)
mc_results = run_monte_carlo_simulation(N_SIMULATIONS)

# 결과 분석
print("\n" + "=" * 60)
print("Monte Carlo Simulation 결과")
print("=" * 60)

# 평균 매출
print("\n[평균 매출]")
print(f"Type 20 (대조군): {mc_results['total_revenue_type20'].mean():,.0f}원")
print(f"Type 21 (실험군): {mc_results['total_revenue_type21'].mean():,.0f}원")
print(f"매출 차이: {(mc_results['total_revenue_type20'] - mc_results['total_revenue_type21']).mean():,.0f}원")

# 전환율
mc_results['conversion_rate_type20'] = mc_results['n_purchasers_type20'] / mc_results['n_users_type20']
mc_results['conversion_rate_type21'] = mc_results['n_purchasers_type21'] / mc_results['n_users_type21']

print("\n[평균 전환율]")
print(f"Type 20 (대조군): {mc_results['conversion_rate_type20'].mean():.2%}")
print(f"Type 21 (실험군): {mc_results['conversion_rate_type21'].mean():.2%}")

# ARPPU (Average Revenue Per Paying User)
mc_results['arppu_type20'] = mc_results['total_revenue_type20'] / mc_results['n_purchasers_type20']
mc_results['arppu_type21'] = mc_results['total_revenue_type21'] / mc_results['n_purchasers_type21']

print("\n[평균 ARPPU]")
print(f"Type 20 (대조군): {mc_results['arppu_type20'].mean():,.0f}원")
print(f"Type 21 (실험군): {mc_results['arppu_type21'].mean():,.0f}원")

# ARPU (Average Revenue Per User)
mc_results['arpu_type20'] = mc_results['total_revenue_type20'] / mc_results['n_users_type20']
mc_results['arpu_type21'] = mc_results['total_revenue_type21'] / mc_results['n_users_type21']

print("\n[평균 ARPU]")
print(f"Type 20 (대조군): {mc_results['arpu_type20'].mean():,.0f}원")
print(f"Type 21 (실험군): {mc_results['arpu_type21'].mean():,.0f}원")

# 신뢰구간
print("\n[95% 신뢰구간 - 매출 차이]")
revenue_diff = mc_results['total_revenue_type20'] - mc_results['total_revenue_type21']
ci_lower = np.percentile(revenue_diff, 2.5)
ci_upper = np.percentile(revenue_diff, 97.5)
print(f"95% CI: [{ci_lower:,.0f}원, {ci_upper:,.0f}원]")
print(f"Type 20이 더 높을 확률: {(revenue_diff > 0).mean():.1%}")

# Monte Carlo 결과도 저장
mc_results.to_csv(f'{saved_path}/monte_carlo_results.csv', index=False, encoding='utf-8-sig')
print(f"\nMonte Carlo 결과가 '{saved_path}/monte_carlo_results.csv'에 저장되었습니다.")

# ============================================================================
# Monte Carlo 시각화
# ============================================================================

print("\nMonte Carlo 시각화 생성 중...")

plt.figure(figsize=(15, 10))

# 1. 매출 분포
plt.subplot(2, 3, 1)
plt.hist(mc_results['total_revenue_type20'], bins=50, alpha=0.5, label='Type 20')
plt.hist(mc_results['total_revenue_type21'], bins=50, alpha=0.5, label='Type 21')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.title('Revenue Distribution')
plt.legend()

# 2. 매출 차이 분포
plt.subplot(2, 3, 2)
plt.hist(revenue_diff, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', label='No difference')
plt.xlabel('Revenue Difference (Type 20 - Type 21)')
plt.ylabel('Frequency')
plt.title('Revenue Difference Distribution')
plt.legend()

# 3. 전환율 분포
plt.subplot(2, 3, 3)
plt.hist(mc_results['conversion_rate_type20'], bins=50, alpha=0.5, label='Type 20')
plt.hist(mc_results['conversion_rate_type21'], bins=50, alpha=0.5, label='Type 21')
plt.xlabel('Conversion Rate')
plt.ylabel('Frequency')
plt.title('Conversion Rate Distribution')
plt.legend()

# 4. ARPPU 분포
plt.subplot(2, 3, 4)
plt.hist(mc_results['arppu_type20'], bins=50, alpha=0.5, label='Type 20')
plt.hist(mc_results['arppu_type21'], bins=50, alpha=0.5, label='Type 21')
plt.xlabel('ARPPU')
plt.ylabel('Frequency')
plt.title('ARPPU Distribution')
plt.legend()

# 5. Box plot - Revenue
plt.subplot(2, 3, 5)
plt.boxplot([mc_results['total_revenue_type20'], mc_results['total_revenue_type21']],
            labels=['Type 20', 'Type 21'])
plt.ylabel('Total Revenue')
plt.title('Revenue Comparison')

# 6. Cumulative probability
plt.subplot(2, 3, 6)
sorted_diff = np.sort(revenue_diff)
cumulative = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
plt.plot(sorted_diff, cumulative)
plt.axvline(0, color='red', linestyle='--', label='No difference')
plt.xlabel('Revenue Difference')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Revenue Difference')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = f'{saved_path}/monte_carlo_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Monte Carlo 시각화가 '{viz_path}'에 저장되었습니다.")
plt.close()

# ============================================================================
# 통계적 유의성 검정 (t-test)
# ============================================================================

print("\n" + "=" * 60)
print("통계적 유의성 검정 (t-test)")
print("=" * 60)

# 1. CVR 차이 검정
cvr_diff = mc_results['conversion_rate_type20'] - mc_results['conversion_rate_type21']
t_stat_cvr, p_value_cvr = stats.ttest_1samp(cvr_diff, 0)

print("\n[1. CVR 차이 검정]")
print(f"Type 20 평균 CVR: {mc_results['conversion_rate_type20'].mean():.2%}")
print(f"Type 21 평균 CVR: {mc_results['conversion_rate_type21'].mean():.2%}")
print(f"CVR 차이 (Type 20 - Type 21): {cvr_diff.mean():.2%}")
print(f"t-statistic: {t_stat_cvr:.4f}")
print(f"p-value: {p_value_cvr:.6f}")
print(f"통계적으로 유의함 (p < 0.05): {'YES ✓' if p_value_cvr < 0.05 else 'NO'}")
print(f"Type 20 CVR이 더 높음: {'YES ✓' if cvr_diff.mean() > 0 else 'NO'}")

# 2. Revenue 차이 검정
t_stat_rev, p_value_rev = stats.ttest_1samp(revenue_diff, 0)

print("\n[2. Total Revenue 차이 검정]")
print(f"Type 20 평균 매출: {mc_results['total_revenue_type20'].mean():,.0f}원")
print(f"Type 21 평균 매출: {mc_results['total_revenue_type21'].mean():,.0f}원")
print(f"매출 차이 (Type 20 - Type 21): {revenue_diff.mean():,.0f}원")
print(f"t-statistic: {t_stat_rev:.4f}")
print(f"p-value: {p_value_rev:.6f}")
print(f"통계적으로 유의함 (p < 0.05): {'YES ✓' if p_value_rev < 0.05 else 'NO'}")
print(f"Type 20 매출이 더 높을 확률: {(revenue_diff > 0).mean():.1%}")

# 3. ARPPU 차이 검정
arppu_diff = mc_results['arppu_type21'] - mc_results['arppu_type20']
t_stat_arppu, p_value_arppu = stats.ttest_1samp(arppu_diff, 0)

print("\n[3. ARPPU 차이 검정]")
print(f"Type 20 평균 ARPPU: {mc_results['arppu_type20'].mean():,.0f}원")
print(f"Type 21 평균 ARPPU: {mc_results['arppu_type21'].mean():,.0f}원")
print(f"ARPPU 차이 (Type 21 - Type 20): {arppu_diff.mean():,.0f}원")
print(f"t-statistic: {t_stat_arppu:.4f}")
print(f"p-value: {p_value_arppu:.6f}")
print(f"통계적으로 유의함 (p < 0.05): {'YES ✓' if p_value_arppu < 0.05 else 'NO'}")
print(f"Type 21 ARPPU가 더 높음: {'YES ✓' if arppu_diff.mean() > 0 else 'NO'}")

# 4. ARPU 차이 검정
arpu_diff = mc_results['arpu_type20'] - mc_results['arpu_type21']
t_stat_arpu, p_value_arpu = stats.ttest_1samp(arpu_diff, 0)

print("\n[4. ARPU 차이 검정]")
print(f"Type 20 평균 ARPU: {mc_results['arpu_type20'].mean():,.0f}원")
print(f"Type 21 평균 ARPU: {mc_results['arpu_type21'].mean():,.0f}원")
print(f"ARPU 차이 (Type 20 - Type 21): {arpu_diff.mean():,.0f}원")
print(f"t-statistic: {t_stat_arpu:.4f}")
print(f"p-value: {p_value_arpu:.6f}")
print(f"통계적으로 유의함 (p < 0.05): {'YES ✓' if p_value_arpu < 0.05 else 'NO'}")

# 5. 요약 테이블
print("\n" + "=" * 60)
print("검정 결과 요약")
print("=" * 60)
print(f"{'Metric':<20} {'Type 20 > Type 21':<20} {'p-value':<15} {'Significant':<15}")
print("-" * 70)
print(f"{'CVR':<20} {(cvr_diff.mean() > 0):<20} {p_value_cvr:<15.6f} {(p_value_cvr < 0.05):<15}")
print(f"{'Total Revenue':<20} {(revenue_diff.mean() > 0):<20} {p_value_rev:<15.6f} {(p_value_rev < 0.05):<15}")
print(f"{'ARPU':<20} {(arpu_diff.mean() > 0):<20} {p_value_arpu:<15.6f} {(p_value_arpu < 0.05):<15}")
print(f"{'ARPPU (Type 21>20)':<20} {(arppu_diff.mean() > 0):<20} {p_value_arppu:<15.6f} {(p_value_arppu < 0.05):<15}")

# ============================================================================
# 베이지안 분석
# ============================================================================

print("\n" + "="*70)
print("Bayesian Analysis: Revenue Difference (Monte Carlo Based)")
print("="*70)

# Monte Carlo 시뮬레이션 결과를 베이지안 사후 분포로 사용
revenue_type20 = mc_results['total_revenue_type20'].values
revenue_type21 = mc_results['total_revenue_type21'].values
revenue_diff_samples = revenue_type20 - revenue_type21

print(f"""
Type 20 (Control): n={len(revenue_type20)}, mean={revenue_type20.mean():,.0f} KRW
Type 21 (Treatment): n={len(revenue_type21)}, mean={revenue_type21.mean():,.0f} KRW
Actual difference: {revenue_diff_samples.mean():,.0f} KRW ({(revenue_diff_samples.mean()/revenue_type21.mean()*100):.1f}%)
""")

# 베이지안 확률 계산
prob_type20_higher = (revenue_diff_samples > 0).mean()

# 실질적으로 중요한 차이 확률
threshold_5pct = 0.05 * revenue_type21.mean()
prob_meaningful_increase_5pct = (revenue_diff_samples > threshold_5pct).mean()

threshold_10pct = 0.10 * revenue_type21.mean()
prob_meaningful_increase_10pct = (revenue_diff_samples > threshold_10pct).mean()

print(f"""
{'='*70}
Bayesian Posterior Analysis (Revenue):
{'='*70}

Probability that Type 20 has HIGHER revenue: {prob_type20_higher:.1%}

Revenue Difference (Type 20 - Type 21):
  Mean: {revenue_diff_samples.mean():,.0f} KRW
  Median: {np.median(revenue_diff_samples):,.0f} KRW
  95% Credible Interval: [{np.percentile(revenue_diff_samples, 2.5):,.0f} KRW,
                          {np.percentile(revenue_diff_samples, 97.5):,.0f} KRW]

Revenue Difference (% change relative to Type 21):
  Mean: {revenue_diff_samples.mean()/revenue_type21.mean()*100:.1f}%
  95% Credible Interval: [{np.percentile(revenue_diff_samples, 2.5)/revenue_type21.mean()*100:.1f}%,
                          {np.percentile(revenue_diff_samples, 97.5)/revenue_type21.mean()*100:.1f}%]

Scenario Probabilities:
  Type 20 revenue > Type 21 revenue + 5%:  {prob_meaningful_increase_5pct:.1%}
  Type 20 revenue > Type 21 revenue + 10%: {prob_meaningful_increase_10pct:.1%}
  Type 20 revenue > Type 21 revenue:       {prob_type20_higher:.1%}

{'='*70}
""")

# ============================================================================
# CVR에 대한 베이지안 분석
# ============================================================================

print("\n" + "="*70)
print("Bayesian Analysis: Conversion Rate Difference")
print("="*70)

cvr_type20 = mc_results['conversion_rate_type20'].values
cvr_type21 = mc_results['conversion_rate_type21'].values
cvr_diff_samples = cvr_type20 - cvr_type21

prob_cvr_type20_higher = (cvr_diff_samples > 0).mean()

threshold_cvr_5pp = 0.05
prob_cvr_meaningful = (cvr_diff_samples > threshold_cvr_5pp).mean()

print(f"""
Type 20 CVR: mean={cvr_type20.mean():.2%}
Type 21 CVR: mean={cvr_type21.mean():.2%}

Probability that Type 20 has HIGHER CVR: {prob_cvr_type20_higher:.1%}

CVR Difference (Type 20 - Type 21):
  Mean: {cvr_diff_samples.mean():.2%}
  Median: {np.median(cvr_diff_samples):.2%}
  95% Credible Interval: [{np.percentile(cvr_diff_samples, 2.5):.2%},
                          {np.percentile(cvr_diff_samples, 97.5):.2%}]

Probability of meaningful CVR increase (>5%p): {prob_cvr_meaningful:.1%}
""")

# ============================================================================
# ARPPU에 대한 베이지안 분석
# ============================================================================

print("\n" + "="*70)
print("Bayesian Analysis: ARPPU Difference")
print("="*70)

arppu_type20 = mc_results['arppu_type20'].values
arppu_type21 = mc_results['arppu_type21'].values
arppu_diff_samples = arppu_type21 - arppu_type20

prob_arppu_type21_higher = (arppu_diff_samples > 0).mean()

print(f"""
Type 20 ARPPU: mean={arppu_type20.mean():,.0f} KRW
Type 21 ARPPU: mean={arppu_type21.mean():,.0f} KRW

Probability that Type 21 has HIGHER ARPPU: {prob_arppu_type21_higher:.1%}

ARPPU Difference (Type 21 - Type 20):
  Mean: {arppu_diff_samples.mean():,.0f} KRW
  Median: {np.median(arppu_diff_samples):,.0f} KRW
  95% Credible Interval: [{np.percentile(arppu_diff_samples, 2.5):,.0f} KRW,
                          {np.percentile(arppu_diff_samples, 97.5):,.0f} KRW]

ARPPU Difference (% change relative to Type 20):
  Mean: {arppu_diff_samples.mean()/arppu_type20.mean()*100:.1f}%
""")

# ============================================================================
# 베이지안 시각화
# ============================================================================

print("\n베이지안 분석 시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Revenue 사후 분포
axes[0, 0].hist(revenue_type20, bins=50, alpha=0.5, label='Type 20', density=True, color='blue')
axes[0, 0].hist(revenue_type21, bins=50, alpha=0.5, label='Type 21', density=True, color='orange')
axes[0, 0].axvline(revenue_type20.mean(), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(revenue_type21.mean(), color='orange', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Total Revenue (KRW)')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].set_title('Posterior Distribution: Total Revenue')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Revenue 차이 분포
axes[0, 1].hist(revenue_diff_samples, bins=50, alpha=0.7, color='coral')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
axes[0, 1].axvline(revenue_diff_samples.mean(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {revenue_diff_samples.mean():,.0f} KRW')
axes[0, 1].fill_betweenx([0, axes[0, 1].get_ylim()[1]],
                          np.percentile(revenue_diff_samples, 2.5),
                          np.percentile(revenue_diff_samples, 97.5),
                          alpha=0.3, color='coral', label='95% CI')
axes[0, 1].set_xlabel('Revenue Difference (Type 20 - Type 21, KRW)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Effect Distribution: Revenue')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Revenue CDF
sorted_rev_diff = np.sort(revenue_diff_samples)
cdf_rev = np.arange(1, len(sorted_rev_diff) + 1) / len(sorted_rev_diff)

axes[0, 2].plot(sorted_rev_diff, cdf_rev, linewidth=2, color='teal')
axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
axes[0, 2].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
axes[0, 2].fill_betweenx([0, 1], 0, sorted_rev_diff.max(), alpha=0.2, color='green',
                          label=f'Type 20 Higher: {prob_type20_higher:.1%}')
axes[0, 2].set_xlabel('Revenue Difference (KRW)')
axes[0, 2].set_ylabel('Cumulative Probability')
axes[0, 2].set_title('CDF: Revenue Difference')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 4. CVR 사후 분포
axes[1, 0].hist(cvr_type20, bins=50, alpha=0.5, label='Type 20', density=True, color='blue')
axes[1, 0].hist(cvr_type21, bins=50, alpha=0.5, label='Type 21', density=True, color='orange')
axes[1, 0].axvline(cvr_type20.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 0].axvline(cvr_type21.mean(), color='orange', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Conversion Rate')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].set_title('Posterior Distribution: CVR')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5. ARPPU 사후 분포
axes[1, 1].hist(arppu_type20, bins=50, alpha=0.5, label='Type 20', density=True, color='blue')
axes[1, 1].hist(arppu_type21, bins=50, alpha=0.5, label='Type 21', density=True, color='orange')
axes[1, 1].axvline(arppu_type20.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 1].axvline(arppu_type21.mean(), color='orange', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('ARPPU (KRW)')
axes[1, 1].set_ylabel('Probability Density')
axes[1, 1].set_title('Posterior Distribution: ARPPU')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. 시나리오 확률 (Revenue)
revenue_pct_diff = (revenue_diff_samples / revenue_type21.mean()) * 100
decision_thresholds = [-10, -5, 0, 5, 10, 15, 20]
decision_probs = [(revenue_pct_diff > t).mean() for t in decision_thresholds]

axes[1, 2].barh(range(len(decision_thresholds)), decision_probs, color='steelblue')
axes[1, 2].set_yticks(range(len(decision_thresholds)))
axes[1, 2].set_yticklabels([f'> {t}%' for t in decision_thresholds])
axes[1, 2].set_xlabel('Probability')
axes[1, 2].set_title('Scenario Probabilities\n(Type 20 vs Type 21)')
axes[1, 2].grid(axis='x', alpha=0.3)

for i, (prob, thresh) in enumerate(zip(decision_probs, decision_thresholds)):
    axes[1, 2].text(prob + 0.02, i, f'{prob:.1%}', va='center')

plt.tight_layout()

# dataSave 폴더에 저장
bayesian_viz_path = f'{saved_path}/bayesian_analysis.png'
plt.savefig(bayesian_viz_path, dpi=300, bbox_inches='tight')
print(f"\n✓ 베이지안 분석 시각화가 '{bayesian_viz_path}'에 저장되었습니다.")
plt.close()

# ============================================================================
# 의사결정 요약
# ============================================================================

print("\n" + "="*70)
print("Decision Summary: Frequentist vs Bayesian")
print("="*70)

print(f"""
[Revenue Comparison]
Frequentist:
  - p-value: {p_value_rev:.4f}
  - Statistically significant (p<0.05): {p_value_rev < 0.05}
  
Bayesian:
  - Probability Type 20 > Type 21: {prob_type20_higher:.1%}
  - 95% Credible Interval: [{np.percentile(revenue_diff_samples, 2.5):,.0f}, {np.percentile(revenue_diff_samples, 97.5):,.0f}] KRW
  - Probability of >5% increase: {prob_meaningful_increase_5pct:.1%}
  - Probability of >10% increase: {prob_meaningful_increase_10pct:.1%}

[CVR Comparison]
Frequentist:
  - p-value: {p_value_cvr:.6f}
  - Statistically significant (p<0.05): {p_value_cvr < 0.05}
  
Bayesian:
  - Probability Type 20 > Type 21: {prob_cvr_type20_higher:.1%}
  - Mean difference: {cvr_diff_samples.mean():.2%}

[ARPPU Comparison]
Frequentist:
  - p-value: {p_value_arppu:.6f}
  - Statistically significant (p<0.05): {p_value_arppu < 0.05}
  
Bayesian:
  - Probability Type 21 > Type 20: {prob_arppu_type21_higher:.1%}
  - Mean difference: {arppu_diff_samples.mean():,.0f} KRW

{'='*70}
Recommendation:
- Type 20 shows {prob_type20_higher:.1%} probability of higher revenue
- Type 20 has significantly higher CVR (p < 0.001)
- Type 21 has higher ARPPU but lower overall revenue
- Decision: {'Consider adopting Type 20' if prob_type20_higher > 0.75 else 'Need more data or analysis'}
{'='*70}
""")

print("\n" + "=" * 60)
print("모든 작업이 완료되었습니다!")
print("=" * 60)