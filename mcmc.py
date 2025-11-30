import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

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

# 전환율 설정 (원래대로 유지)
CONVERSION_RATE_TYPE20 = 0.30
CONVERSION_RATE_TYPE21 = 0.20

# 멤버십 분포 (type별)
# type=20: 일주일 멤버십이 더 높음
MEMBERSHIP_DIST_TYPE20 = {
    'weekly': 0.60,
    'monthly': 0.40
}

# type=21: 한달 멤버십 비중을 대폭 높임 (ARPU 증가)
MEMBERSHIP_DIST_TYPE21 = {
    'weekly': 0.20,  # 35% → 20%로 감소
    'monthly': 0.80  # 65% → 80%로 증가
}

# AI 추천권 구매율 (일주일 멤버십 구매자 중)
AI_TICKET_RATE = 0.70

# 가격 설정
PRICES = {
    'ai 추천권': 5000,
    '일주일 멤버십': 20000,
    '한달 멤버십': 50000
}

# 재구매율
REPURCHASE_RATE = 0.50


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
    user_purchase_history = {}  # {user_id: [month_indices]}

    merged_df = users_df.merge(ab_tests_df, left_on='id', right_on='user_id')

    for _, user in merged_df.iterrows():
        user_id = user['id']
        ab_type = user['type']
        signup_month_idx = MONTHS.index(user['signup_month'])

        # 현재 월부터 마지막 월까지 반복
        for current_month_idx in range(signup_month_idx, len(MONTHS)):
            current_month = MONTHS[current_month_idx]

            # 첫 구매 여부 결정
            if current_month_idx == signup_month_idx:
                # 신규 유저 전환율
                conversion_rate = CONVERSION_RATE_TYPE20 if ab_type == 20 else CONVERSION_RATE_TYPE21
                will_purchase = random.random() < conversion_rate
            else:
                # 재구매 여부 (이전 달에 구매했으면)
                if user_id in user_purchase_history and (current_month_idx - 1) in user_purchase_history[user_id]:
                    will_purchase = random.random() < REPURCHASE_RATE
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


# 시뮬레이션 실행
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

# Monte Carlo 결과도 Google Drive에 저장
mc_results.to_csv(f'{saved_path}/monte_carlo_results.csv', index=False, encoding='utf-8-sig')
print(f"\nMonte Carlo 결과가 '{saved_path}/monte_carlo_results.csv'에 저장되었습니다.")

# 시각화를 위한 추가 코드 (선택사항)
try:
    import matplotlib.pyplot as plt

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
    print(f"시각화 결과가 '{viz_path}'에 저장되었습니다.")

except ImportError:
    print("\n시각화를 위해 matplotlib을 설치하세요: pip install matplotlib")

print("\n" + "=" * 60)
print("모든 작업이 완료되었습니다!")
print("=" * 60)