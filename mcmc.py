# =======================================================
# Simulation: CVR 30%/20%, ARPU/ARPPU 계산, LPM/OLS 회귀
# =======================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import statsmodels.formula.api as smf

# ===================== CONFIG =====================
np.random.seed(42)
random.seed(42)

MONTHS = ['2025-05', '2025-06', '2025-07', '2025-08', '2025-09', '2025-10']
EARLY_MONTHS = ['2025-05', '2025-06', '2025-07']
LATE_MONTHS = ['2025-08', '2025-09', '2025-10']

USERS_PER_MONTH = [300, 600, 200, 200, 200, 200]
MALE_RATIO = 600 / 900
MALE_BIRTH_MEAN = 1995
MALE_BIRTH_STD = 3
FEMALE_BIRTH_MEAN = 1998
FEMALE_BIRTH_STD = 3

CONVERSION_RATE_TYPE20 = 0.3
CONVERSION_RATE_TYPE21 = 0.2

MEMBERSHIP_DIST_TYPE20 = {'weekly': 0.7, 'monthly': 0.3}
MEMBERSHIP_DIST_TYPE21 = {'weekly': 0.25, 'monthly': 0.75}
AI_TICKET_RATE = 0.75

PRICES = {'ai 추천권': 5000, '일주일 멤버십': 30000, '한달 멤버십': 100000}

REPURCHASE_RATE_TYPE20 = 0.5
REPURCHASE_RATE_TYPE21 = 0.4

ANSWER_RATES_EARLY = [0.99, 0.98, 0.95, 0.85, 0.83]
ANSWER_RATES_LATE = [0.99, 0.98, 0.95, 0.90, 0.88]

# ===================== FUNCTIONS =====================
def generate_users(month, n_users_per_month):
    users = []
    start_date = datetime.strptime(f"{month}-01", "%Y-%m-%d")
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    for i in range(n_users_per_month):
        random_day = random.randint(0, (end_date - start_date).days)
        timestamp = start_date + timedelta(days=random_day,
                                           hours=random.randint(0,23),
                                           minutes=random.randint(0,59))
        gender = 'M' if random.random() < MALE_RATIO else 'F'
        birth = int(np.random.normal(MALE_BIRTH_MEAN if gender=='M' else FEMALE_BIRTH_MEAN,
                                     MALE_BIRTH_STD if gender=='M' else FEMALE_BIRTH_STD))
        users.append({
            'timestamp': timestamp,
            'user_id': f"{month}_user_{i+1}",
            'gender': gender,
            'birth': birth,
            'signup_month': month
        })
    return users

def generate_ab_tests(users_df):
    ab_tests = []
    for _, user in users_df.iterrows():
        if user['signup_month'] == '2025-06':
            ab_type = 20 if random.random() < 0.5 else 21
            ab_tests.append({
                'timestamp': user['timestamp'],
                'user_id': user['user_id'],
                'type': ab_type
            })
    return ab_tests

def generate_answers(users_df):
    answers = []
    for _, user in users_df.iterrows():
        rates = ANSWER_RATES_EARLY if user['signup_month'] in EARLY_MONTHS else ANSWER_RATES_LATE
        for answer_id in range(1,6):
            # 랜덤 노이즈 적용
            rate = max(0, min(1, np.random.normal(rates[answer_id-1], 0.05)))
            if random.random() < rate:
                created = user['timestamp'] + timedelta(hours=random.randint(0,24))
                answers.append({
                    'created': created,
                    'answer_id': answer_id,
                    'user_id': user['user_id'],
                    'answer': random.choice([1,2])
                })
    return answers

def generate_payments(users_df, ab_tests_df):
    payments = []
    user_hist = {}
    ab_map = dict(zip(ab_tests_df['user_id'], ab_tests_df['type']))
    for _, user in users_df.iterrows():
        user_id = user['user_id']
        signup_idx = MONTHS.index(user['signup_month'])
        ab_type = ab_map[user_id] if user_id in ab_map else 20
        rep_rate = REPURCHASE_RATE_TYPE20 if ab_type==20 else REPURCHASE_RATE_TYPE21
        for m_idx in range(signup_idx, len(MONTHS)):
            month = MONTHS[m_idx]
            if m_idx == signup_idx:
                conv_rate = CONVERSION_RATE_TYPE20 if ab_type==20 else CONVERSION_RATE_TYPE21
                will_purchase = random.random() < conv_rate
            else:
                will_purchase = user_id in user_hist and (m_idx-1) in user_hist[user_id] and random.random() < rep_rate
            if will_purchase:
                user_hist.setdefault(user_id, []).append(m_idx)
                membership = 'weekly' if random.random() < (MEMBERSHIP_DIST_TYPE20['weekly'] if ab_type==20 else MEMBERSHIP_DIST_TYPE21['weekly']) else 'monthly'
                order_name = '일주일 멤버십' if membership=='weekly' else '한달 멤버십'
                start = datetime.strptime(f"{month}-01", "%Y-%m-%d")
                end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                random_day = random.randint(0, (end-start).days)
                created = start + timedelta(days=random_day,
                                            hours=random.randint(0,23),
                                            minutes=random.randint(0,59))
                payments.append({
                    'created': created,
                    'user_id': user_id,
                    'order_name': order_name,
                    'amount': PRICES[order_name],
                    'cancel': 0
                })
                if ab_type==20 and membership=='weekly' and random.random() < AI_TICKET_RATE:
                    payments.append({
                        'created': created + timedelta(seconds=random.randint(1,300)),
                        'user_id': user_id,
                        'order_name': 'ai 추천권',
                        'amount': PRICES['ai 추천권'],
                        'cancel': 0
                    })
    return payments

# ===================== RUN SIMULATION =====================
all_users = []
for idx, m in enumerate(MONTHS):
    all_users.extend(generate_users(m, USERS_PER_MONTH[idx]))
users_df = pd.DataFrame(all_users)

ab_tests_df = pd.DataFrame(generate_ab_tests(users_df))
answers_df = pd.DataFrame(generate_answers(users_df))
payments_df = pd.DataFrame(generate_payments(users_df, ab_tests_df))

# ===================== SAVE CSVs =====================
os.makedirs('dataSave', exist_ok=True)
users_df.to_csv('dataSave/users.csv', index=False, encoding='utf-8-sig')
ab_tests_df.to_csv('dataSave/ab_tests.csv', index=False, encoding='utf-8-sig')
answers_df.to_csv('dataSave/answers.csv', index=False, encoding='utf-8-sig')
payments_df.to_csv('dataSave/payments.csv', index=False, encoding='utf-8-sig')

# ===================== PER-USER METRICS =====================
user_metrics = users_df.merge(ab_tests_df[['user_id','type']], on='user_id', how='left')

# CVR: 결제 여부 기준
user_conv = payments_df.groupby('user_id').size().rename('converted')
user_conv = (user_conv > 0).astype(int)
user_metrics = user_metrics.merge(user_conv, on='user_id', how='left')
user_metrics['converted'] = user_metrics['converted'].fillna(0)

# ARPU: 전체 사용자 기준
user_rev = payments_df.groupby('user_id')['amount'].sum().rename('revenue')
user_metrics = user_metrics.merge(user_rev, on='user_id', how='left')
user_metrics['revenue'] = user_metrics['revenue'].fillna(0)
user_metrics['arpu'] = user_metrics['revenue']

# ARPPU: 구매자 기준
arppu_metrics = user_metrics[user_metrics['converted']==1].copy()
arppu_metrics['arppu'] = arppu_metrics['revenue']

# ===================== EDA =====================
eda = user_metrics.groupby("type")["converted"].agg(
    conversions="sum",
    non_conversions=lambda x: (1-x).sum(),
    total="count",
    cvr=lambda x: x.mean()
)
print("=== EDA: A/B Group Conversion Status ===")
print(eda)

# ===================== REGRESSION =====================
# LPM (CVR)
print("\n=== OLS Regression (LPM): CVR ~ A/B Type ===")
lpm_model = smf.ols("converted ~ C(type)", data=user_metrics).fit()
print(lpm_model.summary())

# ARPU (전체 사용자)
print("\n=== OLS Regression: ARPU ~ A/B Type ===")
model_arpu = smf.ols("arpu ~ C(type)", data=user_metrics).fit()
print(model_arpu.summary())

# ARPPU (구매자 기준)
print("\n=== OLS Regression: ARPPU ~ A/B Type ===")
model_arppu = smf.ols("arppu ~ C(type)", data=arppu_metrics).fit()
print(model_arppu.summary())

# ===================== Before/After 응답률 =====================
before_months = ['2025-05','2025-06','2025-07']
after_months = ['2025-08','2025-09','2025-10']

# 전체 사용자 x answer_id 모든 조합
all_users_answers = pd.MultiIndex.from_product(
    [users_df['user_id'], range(1,6)],
    names=['user_id','answer_id']
).to_frame(index=False)

# 실제 응답 여부 반영
answers_lookup = set(zip(answers_df['user_id'], answers_df['answer_id']))
all_users_answers['answered'] = all_users_answers.apply(
    lambda row: 1 if (row['user_id'], row['answer_id']) in answers_lookup else 0,
    axis=1
)

# Before/After 응답률 계산
all_users_answers = all_users_answers.merge(users_df[['user_id','signup_month']], on='user_id')
before_rate = all_users_answers[all_users_answers['signup_month'].isin(before_months)].groupby('answer_id')['answered'].mean()
after_rate = all_users_answers[all_users_answers['signup_month'].isin(after_months)].groupby('answer_id')['answered'].mean()

response_rates = pd.concat([before_rate, after_rate], axis=1)
response_rates.columns = ['before_response_rate','after_response_rate']

print("\n=== Before/After Response Rates ===")
print(response_rates)
