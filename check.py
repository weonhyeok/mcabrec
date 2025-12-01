# Check with Results from Redash

import pandas as pd

# 저장된 CSV 파일 읽기
users_df = pd.read_csv('dataSave/users.csv')
ab_tests_df = pd.read_csv('dataSave/ab_tests.csv')
payments_df = pd.read_csv('dataSave/payments.csv')
answers_df = pd.read_csv('dataSave/answers.csv')

print("=" * 60)
print("저장된 CSV 데이터 검증")
print("=" * 60)

# AB 테스트와 결제 데이터 조인
merged_df = ab_tests_df.merge(
    payments_df, 
    left_on='user_id', 
    right_on='user_id', 
    how='left'
)

# Type별 집계
result = ab_tests_df.groupby('type').agg(
    num_tot_users=('user_id', 'nunique')
).reset_index()

# 결제 유저 수 계산
paid_users = payments_df.merge(ab_tests_df, on='user_id').groupby('type').agg(
    num_paid_users=('user_id', 'nunique'),
    total_revenue=('amount', 'sum'),
    total_orders=('order_name', 'count')
).reset_index()

# 결과 병합
result = result.merge(paid_users, on='type', how='left')

# 지표 계산
result['CVR'] = round(100.0 * result['num_paid_users'] / result['num_tot_users'], 2)
result['ARPU'] = round(result['total_revenue'] / result['num_tot_users'], 2)
result['ARPPU'] = round(result['total_revenue'] / result['num_paid_users'], 2)

print("\n[CSV 데이터 분석 결과]")
print(result)

print("\n[상세 비교]")
for _, row in result.iterrows():
    print(f"\nType {int(row['type'])}:")
    print(f"  총 유저: {int(row['num_tot_users'])}명")
    print(f"  결제 유저: {int(row['num_paid_users'])}명")
    print(f"  CVR: {row['CVR']}%")
    print(f"  총 매출: {int(row['total_revenue']):,}원")
    print(f"  ARPU: {row['ARPU']:,.2f}원")
    print(f"  ARPPU: {row['ARPPU']:,.2f}원")
    print(f"  총 주문: {int(row['total_orders'])}건")

# 추가: 주문 타입별 분포 확인
print("\n" + "=" * 60)
print("주문 타입별 분포")
print("=" * 60)
order_dist = payments_df.merge(ab_tests_df, on='user_id').groupby(['type', 'order_name']).size().reset_index(name='count')
for type_val in [20, 21]:
    print(f"\nType {type_val}:")
    type_orders = order_dist[order_dist['type'] == type_val]
    for _, row in type_orders.iterrows():
        print(f"  {row['order_name']}: {row['count']}건")

        
