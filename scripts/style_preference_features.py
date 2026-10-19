filtered_train_label_df.describe()
filtered_valid_label_df.describe()
# 스타일 선호 정보를 처리하는 함수
def process_style_preference(df, json_dir, data_type='train'):
    """
    df: DataFrame containing the Filename column.
    json_dir: Directory path where the json files are stored.
    data_type: 'train' 또는 'valid'로 구분하여 처리. (default: 'train')
    """

    # 새로운 데이터프레임을 위한 딕셔너리 생성(각 응답자 ID, 스타일 선호, 스타일 비선호 저장)
    style_preference_data = {
        "응답자 ID": [],
        # data_type에 저장된 문자열 첫 글자 대문자로 변환(ex. Train 스타일 선호)
        f"{data_type.capitalize()} 스타일 선호": [],
        f"{data_type.capitalize()} 스타일 비선호": []
    }

    # JSON 파일들을 탐색하고 데이터 처리
    for idx, row in df.iterrows():   # DataFrame인 df를 iterrows()를 사용해 각 행 반복 처리
        filename = row['File_Name']  # json 파일명 추출
        filepath = os.path.join(json_dir, filename)  # 파일 경로 설정

        # 파일 존재 여부 확인 후 처리
        if os.path.exists(filepath): # 파일 경로가 실제 존재하는지 확인
            # JSON 파일 읽기(파일 읽기 모드로 열고 내용 로드)
            with open(filepath, 'r') as file:
                data = json.load(file)

            # 응답자 ID 및 Q5 값 추출(응답자 ID와 스타일 선호 여부 추출)
            respondent_id = data['user']['R_id']
            preference = data['item']['survey']['Q5']
            img_name = data['item']['imgName']

            # 추출한 응답자 ID를 딕셔너리의 리스트에 추가
            style_preference_data['응답자 ID'].append(respondent_id)

            # 선호 여부에 따라 분류
            if preference == 2:  # Q5 값이 2면 선호로 분류(비선호 리스트에는 None 추가)
                style_preference_data[f"{data_type.capitalize()} 스타일 선호"].append(img_name)
                style_preference_data[f"{data_type.capitalize()} 스타일 비선호"].append(None)
            elif preference == 1:  # Q5 값이 1이면 비선호로 분류(선호 리스트에는 None 추가)
                style_preference_data[f"{data_type.capitalize()} 스타일 선호"].append(None)
                style_preference_data[f"{data_type.capitalize()} 스타일 비선호"].append(img_name)
        else:
            print(f"File {filename} does not exist.") # 파일이 존재하지 않으면 오류 메시지 출력

    # 딕셔너리 style_preference_data를 데이터프레임으로 변환 및 반환
    df_style_preference = pd.DataFrame(style_preference_data)

    return df_style_preference

# Example usage:
# train 데이터 처리해 DataFrame 생성
train_result_df = process_style_preference(filtered_train_label_df, train_label_dir, data_type='train')

# valid 데이터 처리해 DataFrame 생성
valid_result_df = process_style_preference(filtered_valid_label_df, valid_label_dir, data_type='valid')

# 결과 출력
print(train_result_df)
print(valid_result_df)
# 유효한 train data 분석시, 응답자 ID별 총 tuple개수(unique로 중복된 ID 하나로 처리, 응답자 ID 개수 계산)
len(train_result_df['응답자 ID'].unique())

# None 값을 제거하고 빈 리스트도 NaN으로 변환하는 함수
def remove_none_and_empty(lst):
    if isinstance(lst, list):
        # None을 제거한 리스트 반환, 빈 리스트일 경우 NaN으로 대체
        cleaned_list = [x for x in lst if x is not None] # 리스트에서 None 제거
        return cleaned_list if cleaned_list else np.nan  # 빈 리스트는 NaN으로 처리
    return lst

# 응답자 ID별로 데이터를 그룹화, 같은 응답자끼리 데이터를 리스트로 묶음
grouped_train = train_result_df.groupby('응답자 ID').agg(list).reset_index()

# 각 열에 대해 remove_none 함수 적용(None 제거 및 빈 리스트 처리)
for col in grouped_train.columns:
    if col != '응답자 ID':  # 응답자 ID 열은 처리하지 않음
        # 그룹화된 데이터에서 None 제거 및 빈 리스트는 NaN으로 변환(열별로 apply 사용)
        grouped_train[col] = grouped_train[col].apply(remove_none_and_empty)

# 결과 출력(None 값이 제거 된 train 데이터)
print(grouped_train)

# 유효한 valid data 분석시, 응답자 ID별 총 tuple개수
len(valid_result_df['응답자 ID'].unique())

# 응답자 ID별로 데이터를 그룹화, 같은 응답자끼리 데이터를 리스트로 묶음
grouped_valid = valid_result_df.groupby('응답자 ID').agg(list).reset_index()

# 각 열에 대해 remove_none 함수 적용(None 제거 및 빈 리스트 처리)
for col in grouped_valid.columns:
    if col != '응답자 ID':  # 응답자 ID 열은 처리하지 않음
        # 그룹화된 데이터에서 None 제거 및 빈 리스트는 NaN으로 변환(열별로 apply 사용)
        grouped_valid[col] = grouped_valid[col].apply(remove_none_and_empty)

# 결과 출력(None 값이 제거 된 valid 데이터)
print(grouped_valid)

# '응답자 ID'를 기준으로 train과 valid 데이터 병합(outer join으로 양쪽 데이터를 모두 유지)
merged_df = pd.merge(grouped_train, grouped_valid, on='응답자 ID', how='inner', suffixes=('_train', '_valid'))

# 결과 출력(train과 valid 병합된 데이터)
print(merged_df)

print(merged_df.columns)

# 'total_count' 컬럼 삭제
merged_df = merged_df.drop(columns=['Total_Count'])

# 상위 100명 추출 후 인덱스 재설정
merged_df_100 = merged_df.head(100).reset_index(drop=True)

# 결과 출력
print(merged_df_100)

# CSV 파일로 저장
file_path = '/content/drive/MyDrive/2_2.csv' # 경로 지정
# 인덱스는 저장하지 않음, 인코딩 문제 방지
merged_df_100.to_csv(file_path, index=False, encoding='utf-8-sig')
print('경로 :', file_path, '2-2 정답 CSV 파일 저장완료') # 저장완료 메시지 출력

# Train과 Valid 응답자 ID 출력 및 교집합 확인
train_ids = set(grouped_train['응답자 ID'])
valid_ids = set(grouped_valid['응답자 ID'])

# 교집합 확인
common_ids = train_ids.intersection(valid_ids)
print(f"공통된 응답자 ID 수: {len(common_ids)}")

# 교집합이 없는 경우, 데이터가 올바르게 로드되고 있는지 확인해야 합니다.
if len(common_ids) == 0:
    print("Train과 Valid 데이터에 공통된 응답자 ID가 없습니다.")
print(train_result_df.head())  # Train 스타일 선호 정보 출력
print(valid_result_df.head())  # Valid 스타일 선호 정보 출력

# 응답자 ID가 제대로 추출되었는지 확인
print(f"Train 응답자 ID 개수: {len(train_result_df['응답자 ID'].unique())}")
print(f"Valid 응답자 ID 개수: {len(valid_result_df['응답자 ID'].unique())}")

# 1. Train과 Valid 응답자 ID 개수 확인
print(f"Train 응답자 ID 개수: {len(train_result_df['응답자 ID'].unique())}")
print(f"Valid 응답자 ID 개수: {len(valid_result_df['응답자 ID'].unique())}")

# 2. 응답자 ID별 스타일 선호 정보 미리보기
print(train_result_df.head())
print(valid_result_df.head())

# 3. Train과 Valid 응답자 ID 교집합 확인
train_ids = set(grouped_train['응답자 ID'])
valid_ids = set(grouped_valid['응답자 ID'])
common_ids = train_ids.intersection(valid_ids)
print(f"공통된 응답자 ID 수: {len(common_ids)}")
if len(common_ids) == 0:
    print("⚠️ Train과 Valid 데이터에 공통된 응답자 ID가 없습니다.")

# 4. JSON 파일 누락 여부 확인
missing_files = []
for idx, row in filtered_train_label_df.iterrows():
    filename = row['File_Name']
    filepath = os.path.join(train_label_dir, filename)
    if not os.path.exists(filepath):
        missing_files.append(filename)
if missing_files:
    print(f"누락된 JSON 파일 수: {len(missing_files)}")
    print("누락된 파일 목록:", missing_files)
else:
    print("모든 JSON 파일이 존재합니다.")

# 5. 병합된 데이터 확인
print(merged_df.head())
print(f"병합된 데이터 응답자 수: {len(merged_df)}")

# 6. 상위 100명 데이터 확인
print(merged_df_100.head())
print(f"상위 100명 응답자 수: {len(merged_df_100)}")
