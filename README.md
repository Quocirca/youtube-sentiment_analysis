# youtube-sentiment_analysis# sentiment_analysis.py
# -----------------------------------------------
# Youtube 댓글 데이터셋 감정 분석 프로젝트
# Kaggle 데이터셋 기반
# -----------------------------------------------

import pandas as pd
from textblob import TextBlob
import os

# 1. 데이터 불러오기
# (같은 폴더에 YoutubeCommentsDataSet.csv 파일이 있어야 함)
file_path = 'YoutubeCommentsDataSet.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"데이터 파일이 없습니다: {file_path}")

df = pd.read_csv(file_path)

# 2. 컬럼명 통일
df.rename(columns={'Comment': 'comment_text', 'Sentiment': 'original_sentiment'}, inplace=True)

# 3. 감정 점수 계산
# polarity: -1(매우 부정) ~ 1(매우 긍정)
df['polarity'] = df['comment_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# 4. 감정 분류 함수
def get_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['predicted_sentiment'] = df['polarity'].apply(get_sentiment)

# 5. 결과 확인
print("=== 샘플 데이터 확인 ===")
print(df.head())

# 6. 분석 결과 저장
output_path = 'sentiment_results.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"분석 결과 저장 완료: {output_path}")

# 7. GitHub 업로드 팁
# - 이 파일(sentiment_analysis.py)과 CSV 파일(sentiment_results.csv, 원본 데이터셋)을 같은 폴더에 둡니다.
# - VS Code에서 Git 초기화 후 commit & push 하면 됩니다.
