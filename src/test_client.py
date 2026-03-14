import requests
import json

# 서버 주소 (로컬 내부망에서 직접 찌르기 때문에 프록시 에러가 발생하지 않습니다)
API_URL = "http://127.0.0.1:8000/analyze/link"

# 테스트해볼 유튜브 쇼츠나 릴스 링크를 넣으세요!
payload = {
    "url": "https://www.youtube.com/shorts/IPvoi5sfXok?feature=share"  # 임의의 쇼츠 주소 예시
}

print("🚀 H200 서버로 링크 전송 중... (약 3~5초 소요)")

try:
    response = requests.post(API_URL, json=payload)
    
    # HTTP 상태 코드가 200(성공)일 때
    if response.status_code == 200:
        print("\n✅ [분석 완료] API 응답 결과:")
        # 결과를 보기 좋게 예쁘게 출력
        result = response.json()
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print(f"\n❌ [에러 발생] 상태 코드: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\n🚨 서버 접속 실패: {e}")