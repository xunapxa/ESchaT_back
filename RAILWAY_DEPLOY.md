# Railway 배포 가이드

## 배포 단계

### 1. Railway 계정 생성 및 프로젝트 연결
1. [Railway](https://railway.app) 접속 후 GitHub로 로그인
2. "New Project" 클릭
3. "Deploy from GitHub repo" 선택
4. 이 저장소 선택

### 2. 환경 변수 설정
Railway 대시보드에서 환경 변수 설정:

```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key (선택사항)
QDRANT_COLLECTION=qa_collection
QA_EXCEL_PATH=Q&A.xlsx
DEBUG_WORKFLOW=0
```

**중요**: Qdrant는 Railway에서 제공하지 않으므로:
- [Qdrant Cloud](https://cloud.qdrant.io)에서 무료 계정 생성
- 또는 자체 호스팅 Qdrant 서버 사용

### 3. 엑셀 파일 업로드
Railway는 파일 시스템이 임시이므로, 엑셀 파일을 저장소에 포함시키거나:
- GitHub에 엑셀 파일 커밋 (권장)
- 또는 S3/클라우드 스토리지 사용

### 4. 배포 확인
- Railway가 자동으로 배포 시작
- 배포 완료 후 제공되는 URL로 `/health` 엔드포인트 테스트
- `/admin/init` 엔드포인트로 데이터 초기화

## 프론트엔드 배포

### Railway에서 함께 배포
1. 같은 프로젝트에 새 서비스 추가
2. 프론트엔드 저장소 연결
3. 빌드 명령어 설정 (예: `npm run build`)
4. 시작 명령어 설정 (예: `npm start`)

### 또는 Vercel 사용 (프론트엔드만)
- 프론트엔드는 Vercel이 더 간단할 수 있음
- CORS 설정 확인 필요

## 주의사항

1. **Qdrant 연결**: Railway에서 Qdrant Cloud URL 사용 시 공개 URL이어야 함
2. **엑셀 파일**: 저장소에 포함하거나 클라우드 스토리지 사용
3. **모델 다운로드**: SentenceTransformer 모델은 첫 실행 시 자동 다운로드 (시간 소요)
4. **무료 티어 제한**: Railway 무료 티어는 월 $5 크레딧 제공

