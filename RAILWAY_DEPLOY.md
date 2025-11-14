# Railway 배포 가이드

## 배포 단계

### 1. Railway 계정 생성 및 프로젝트 연결
1. [Railway](https://railway.app) 접속 후 GitHub로 로그인
2. "New Project" 클릭
3. "Deploy from GitHub repo" 선택
4. 이 저장소 선택

### 2. Qdrant 서비스 추가 (Railway 템플릿 사용)
1. 같은 프로젝트에서 **"New"** 버튼 클릭
2. **"Template"** 또는 **"Database"** 섹션에서 **Qdrant** 선택
3. 또는 **"Deploy from GitHub"** → Qdrant 공식 저장소 사용
4. Qdrant 서비스가 자동으로 배포됨
5. 배포 완료 후 Qdrant 서비스의 **"Variables"** 탭에서 내부 URL 확인

### 3. 환경 변수 설정
FastAPI 서비스의 환경 변수 설정:

```
QDRANT_URL=http://qdrant:6333  (같은 프로젝트 내 서비스 간 통신)
또는
QDRANT_URL=your_qdrant_service_url  (Railway가 제공하는 내부 URL)
QDRANT_API_KEY=  (선택사항, 템플릿에 따라 다를 수 있음)
QDRANT_COLLECTION=qa_collection
QA_EXCEL_PATH=Q&A.xlsx
DEBUG_WORKFLOW=0
```

**참고**: 
- Railway 내부 서비스 간 통신은 내부 네트워크 사용 (더 빠름)
- Qdrant 템플릿이 없다면 [Qdrant Cloud](https://cloud.qdrant.io) 사용 가능

### 4. 엑셀 파일 업로드
Railway는 파일 시스템이 임시이므로, 엑셀 파일을 저장소에 포함시키거나:
- GitHub에 엑셀 파일 커밋 (권장)
- 또는 S3/클라우드 스토리지 사용

### 5. 배포 확인
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

## 빌드 타임아웃 해결

빌드 타임아웃이 발생하는 경우:

1. **Dockerfile 사용** (권장)
   - 프로젝트에 `Dockerfile`이 있으면 Railway가 자동으로 사용
   - 더 효율적인 빌드 프로세스

2. **.railwayignore 파일**
   - 불필요한 파일 제외로 빌드 시간 단축
   - `__pycache__`, `.history` 등 자동 제외

3. **Railway 설정에서 타임아웃 증가**
   - 서비스 → Settings → Build & Deploy
   - Build timeout 설정 확인/증가

4. **재배포**
   - 변경사항 커밋 후 자동 재배포
   - Dockerfile 사용 시 더 빠른 빌드

## 주의사항

1. **Qdrant 연결**: Railway 내부 서비스 사용 시 내부 URL 사용 (더 빠르고 안전)
2. **엑셀 파일**: 저장소에 포함하거나 클라우드 스토리지 사용
3. **모델 다운로드**: SentenceTransformer 모델은 첫 실행 시 자동 다운로드 (시간 소요)
4. **무료 티어 제한**: Railway 무료 티어는 월 $5 크레딧 제공
5. **빌드 최적화**: Dockerfile과 .railwayignore로 빌드 시간 단축

