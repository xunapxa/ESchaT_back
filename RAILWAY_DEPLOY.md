# Railway 배포 가이드

## 배포 단계

### 1. Railway 계정 생성 및 프로젝트 연결
1. [Railway](https://railway.app) 접속 후 GitHub로 로그인
2. "New Project" 클릭
3. "Deploy from GitHub repo" 선택
4. 이 저장소 선택

### 2. Qdrant 서비스 추가 (Railway 템플릿 사용)
1. 같은 프로젝트에서 **"New"** 버튼 클릭
2. **"Template"** 섹션에서 **Qdrant** 선택
3. Qdrant 서비스가 자동으로 배포됨

**중요: Qdrant 서비스 환경 변수 설정**
Qdrant 서비스의 Variables 탭에서 다음 환경 변수 추가:

```
QDRANT__SERVICE__HTTP_HOST=0.0.0.0
```

이 설정은 IPv6 연결 문제를 해결합니다.

### 3. 환경 변수 설정

**옵션 A: Qdrant Cloud 사용 (권장 - 더 안정적)**

1. [Qdrant Cloud](https://cloud.qdrant.io)에서 무료 클러스터 생성
2. 클러스터 URL과 API Key 복사
3. FastAPI 서비스의 Variables에 설정:

```
QDRANT_URL=https://your-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
QDRANT_COLLECTION=qa_collection
QA_EXCEL_PATH=ESTSoft.xlsx
DEBUG_WORKFLOW=0
```

**옵션 B: Railway Qdrant 서비스 사용**

```
QDRANT_URL=https://qdrant-production-xxxx.up.railway.app
QDRANT_COLLECTION=qa_collection
QA_EXCEL_PATH=ESTSoft.xlsx
DEBUG_WORKFLOW=0
```

**참고**: 
- **Qdrant Cloud 사용을 강력히 권장** (더 안정적이고 빠름)
- Railway의 Qdrant는 내부 네트워크 연결에 IPv6 문제가 있을 수 있음
- Qdrant Cloud는 무료 티어 제공 (제한적이지만 충분함)

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

빌드 타임아웃이 계속 발생하는 경우:

### 방법 1: Railway 설정에서 타임아웃 증가 (권장)
1. Railway 대시보드 → 서비스 선택
2. **Settings** → **Build & Deploy** 탭
3. **Build timeout** 설정을 **30분** 또는 **60분**으로 증가
4. 저장 후 재배포

### 방법 2: Procfile 사용 (Dockerfile 대신)
Dockerfile이 타임아웃을 유발하는 경우:
1. Dockerfile을 임시로 이름 변경: `Dockerfile.bak`
2. Railway가 자동으로 Procfile 사용
3. Procfile이 더 관대한 타임아웃 정책 적용

### 방법 3: Dockerfile 최적화
- pip timeout 증가 (이미 적용됨)
- 불필요한 패키지 제외
- 멀티스테이지 빌드 사용

### 방법 4: .railwayignore 파일
- 불필요한 파일 제외로 빌드 시간 단축
- `__pycache__`, `.history` 등 자동 제외

## 주의사항

1. **Qdrant 연결**: Railway 내부 서비스 사용 시 내부 URL 사용 (더 빠르고 안전)
2. **엑셀 파일**: 저장소에 포함하거나 클라우드 스토리지 사용
3. **모델 다운로드**: SentenceTransformer 모델은 첫 실행 시 자동 다운로드 (시간 소요)
4. **무료 티어 제한**: Railway 무료 티어는 월 $5 크레딧 제공
5. **빌드 최적화**: Dockerfile과 .railwayignore로 빌드 시간 단축

