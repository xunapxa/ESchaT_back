import os
import re
from typing import Dict, List, Optional

import pandas as pd

DEBUG_PARSE = os.getenv("DEBUG_PARSE", "0") == "1"


def _p(msg: str) -> None:
    if DEBUG_PARSE:
        print(f"[PARSE] {msg}")


def read_excel_flex(filepath: str) -> pd.DataFrame:
    def try_read(header_row: Optional[int]) -> Optional[pd.DataFrame]:
        try:
            df_local = pd.read_excel(filepath, header=header_row)
            df_local.columns = [str(c).strip() for c in df_local.columns]
            return df_local
        except Exception:
            return None

    candidates: List[pd.DataFrame] = []
    for hdr in [0, 1, 2, 3, 4, 5, None]:
        df_try = try_read(hdr)
        if df_try is not None:
            candidates.append(df_try)
    if not candidates:
        raise ValueError("엑셀 파일을 읽을 수 없습니다.")
    return max(candidates, key=lambda d: d.shape[1])


def parse_qa_data(df: pd.DataFrame) -> List[Dict[str, str]]:
    possible_content_cols = ["내용", "내 용", "content", "Content", "contents", "Contents"]

    # 0) 모든 컬럼에 대해 Q/A 신호를 스코어링하여 최적의 내용 컬럼을 찾는다
    def qa_signal_count(series_like: pd.Series) -> int:
        lines = []
        for v in series_like.astype(str).fillna("").tolist():
            if not v or v.lower() == "nan":
                continue
            parts = re.split(r"[\r\n]+", str(v))
            for p in parts:
                p = p.strip()
                if p:
                    lines.append(p)
        cnt_q = sum(bool(re.match(r"^\s*Q[\.\)\:]?\s+", s, re.IGNORECASE)) for s in lines)
        cnt_a = sum(bool(re.match(r"^\s*A[\.\)\:]?\s+", s, re.IGNORECASE)) for s in lines)
        return cnt_q + cnt_a

    best_col = None
    best_score = -1
    for col in df.columns:
        try:
            score = qa_signal_count(df[col])
            if DEBUG_PARSE:
                _p(f"score col='{col}' -> {score}")
            if score > best_score:
                best_score = score
                best_col = col
        except Exception:
            continue

    # 1) 우선순위: 사전 지정된 이름 > 스코어가 가장 높은 컬럼
    content_col = next((c for c in possible_content_cols if c in df.columns), None)
    if best_col is not None and best_score > 0:
        content_col = best_col

    if content_col is None:
        if df.shape[1] >= 2:
            df = df.iloc[:, [1]].copy()
            df.columns = ["내용"]
            content_col = "내용"
        else:
            raise ValueError("엑셀에서 '내용' 컬럼을 찾지 못했습니다.")

    series = df[content_col].astype(str).fillna("")
    _p(f"content_col='{content_col}', total_rows={len(series)} (best_score={best_score})")
    for idx, val in list(series.head(8).items()):
        _p(f"row[{idx}] = {val}")

    possible_idx_cols = ["순번", "번호", "No", "no", "index", "Index"]
    idx_col = next((c for c in possible_idx_cols if c in df.columns), None)
    _p(f"index_col='{idx_col}'")

    def normalize_cell(text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"^(Q|A)[\.\)\:]?\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def is_q(text: str) -> bool:
        return bool(re.match(r"^\s*Q[\.\)\:]?\s+", str(text), re.IGNORECASE))

    def is_a(text: str) -> bool:
        return bool(re.match(r"^\s*A[\.\)\:]?\s+", str(text), re.IGNORECASE))

    records: List[Dict[str, str]] = []

    if idx_col:
        group_ids = df[idx_col].replace("", pd.NA)
        group_ids = group_ids.bfill().ffill()
        df["_gid"] = group_ids
        _p(f"total_groups={df['_gid'].nunique()}")
        for gid, grp in df.groupby("_gid"):
            texts = grp[content_col].astype(str).fillna("")
            q_text = next((normalize_cell(t) for t in texts if is_q(t)), "")
            a_text = next((normalize_cell(t) for t in texts if is_a(t)), "")
            _p(f"group={gid} -> Q='{q_text[:60]}', A='{a_text[:60]}'")
            if q_text and a_text:
                records.append({"question": q_text, "answer": a_text})
        df.drop(columns=["_gid"], errors="ignore", inplace=True)
    else:
        i = 0
        n = len(series)
        while i < n:
            cell = series.iloc[i]
            if is_q(cell):
                question = normalize_cell(cell)
                j = i + 1
                answer = ""
                while j < n:
                    nxt = series.iloc[j]
                    if is_a(nxt):
                        answer = normalize_cell(nxt)
                        break
                    if str(nxt).strip() == "":
                        j += 1
                        continue
                    if is_q(nxt):
                        break
                    j += 1
                _p(f"scan i={i} -> Q='{question[:60]}', A_found={bool(answer)}")
                if question and answer:
                    records.append({"question": question, "answer": answer})
                i = j if j > i else i + 1
            else:
                i += 1

    if not records:
        q_candidates = [s for s in series if bool(re.match(r"^\s*Q[\.\)\:]?\s+", str(s), re.IGNORECASE))]
        a_candidates = [s for s in series if bool(re.match(r"^\s*A[\.\)\:]?\s+", str(s), re.IGNORECASE))]
        _p(f"no records from primary path. q_candidates={len(q_candidates)}, a_candidates={len(a_candidates)}")
        for sample in q_candidates[:3]:
            _p(f"Q? sample: {sample}")
        for sample in a_candidates[:3]:
            _p(f"A? sample: {sample}")

        # Fallback 1: 셀 내 줄바꿈 분해 후 전열 스캔
        all_lines: List[str] = []
        for _, row in df.iterrows():
            for val in row.values.tolist():
                if pd.isna(val):
                    continue
                text = str(val)
                # 줄바꿈으로 분해
                for piece in re.split(r"[\r\n]+", text):
                    piece = piece.strip()
                    if piece:
                        all_lines.append(piece)
        _p(f"fallback1: flattened lines count={len(all_lines)}")

        i = 0
        n = len(all_lines)
        while i < n:
            cur = all_lines[i]
            if is_q(cur):
                q_text = normalize_cell(cur)
                j = i + 1
                a_text = ""
                while j < n:
                    nxt = all_lines[j]
                    if is_a(nxt):
                        a_text = normalize_cell(nxt)
                        break
                    if is_q(nxt):
                        break
                    j += 1
                _p(f"fallback1 i={i} -> Q='{q_text[:60]}', A_found={bool(a_text)}")
                if q_text and a_text:
                    records.append({"question": q_text, "answer": a_text})
                i = j if j > i else i + 1
            else:
                i += 1

        # Fallback 2: 여전히 없으면 전체 셀에서 Q/A를 순차 추출
        if not records:
            linear = [str(x).strip() for x in df.astype(str).values.ravel().tolist() if str(x).strip()]
            _p(f"fallback2: linear cells count={len(linear)}")
            i = 0
            n = len(linear)
            while i < n:
                cur = linear[i]
                if is_q(cur):
                    q_text = normalize_cell(cur)
                    j = i + 1
                    a_text = ""
                    while j < n:
                        nxt = linear[j]
                        if is_a(nxt):
                            a_text = normalize_cell(nxt)
                            break
                        if is_q(nxt):
                            break
                        j += 1
                    _p(f"fallback2 i={i} -> Q='{q_text[:60]}', A_found={bool(a_text)}")
                    if q_text and a_text:
                        records.append({"question": q_text, "answer": a_text})
                    i = j if j > i else i + 1
                else:
                    i += 1

    if not records:
        raise ValueError("Q/A 패턴을 찾지 못했습니다. 'Q.' 줄 다음에 'A.' 줄이 필요합니다.")

    return records

