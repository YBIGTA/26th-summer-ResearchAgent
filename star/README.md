# Chess verifier pipelines (`star/`)

이 폴더에는 Stockfish 기반 검증 루프가 포함되어 있습니다. `python -m star.cli`로 실행하면 새 JSONL 샘플이 생성됩니다.

## 준비
1. Stockfish 바이너리 경로를 준비합니다. (예: `/usr/games/stockfish`)
2. 의존성 설치: `pip install -r requirements.txt`

## 실행 예시
### Iterative SFT
```bash
python -m star.cli --mode sft_iter --engine_path /usr/games/stockfish \
  --fen "8/8/8/8/8/8/8/8 w - - 0 1" --iterations 1 --num_candidates 1 --out_dir /tmp/star_runs
```
- 생성 파일: `/tmp/star_runs/sft_iter_data.jsonl`
- 기본 FEN은 체스 시작 포지션이며 `--enable_rationale`로 rationale 파싱을 켤 수 있습니다.

### Iterative DPO
```bash
python -m star.cli --mode dpo_iter --engine_path /usr/games/stockfish \
  --fen "8/8/8/8/8/8/8/8 w - - 0 1" --iterations 1 --num_candidates 1 --out_dir /tmp/star_runs
```
- 생성 파일: `/tmp/star_runs/dpo_iter_data.jsonl`

### 테스트 스크립트
`STOCKFISH_PATH` 환경 변수를 설정하면 간단한 통합 테스트를 실행할 수 있습니다.
```bash
STOCKFISH_PATH=/usr/games/stockfish python -m unittest star.utils_test
```

생성된 JSONL은 `star/data/schema.py`의 데이터 구조를 따르며, `star/data/io.py`로 로드/저장할 수 있습니다.
