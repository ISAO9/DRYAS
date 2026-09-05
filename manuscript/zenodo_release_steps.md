# GitHub → Zenodo で DOI を取る手順（DRYAS）

## 0. 事前に決めること
- GitHub のリポジトリ名: `DRYAS`（公開）。README・LICENSE（MIT）・CITATION.cff・.zenodo.json は同梱済み。
- `.gitignore` で除外: `.venv/`、`data/raw/`（NOAA から 001 が再取得）、`manuscript/*.docx`（投稿版は別管理）。
- `data/*.parquet`（派生表）と `PDF/` は**含める**（論文の数字を再現できる状態が Zenodo の価値）。合計 20 MB 程度なら問題なし。

## 1. リポジトリを作って push
```bash
cd DRYAS
git init
git add .
git commit -m "DRYAS v1.0.0: detection limits for far-field earthquake signals (scripts 001-010)"
git branch -M main
git remote add origin https://github.com/ISAO9/DRYAS.git
git push -u origin main
```
`CITATION.cff` の `repository-code` と `.zenodo.json` の内容を自分のユーザー名に直してから commit。

## 2. Zenodo と GitHub を連携（初回のみ）
1. https://zenodo.org にログイン（GitHub アカウントで可）。
2. 右上メニュー → **GitHub** → 連携を承認。
3. リポジトリ一覧で `DRYAS` のスイッチを **ON**。

## 3. GitHub でリリースを作る（これが DOI のトリガー）
1. GitHub の `DRYAS` → **Releases** → **Create a new release**。
2. Tag: `v1.0.0`、Title: `DRYAS v1.0.0 (manuscript submission)`、説明に「Code and derived data for: Detection limits for far-field earthquake signals in tree-ring networks (Dendrochronologia, submitted)」。
3. **Publish release**。数分で Zenodo 側にレコードが自動作成され、`.zenodo.json` のメタデータが使われる。

## 4. DOI を原稿に入れる
- Zenodo のレコードページに **バージョン DOI**（例 10.5281/zenodo.1234567）と **Concept DOI**（全バージョン共通、末尾の番号が 1 小さい）が出る。論文には **Concept DOI** を使う（改訂で v1.0.1 を出しても DOI が変わらない）。
- `manuscript/DRYAS_manuscript_v6.md` の Data availability の `10.5281/zenodo.【record】` と `github.com/【user】/DRYAS` を置換 → `uv run python src/008_manuscript_tables.py && uv run python src/009_build_docx.py`。

## 5. 査読対応でコードを直したとき
`git tag v1.0.1 && git push --tags` → GitHub で Release を作る → Zenodo に新バージョンが自動追加（Concept DOI は不変）。

## 6. EarthArXiv プレプリント（投稿と同時に出す場合）
- https://eartharxiv.org → Submit → PDF（009 の docx を PDF 化）→ ライセンス CC BY 4.0 → 「Under review at Dendrochronologia」を記載。Zenodo DOI を "Related identifiers" に入れる。
