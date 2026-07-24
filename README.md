# AI異常検知：閾値と評価指標の説明アプリ

AI外観検査（良品学習型の異常検知）における **異常スコア・閾値・評価指標** を、
統計や機械学習に詳しくない製造現場のエンジニアにも直感的に理解してもらうための
教育用 Streamlit アプリです。

同じ OK / NG のスコア分布に対して閾値を動かすと、混同行列・各指標・ROC/PR/F1 曲線が
**すべて連動して変化する** ことを、一つの画面で見せることを目的にしています。

> ※ 表示されるデータはすべてアプリ内で生成する疑似データです。実案件のデータは含みません。

## 画面構成

| タブ | 内容 |
|------|------|
| ① 閾値とは何か | OK/NG の異常スコア分布・散布図・閾値スライダー |
| ② 混同行列と評価指標 | 混同行列（TN/FP/FN/TP）と Recall / Specificity / Precision / Accuracy / F1 |
| ③ ROC / PR / F1 曲線 | 3 つの評価曲線と、閾値の決め方（F1最大・見逃しゼロ優先・過検知率上限・コスト最小化・手動） |

判定規則は全体を通して **「異常スコア ≥ 閾値 → NG 判定」** で統一しています。

## ローカルでの実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。

## 技術構成

- **UI**: Streamlit + Plotly
- **計算**: NumPy / SciPy / scikit-learn（ROC・PR・AUC）
- `core/` … Streamlit 非依存の計算ロジック（分布生成・指標・曲線・閾値決定）
- `app.py` … UI 層

## デプロイ

[Streamlit Community Cloud](https://streamlit.io/cloud) から、このリポジトリの
`app.py` をメインファイルに指定して公開できます。GPU 不要です。

---

© TOMOMI RESEARCH Inc.
