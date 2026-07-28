# リポジトリ全面改修レポート

## 改修の目的

旧構造では，論文上の実験番号や図の単位が Python パッケージ，設定クラス，
runner，artifact schema，plot recipe の境界になっていました。その結果，統合
CLI の内部で個別 runner を呼び分けるだけになり，新しい実験を追加するたびに
同種の実装が増える構造でした。

今回の改修では，研究の本体である

```text
有限入力
→ サロゲートダイナミクス
→ 観測
→ 特徴変換・リードアウト
→ 選択
→ 凍結
→ テスト評価
```

を一つの共通実行系として実装しました。

## 主な変更

- core package から論文名・実験番号に基づく namespace と条件分岐を削除
- 単一条件と sweep を同じ `StudyRunner` に統合
- 妥当性検証を予測 study から独立した `pol.validation` に分離
- target dataset を検証証明書に結び付いた content-addressed artifact として実装
- `n_ref`, `n_tar`, `n_sur`, `J`, `q` の所有責任を分離
- `n_tar < J` を正当な設定として明示的に検証
- 有限 `n_tar` 入力からのみサロゲート初期状態を構築し，高周波情報漏洩を禁止
- 主誤差を収束確認済み `n_ref` 上の連続場近似として評価し，`n_tar` 上の
  data-space error も別に保存
- `J × q` と `n_tar × n_sur` を別々の相図として実装
- PDE を共通 evolution interface で扱い，heat/Burgers/reaction-diffusion を登録
- PDE ダイナミクスを使わない `static_input` baseline も同じ Trial/Study 経路で実行
- 直接デコード，アフィン ridge，ランダム非線形特徴 + ridge を共通 readout として実装
- validation-only selection と test 評価の耐久境界を明示
- selection record と frozen plan を保存・hash 検証・read-back した後にのみ test を解禁
- artifact と study run に exact byte manifest と transactional publication を導入
- experiment 固有 plotter を long-form table に対する汎用 reporter に置換
- strict Pydantic configuration により未知キーを JSON path 付きで拒否
- smoke profile と main profile を分離

## 実行体系

```bash
python -m pol validate <validation-spec>
python -m pol data build <dataset-spec>
python -m pol run <study-spec>
python -m pol verify <artifact-or-run>
```

単一実行は sweep と別物ではなく，要素数 1 の study として扱われます。

## ディレクトリ上の原則

`pol/` は再利用可能な科学計算基盤だけを含みます。`studies/` は研究上の問いに
対応する条件の組合せだけを保存する宣言的な領域であり，core package から import
されません。

今後，新しいプロットや問いを追加する場合は，まず study JSON の axes，variants，
search，diagnostics，reporters の組合せで表現します。新しい Python runner を追加
するのは，新しい数値系・観測・readout・診断・可視化 primitive が本当に必要な
場合だけです。
