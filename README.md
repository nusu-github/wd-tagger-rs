# wd-tagger-rs

このプロジェクトは、SmilingWolfが作成したWaifuDiffusion Tagger (wd-tagger)のRust実装です。アニメスタイルのイラストを処理し、画像に含まれる要素を説明するタグを生成します。

## WaifuDiffusion Taggerについて

WaifuDiffusion Taggerは、アニメスタイルのイラストを分析し、画像内のさまざまな要素を説明するタグのリストを提供するツールです。これらのタグには、キャラクターの特徴、衣装アイテム、アートスタイルなどが含まれます。

オリジナルプロジェクト:

- リポジトリ: [SmilingWolf/SW-CV-ModelZoo](https://github.com/SmilingWolf/SW-CV-ModelZoo)
- Hugging Faceデモ: [SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger)

## 特徴

- パフォーマンス向上のためのRust実装
- 画像のバッチ処理
- マルチスレッド実行
- 一般的な分類とキャラクター分類の設定可能なしきい値
- CUDAデバイスのサポート（オプション）
- ユーザーフレンドリーなインターフェースによる進捗追跡

## 前提条件

- Rust（最新の安定版）
- CUDAツールキット（オプション）

## インストール

1. このリポジトリをクローンします：
   ```shell
   git clone https://github.com/yourusername/waifu-diffusion-tagger-rust.git
   cd waifu-diffusion-tagger-rust
   ```

2. プロジェクトをビルドします：
   ```shell
   cargo build --release
   ```

## 使用方法

以下のコマンドでツールを実行します：

```shell
wd-tagger-rs
```

### オプション

- `<INPUT_DIR>`: 処理するアニメスタイルの画像を含む入力ディレクトリ
- `<OUTPUT_DIR>`: 生成されたタグの出力ディレクトリ
- `-m, --model_name <MODEL_NAME>`: モデル名 （デフォルト：wd-swinv2-tagger-v3）
- `-d, --device-id <DEVICE_ID>`: GPUデバイスID（デフォルト：0）
- `-b, --batch-size <BATCH_SIZE>`: 処理のバッチサイズ（デフォルト：1）
- `--model-path <MODEL_PATH>`: ONNXモデルファイルへのパス（オリジナルのWaifuDiffusion Taggerモデルから変換したもの）
- `--csv-path <CSV_PATH>`: ラベル情報を含むCSVファイルへのパス
- `--general-threshold <GENERAL_THRESHOLD>`: 一般的な分類のしきい値（デフォルト：0.35）
- `--general-mcut-enabled`: 一般的な分類でMCUTを有効にする
- `--character-threshold <CHARACTER_THRESHOLD>`: キャラクター分類のしきい値（デフォルト：0.85）
- `--character-mcut-enabled`: キャラクター分類でMCUTを有効にする

### 例

```
wd-tagger-rs -m wd-swinv2-tagger-v3 --general-threshold 0.4 --character-threshold 0.8 --general-mcut-enabled /path/to/images /path/to/output
```

## 出力

このツールは入力ディレクトリ内のアニメスタイルの画像を処理し、出力ディレクトリにテキストファイルを生成します。各テキストファイルには、画像内で識別された要素に対応するカンマ区切りのタグが含まれています。

## ライセンス

- [Apache License 2.0](LICENSE)

## 謝辞

- [SW-CV-ModelZoo](https://github.com/SmilingWolf/SW-CV-ModelZoo)
- [SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger)
