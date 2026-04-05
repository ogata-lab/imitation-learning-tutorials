# Imitation Learning Tutorials

模倣学習（Imitation Learning）の基礎から最新手法までを体系的に学ぶチュートリアルシリーズです。PyTorch の基礎から始まり、RNN・Attention・拡散モデル・Flow Matching・ACT といった手法を、Robomimic データセットを用いて段階的に学習します。

## チュートリアル一覧

### 基礎編（第0回〜第2回）

| # | タイトル | 内容 |
|---|---------|------|
| [00](00_pytorch_basics.ipynb) | PyTorch の基礎 | テンソル操作、nn.Module、自動微分、損失関数、オプティマイザ、学習ループ |
| [01](01_preprocessing.ipynb) | 前処理の理解 | Min-Max 正規化、テンソル変換、画像の逆正規化 |
| [02](02_layers.ipynb) | PyTorch の Layer 理解 | BatchNorm、LayerNorm、SpatialSoftmax、Transformer（Self-Attention） |

### 表現学習・データセット（第3回〜第4回）

| # | タイトル | 内容 |
|---|---------|------|
| [03](03_cae.ipynb) | 畳み込みオートエンコーダ（CAE） | Encoder-Decoder、Conv2d/ConvTranspose2d、画像圧縮・復元、U-Net |
| [04](04_robomimic_dataset.ipynb) | Robomimic データセットの理解 | Robomimic / LeRobot フォーマット、Dataset クラスの実装 |

### 時系列モデル（第5回〜第10回）

| # | タイトル | 内容 |
|---|---------|------|
| [05](05_rnn.ipynb) | RNN を用いた時系列予測学習 | RNN/LSTM、BPTT、ゲート機構、時系列予測 |
| [06](06_mamba.ipynb) | Mamba を用いた時系列学習 | 状態空間モデル（SSM）、Selective SSM、O(N) 計算量 |
| [07](07_cnnrnn.ipynb) | CNNRNN | CNN+RNN によるマルチモーダル学習（画像＋関節角度） |
| [08](08_sarnn.ipynb) | SARNN | Spatial Attention RNN、キーポイント抽出、注意の可視化 |
| [09](09_hsarnn.ipynb) | HSARNN | 階層型 Spatial Attention RNN、マルチストリーム統合 |
| [10](10_stochastic_rnn.ipynb) | Stochastic RNN | 確率的予測、不確実性モデリング、潜在変数モデル |

### 生成モデルによる行動生成（第11回〜第13回）

| # | タイトル | 内容 |
|---|---------|------|
| [11](11_diffusion_policy.ipynb) | Diffusion Policy | DDPM/DDIM、ノイズ除去による行動生成、1D U-Net |
| [12](12_flow_matching.ipynb) | Flow Matching | Continuous Normalizing Flows、直線パス学習、効率的な行動生成 |
| [13](13_act.ipynb) | ACT | Action Chunking with Transformers、CVAE、Transformer Encoder/Decoder |

## 学習の流れ

```
基礎 (00-02) → 表現学習 (03-04) → 時系列モデル (05-10) → 生成モデル (11-13)
```

第0回〜第4回で PyTorch とデータの扱い方を学んだ後、第5回以降では Robomimic データセットを用いて、決定論的な時系列モデルから確率的モデル、さらに最新の生成モデルベースの手法へと段階的にステップアップしていきます。

## 主な使用ライブラリ

- PyTorch (`torch`, `torch.nn`, `torch.optim`)
- NumPy
- Matplotlib
- torchvision
