# 画像生成

画像生成プログラムのPyTorchを使った実装

# Contents

- DCGAN
- VAE
- CVAE (カテゴリごとに画像を生成するモデル)

# CVAEについて

CVAEはVAEにの入力にカテゴリの情報を付与して学習、推論を行うモデルです。[参照](https://qiita.com/kenchin110100/items/7ceb5b8e8b21c551d69a#conditional-variational-auto-encodercvae)

Encoder,Decoderの入力にそれぞれラベルを付与しています。

# Requirements
pytorch (ver. 3.0)
torchvision (ver. 0.2.0)
