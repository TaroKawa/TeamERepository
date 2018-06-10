## スクレイピング用コードを動かすためには

### 必要ライブラリなど
#### Python
* selenium
* pyvirtualdisplay

#### OS側
* xvfb
* xephyr

#### 参考
http://pyvirtualdisplay.readthedocs.io/en/latest/

### Usage
* TeamERepositoryをクローンしデータ保存用ディレクトリ作成

```bash
$ git clone https://github.com/TaroKawa/TeamERepository.git
$ mkdir TeamERepository/data
```

* コードのあるディレクトリに移動

```bash
$ cd TeamERepository/scraping/test
```

* google_img_query.pyをエディタで開く

例としてvimで開く場合は

```bash
$ vim google_img_query.py
```

* 最終行のクエリワードを書き換える

```Python
if __name__ == "__main__":
    query_on_google_img("ワンピース 赤")
```

の"ワンピース 赤"の部分を自分の検索したいワードに書き換えてください。
現状、ワンピースの次のワード(上の例では"赤")ごとにデータ保存用のディレクトリを作成しているので、検索ワードは"ワンピース 〇〇"のようにお願いします。

* google_img_query.pyを保存し、実行する。

```bash
$ python google_img_query.py
```

* dataディレクトリを確認

```bash
$ cd ../../data
$ ls
```

この結果で先ほどのクエリワードの二語目がディレクトリとして出来ていて、その中身が写真になっていれば成功です。
