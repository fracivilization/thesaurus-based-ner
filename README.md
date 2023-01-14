# two-stage-distant-ner
Two Stage Distant NER (Using NP Chunker)

## Preprocess for UMLS
1. Please Dwonload 2021AA-full from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html"
2. unzip the folder and unzip the mmsys system
3. Run Metamorphosys
    ```sh
    ./run_linux.sh # or ./run_mc.sh or run.bat
    ```
    1. Press "Install UMLS"
    2. Set Install Settings
       1. Source: data/2021AA-full
       2. Destination: data
    3. Select New configuration
    4. License Agreement Notice: Accept
    5. Select Default Subset Configuration: Active Subset
    6. Source List>Select sources to INCLUDE in subset
       1.  Select all
    7. Done>Begin Subset> Would you like to save the changes?: No
    8.  MetamorphoSys Subset Log>OK

# Makefileの読み方
本研究では前処理の過程が複雑なのでMakefileにタスク間の依存関係を記載している。
基本的なターゲットはプロジェクト直下のMakefile内に記載している。
そこから ./makefiles/\_\_init\_\_.mk を介してロードしている、が、これはダミーファイルで以下のように分担している

- makefiles
    - \_\_init\_\_.mk: makefilies以下をロードする用のダミーファイル
    - dependencies.mk: タスク間の依存関係を記載
- variables
    - \_\_init\_\_.mk: variables以下をロードするためのダミーファイル
    - constraints.mk: 定数を記載している
    - arguments.mk: 可変な引数を記載している
        - MSC_ARGS: シングルクラススパン分類器用の引数
        - MSMLC_ARGS: マルチラベルスパン分類器用の引数
    - base_commands.mk: タスク種別ごとにタスクのサブクラスによらない共通部分を記載
        - 例えばGoldデータと疑似データによらないマルチラベルスパン分類の共通コマンドなど
    - targets.mk: 生成対象となるファイル一覧を記載している