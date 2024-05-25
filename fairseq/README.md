# Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/ysngki/XMoE.git
cd fairseq
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
# Data Preparation

## 1. Training Data

We use the **[mtdata](https://github.com/thammegowda/mtdata)** library to prepare the data we need. This library is officially designated by WMT.

First, download the data:

```bash
pip install mtdata==0.4.0
wget https://www.statmt.org/wmt23/mtdata/mtdata.recipes.wmt23-constrained.yml
for ri in wmt23-{enzh,zhen,ende,deen,enhe,heen,enja,jaen,enru,ruen,encs,csuk,enuk,uken}; do
  mtdata get-recipe -ri $ri -o $ri
done
```

The above code is copied from the [WMT23 website](https://www2.statmt.org/wmt23/mtdata/).

For each language pair, such as zhen (Chinese to English), there will be two files in `wmt23-zhen/`, which should be called `train.zh` and `train.en`. These two files have the same number of lines. Each line is a piece of training data. Each line in the two files corresponds one-to-one.

Before further cleaning the data, let's first prepare the test data and validation data.

## 2. Test Set and Validation Set

According to the official recommendation of WMT, past years' test data should be used as the validation set.

>To evaluate your system during development, we suggest using test sets from past WMT years...

First, use **mtdata** to download the data:

```bash
cd wmt23-zhen

# Validation set
mtdata get -l heb-eng --out test/ --merge --dev Statmt-newstest_ruen-20{16,17,18,19}-rus-eng

# Test set
mtdata get -l ces-eng --out test/ --merge --test Statmt-newstest_ceen-2022-ces-eng
```

As you can see, we use data from 18-20 as the validation set and data from 21 as the test set. If you are curious about how many years of data there are, you can use this command to check: `mtdata list -l zho-eng  | cut -f1`.

Both the validation set and the test set are saved in the `test/` folder, but their names are different. Both have two files. For the validation set, the names are `dev.zho` and `dev.eng`, with the suffix being the three-letter abbreviation of the language. The test set follows the same pattern.

For subsequent processing, please rename the files, changing the suffix to the two-letter abbreviation of the language. For example, change ```zho``` to ```zh```.

## 3. Preprocessing

SentencePiece is a more commonly used tokenizer nowadays. One of its advantages is that it does not require normalization or tokenization with moses before learning the tokenizer:

> `--input`: one-sentence-per-line **raw** corpus file. No need to run tokenizer, normalizer, or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.

Moreover, it can directly specify how many tokens the tokenizer will finally have.

### 3.1 File Preparation

Before starting, please organize the previously downloaded files like this:

```
|-orig
  --train.tgt
  --train.src
  |-test
    --test.tgt
    --test.src
    --dev.tgt
    --dev.src
```

Where `orig` represents the folder where you store the data, perhaps called `wmt23-jaen`, but others are also ok.

### 3.2 SentencePiece

The following bash file includes the process of learning the tokenizer and processing the training and test sets.

What you need to do: 1) Make sure you have cloned fairseq, then point `SCRIPTS` to the folder. 2) Check (and set) some variables in the first two blocks separated by #, setting them to your own.

Then, run it with one click, it's that simple!

**Note**: `character_coverage` is modifiable. For Chinese and Japanese, it can be set to 0.9995. For Latin languages, changing it to 1.0 is better.

(Referenced fairseq's [example](https://github.com/facebookresearch/fairseq/issues/1080))

```bash
############################################################
pip install sentencepiece

SCRIPTS=fairseq/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py


BPESIZE=32000
TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

TRAIN_SENTENCE_MAXNUM=20000000 # used for train tokenizer
############################################################

CHAR_COVER=1.0

src=ru
tgt=en
orig=wmt23-ruen

CORPORA=(
    "train"
)

OUTDIR=wmt23-${src}${tgt}-sentencepiece
prep=$OUTDIR

mkdir -p $prep

############################################################
TRAIN_FILES=$(for f in "${CORPORA[@]}"; do echo $orig/$f.${src}; echo $orig/$f.${tgt}; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$prep/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=$CHAR_COVER \
    --model_type=bpe \
    --input_sentence_size=$TRAIN_SENTENCE_MAXNUM \
    --shuffle_input_sentence=true

############################################################
# echo "encoding train/valid/test with learned BPE..."

echo "encoding train with learned BPE..."
python "$SPM_ENCODE" \
    --model "$prep/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $orig/train.${src} $orig/train.${tgt} \
    --outputs $prep/train.${src} $prep/train.${tgt} \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN

echo "encoding valid with learned BPE..."
python "$SPM_ENCODE" \
    --model "$prep/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $orig/test/dev.${src} $orig/test/dev.${tgt} \
    --outputs $prep/valid.${src} $prep/valid.${tgt}

echo "encoding test with learned BPE..."
python "$SPM_ENCODE" \
    --model "$prep/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $orig/test/test.${src} $orig/test/test.${tgt} \
    --outputs $prep/test.${src} $prep/test.${tgt}
```

## 4. Binary Files

Great, you can check the files in the output directory. If successful, there will be two files each for training, testing, and validation data. At this point, if you open the files, they will still be readable, but there will be symbols like `▁` or `@@`. It indicates that a word has been cut into one or more tokens.

Next, we need to convert these text data into binary files for fairseq. This is beneficial for faster training.

You'll notice that after processing the files with SentencePiece, you get two additional things: a model and a vocab.

The processed files, separated by spaces for all tokens, can be found corresponding to the vocab (with exceptions). So, next, we can convert these processed files into a sequence of numbers based on this vocab.

(This is referenced from [issue](https://github.com/facebookresearch/fairseq/issues/859#:~:text=To%20be%20more%20specific%20-%20if%20you%20use%20similar%20preprocess)).

First, convert this vocab into a format that fairseq can use:

```bash
# strip the first three special tokens and append fake counts for each vocabulary
tail -n +4 sentencepiece.bpe.vocab | cut -f1 | sed 's/$/ 100/g' > fairseq.vocab
```

Then start processing:

```bash
TEXT=wmt22-csen-sentencepiece
DICT=wmt22-csen-sentencepiece/fairseq.vocab
DEST=wmt22-csen-fairseq-sentencepiece

fairseq-preprocess --source-lang cs --target-lang en \
--joined-dictionary --tgtdict $DICT \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir $DEST --workers 10
```

# Training

Next, we use 2 GPUs, train for 10 epochs, and only save the checkpoint of the best epoch on the validation set.

## Dense model

```bash
CUDA_VISIBLE_DEVICES=0,1 fairseq-train /data/yuanhang/wmt23/wmt23-deen-fairseq-sentencepiece \
--arch transformer_wmt_en_de --share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
--warmup-updates 512 --dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--moe-loss-coeff 0.01 \
--max-tokens 16384 \
--update-freq 32 \
--max-source-positions 256 \
--max-target-positions 256 \
--max-epoch 10 \
--save-interval-updates 100 \
--keep-interval-updates 10 \
--no-epoch-checkpoints \
--amp \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--find-unused-parameters \
--log-interval 1 \
--wandb-project deen_sentencepiece \
--save-dir checkpoints/wmt23-deen-base/ |& tee logs/wmt23-deen-base.log
```

## MoE

```bash
CUDA_VISIBLE_DEVICES=2,3 fairseq-train /data/yuanhang/wmt23/wmt23-deen-fairseq-sentencepiece \
--arch transformer_moe_ep64_s2_nod_top2_cf2_ecf2 --share-decoder-input-output-embed \
--moe-type threshold \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
--warmup-updates 512 --dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--moe-loss-coeff 0.01 \
--max-tokens 16384 \
--update-freq 32 \
--max-source-positions 256 \
--max-target-positions 256 \
--max-epoch 10 \
--save-interval-updates 100 \
--keep-interval-updates 10 \
--no-epoch-checkpoints \
--amp \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--find-unused-parameters \
--log-interval 1 \
--wandb-project deen_sentencepiece \
--save-dir checkpoints/wmt23-deen-ep64_s2_nod_t0.9_cf2_ecf2/ |& tee logs/wmt23-deen-ep64_s2_nod_t0.9_cf2_ecf2.log
```



# Evaluation

```bash
DATASET=wmt22
LANGPAIR=uk-en
SRCLANG=uk
TGTLANG=en

SPM_ENCODE=scripts/spm_encode.py
DATABIN=/data/yuanhang/mtdata/wmt23-deen-fairseq-sentencepiece
SPM_MODEL=$DATABIN/sentencepiece.bpe.model
MODEL=

###################################################
### Evaluate directly
sacrebleu -t $DATASET -l $LANGPAIR --echo src \
| python $SPM_ENCODE --model $SPM_MODEL \
| CUDA_VISIBLE_DEVICES=3 fairseq-interactive $DATABIN --path $MODEL \
    -s $SRCLANG -t $TGTLANG \
    --beam 5 --remove-bpe=sentencepiece --buffer-size 1024 --max-tokens 16384 \
| grep ^H- | cut -f 3- \
| sacrebleu -t $DATASET -l $LANGPAIR -m bleu chrf --chrf-word-order 2

###################################################
### Save datasets before evaluting
sacrebleu -t wmt22 -l $LANGPAIR --echo src > src.$LANGPAIR
sacrebleu -t wmt22 -l $LANGPAIR --echo ref > ref.$LANGPAIR

sacrebleu -t wmt21 -l $LANGPAIR --echo src >> src.$LANGPAIR
sacrebleu -t wmt21 -l $LANGPAIR --echo ref >> ref.$LANGPAIR

cat src.$LANGPAIR \
| python $SPM_ENCODE --model $SPM_MODEL \
| CUDA_VISIBLE_DEVICES=2 fairseq-interactive $DATABIN --path $MODEL \
    -s $SRCLANG -t $TGTLANG \
    --beam 5 --remove-bpe=sentencepiece --buffer-size 1024 --max-tokens 16384 \
| grep ^H- | cut -f 3- \
| sacrebleu ref.$LANGPAIR -m bleu chrf --chrf-word-order 2
```

