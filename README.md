## Dialog System Technology Challenge 12 - Controllable Conversation Theme Detection track

See the task description at https://dstc12.dstc.community/tracks

## Getting started
Setting up environment and installing packages:
```
conda create -n dstc12 python=3.11
pip install -r requirements.txt
. ./set_paths.sh
```

## Getting familiar with the baseline code

Running theme detection
```
python scripts/run_theme_detection.py <dataset_file> <preferences_file> <result_file>
```

e.g. for Banking:

```
python scripts/run_theme_detection.py \
    dstc12-data/AppenBaking/all.jsonl \
    dstc12-data/AppenBanking/preference_pairs.json \
    appen_banking_predicted.jsonl
```

Running evaluation:

coming up

## Running LLM
Some parts of logic used in this baseline use an LLM being run locally:

* theme labeling in `run_theme_detection.py`
* evaluation of theme labels against the Theme Label Guideline - coming up

We use `lmsys/vicuna-13b-v1.5` by default which we tested on 4x Nvidia V100's (@16GB each). Please feel free to use the local setup or API access that works best for you. In case of any questions, please feel free to contact the organizers e.g. via Github issues.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.

