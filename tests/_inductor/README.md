# test_models_ops.py

test operations used in models with torch.compile

## Overview
This repository provides a test framework for validating models with `torch.compile`.  
You can run all tests, selectively skip tests, or report detailed information about skipped and failed cases.



## How to run tests

### Run tests by default

executes all pytest-style files except those listed in `models/skip_files.yaml`

```
python test_models_ops.py
```


### Run tests with detailed reporting

show additional information about their skip, failed, and error resons

```
python test_models_ops.py --report sfE
```


### Skip file format `models/skip_files.yaml`

'''
skip:
  reason1:
    -- modes/gpt-oss/file1.py
    -- modes/gpt-oss/file2.py
  reason2:
    -- modes/gpt-oss/file2.py

xfail:
  reason3:
    -- modes/granite4-h/file11.py
'''

### Environment variables

* `TEST_MODELS_OPS_IGNORE_SKIP_FILES=1` to run all tests, including those listed in `mode/skip_files.yamli`
* `TEST_MODELS_OPS_ONLY_SKIP_FILES=1` to run only the tests listed in 'models/skip_files.yaml`
