#!/bin/bash

TASK=NER #summarization #SentimentAnalysis #reasoning #OpenDomainQA #NLI #NER #dialogue #ClosedDomainQA
POST_EDIT_FLAG="" #"--post_edit" # empty string for model without editing
EVAL_NAME="base-testing"

BASE_MODEL="gpt2-xl"
EDITED_WEIGHTS_PATH="." # this is considered only when post_edit flag is enabled

python test-task-exec.py $POST_EDIT_FLAG "--task" $TASK "--eval_name" $EVAL_NAME "--base_model" $BASE_MODEL "--edited_weights_path" $EDITED_WEIGHTS_PATH
