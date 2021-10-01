#!/bin/bash

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin
CURRENT_DIR="$(dirname "${BASH_SOURCE[0]}")"

MODEL=aot_test_model.pt
COMPILED_MODEL=aot_test_model.compiled.pt
COMPILED_CODE=aot_test_model.compiled.ll

test_aot_model_compiler() {
  python "$CURRENT_DIR"/aot_test_model.py
  exit_code=$?
  if [[ $exit_code != 0 ]]; then
    echo "Failed to save $MODEL"
    exit 1
  fi

  "$TORCH_BIN_DIR"/test_aot_model_compiler --model "$MODEL" --model_name=aot_test_model --model_version=v1 --input_dims="2,2,2"
  success=1
  if [ ! -f "$COMPILED_MODEL" ] || [ ! -f "$COMPILED_CODE" ]; then
    echo "AOT model compiler failed to generate $COMPILED_MODEL and $COMPILED_CODE"
    success=0
  fi

  [ -f $COMPILED_CODE ] && rm $COMPILED_CODE
  [ -f $COMPILED_MODEL ] && rm $COMPILED_MODEL
  rm "$MODEL"

  if [ !success ]; then
    exit 1
  fi
}

test_aot_model_compiler
