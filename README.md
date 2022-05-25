# Translation (M2M100)

- M2M100를 이용한 다국어 번역 서비스입니다.
- 'Huggingface Transformers' 라이브러리를 이용하여 구현

## Dependencies

- torch==1.8.0
- transformers==4.10.0
- sentencepiece==0.1.95

## How to use M2M100 on Huggingface Transformers Library

- 기존의 M2M100 모델을 trasformers 라이브러리에서 곧바로 사용할 수 있도록 맞췄습니다.
- transformers 라이브러리에서 파이프라인 기능을 import 합니다.

``` python

from transformers import pipeline

# 번역 모델 불러오기 & 번역 파이프라인 설정
pipe=pipeline(task='text2text-generation', model=model_dir, device=0)

# src_lang(소스 언어) 및 tgt_lang(타겟 언어) 지정
# 언어 코드 정보는 lang_code.txt 파일에 있음

pipe.tokenizer.src_lang= 'en'
pipe.tokenizer.tgt_lang= 'ko'

# Input text for translation

text = " This is an artifical intelligence translator. "

# Output
pipe(text, forced_bos_token_id=pipe.tokenizer.get_lang_id(lang='ko'))

# Result
--> '이것은 인공지능 번역기입니다.'

```