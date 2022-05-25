import os

class infer_config():
    base_path="/translation"

    model_path=os.path.join(base_path, "m2m100_1.2B")
    task='text2text-generation'
    src_lang='en'
    tgt_lang='ko'
    split_data=8
    input_file=os.path.join(base_path, "translation_input.txt")
    output_file=os.path.join(base_path, "translation_output.txt")
