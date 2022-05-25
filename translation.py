from ast import parse
import os
import time

import torch
import logging
import argparse

from setproctitle import setproctitle

from utils import list_chunk, get_device, init_logger
from config import infer_config
from transformers import pipeline

setproctitle("Translation")
logger=logging.getLogger(__name__)

''' Translation (M2M-100 Model) '''

class text2text_generation(object):
    def __init__(self, args):
        try:
            self.args=args
            self.model_path=args.model_path
            self.task=args.task
            self.pipe=pipeline(task=self.task, model=self.model_path, device=0)

        except:
            raise Exception("********************* Failed to load pipeline *********************")

    def inference(self):
        # sample input file for inference

        text=[]

        with open(infer_config.input_file, "r", encoding='utf-8') as f:
            for line in f:
                text.append(line.strip())
            text=list(filter(None, text))

        output=[]
        self.pipe.tokenizer.src_lang=args.src_lang

        # inference time check
        begin=time.time()
        # text=args.input_text.strip().split('\n')
        # text=list(filter(None, text))

        if len(text) <= 8:
            result=self.pipe(text, forced_bos_token_id=self.pipe.tokenizer.get_lang_id(lang=args.tgt_lang))
            for outputs in result:
                output.append(outputs['generated_text'])
            torch.cuda.empty_cache()

        else:
            text=list_chunk(text, len(text) // (len(text)//args.split_data))
            for split_outputs in text:
                result = self.pipe(split_outputs, forced_bos_token_id=self.pipe.tokenizer.get_lang_id(lang=args.tgt_lang))
                for outputs in result:
                    output.append(outputs['generate_text'])
                    torch.cuda.empty_cache()
            
        outputs = "\n".join(output).strip()

        end=time.time()
        times=round((end-begin), 2)

        logger.info(output)
        logger.info(times)

        # Write to output file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("{0".format(output))

        return output

if __name__ == "__main__":
    init_logger()

    parser=argparse.ArgumentParser()

    parser.add_argument("--model_path", default=infer_config.model_path, type=str, help="model dir path")
    parser.add_argument("--task", default=infer_config.task, type=str, help="task for transformers")
    parser.add_argument("--src_lang", default=infer_config.src_lang, type=str, help="Select Source Language")
    parser.add_argument("--tgt_lang", default=infer_config.tgt_lang, type=str, help="Select Target Language")
    parser.add_argument("--split_data", default=infer_config.split_data, type=int, help="text split")
    parser.add_argument("--input_file", default=infer_config.input_file, type=str, help="Input Text")
    parser.add_argument("--output_file", default=infer_config.output_file, type=str, help="Save the translation text")

    args=parser.parse_args()

    device=get_device()

    text2text_generation=text2text_generation(args)
    text2text_generation.inference()