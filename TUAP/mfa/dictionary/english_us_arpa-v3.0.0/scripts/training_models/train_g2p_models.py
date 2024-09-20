import os.path
import re

from montreal_forced_aligner.command_line.mfa import mfa_cli
from montreal_forced_aligner.config import TEMPORARY_DIRECTORY

MODEL_VERSION = '3.0.0'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dictionary_dir = os.path.join(root_dir, 'dictionary', 'training')
output_dir = os.path.join(root_dir, 'g2p', 'staging')
temp_dir = TEMPORARY_DIRECTORY
os.makedirs(output_dir, exist_ok=True)

class DefaultArgs:
    def __init__(self, dictionary_path, output_model_path, temporary_directory):
        self.dictionary_path = dictionary_path
        self.output_model_path = output_model_path
        self.temporary_directory = temporary_directory
        self.config_path = None
        self.evaluation_mode = True
        self.num_jobs = 10
        self.debug = True
        self.clean = True


lang_codes = ['czech', 'russian',
              'french', 'german',
              'portuguese_brazil', 'portuguese_portugal',
              'spanish_spain', 'spanish_latin_america',
              'swedish',
              'thai',
              'turkish',
              'english_us','english_us_arpa','english_uk','english_nigeria',
              'korean_jamo', 'korean',
              'hausa', 'swahili',
              'vietnamese_hanoi', 'vietnamese_hue', 'vietnamese_ho_chi_minh_city',
               'ukrainian', 'polish', 'croatian', 'bulgarian',
              'japanese',
              'japanese_katakana',
                #'mandarin_china', 'mandarin_erhua', 'mandarin_taiwan'
              'tamil',
              'hindi',
              'urdu',

              ]

lang_codes = [
              'mandarin_china', 'mandarin_taiwan',
              'mandarin_china_pinyin', 'mandarin_taiwan_pinyin'
              ]


def get_error_rates(lang):
    train_temp_dir = os.path.join(temp_dir, f'{lang}_mfa')
    log_file = os.path.join(train_temp_dir, f'{lang}_mfa.log')
    print("Parsing:", log_file)
    if not os.path.exists(log_file):
        return 1, 1
    wer_pattern = re.compile(r'WER:\s+([\d.]+)$')
    ler_pattern = re.compile(r'LER:\s+([\d.]+)$')
    wer, ler = None, None
    with open(log_file, 'r', encoding='utf8') as f:
        for line in f:
            m = wer_pattern.search(line)
            if m:
                wer = m.group(0)
            m = ler_pattern.search(line)
            if m:
                ler = m.group(0)
    return wer, ler

error_metrics = {}

if __name__ == '__main__':

    for lang in lang_codes:
        print(lang)
        if lang.endswith('arpa'):
            dictionary_path = os.path.join(dictionary_dir, lang + '.dict')
            model_path = os.path.join(output_dir, lang + '.zip')
        else:
            dictionary_path = os.path.join(dictionary_dir, lang + '_mfa.dict')
            model_path = os.path.join(output_dir, lang + '_mfa.zip')
        print(model_path)
        if os.path.exists(model_path):
            error_metrics[lang] = get_error_rates(lang)
            if lang == 'korean':
                error_metrics[lang + '_jamo'] = get_error_rates(lang + '_jamo')
            continue
        unknown= []
        if not os.path.exists(dictionary_path):
            continue

        if lang == 'korean':
            import jamo
            jamo_dict_path = os.path.join(dictionary_dir, lang + '_jamo_mfa.dict')
            jamo_model_path = os.path.join(output_dir, lang + '_jamo_mfa.zip')
            with open(dictionary_path, 'r', encoding='utf8') as inf, open(jamo_dict_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    word, pron = line.split('\t')
                    if re.search(r'[a-zA-Z]+', word):
                        continue
                    jamoed_word = jamo.h2j(word)
                    outf.write(f"{jamoed_word}\t{pron}")
            command = ['train_g2p',
                       jamo_dict_path,
                       jamo_model_path,
                       '--clean',
                       '-j', '10',
                       '--use_mp',
                       '--evaluate',
                       '--num_pronunciations', '2',
                       '--phonetisaurus',
                       '--model_version', MODEL_VERSION,
                       ]
            if lang in {'mandarin_china', 'mandarin_taiwan'}:
                command += ['--phone_order', '4']
            mfa_cli(command, standalone_mode=False)

            error_metrics[lang + '_jamo'] = get_error_rates(lang + '_jamo')
        command = ['train_g2p',
                   dictionary_path,
                   model_path,
                   '--clean',
                   '-j', '10',
                   '--use_mp',
                   '--evaluate',
                   '--num_pronunciations', '2',
                   '--phonetisaurus',
                   '--model_version', MODEL_VERSION,
                   ]
        mfa_cli(command, standalone_mode=False)

        error_metrics[lang] = get_error_rates(lang)

    for k, v in error_metrics.items():
        print(f"{k}: {v[0]}% WER, {v[1]}% LER")
