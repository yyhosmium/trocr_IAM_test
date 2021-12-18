import task
import deit
import deit_models
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms
from fairseq.scoring import BaseScorer, register_scorer
from nltk.metrics.distance import edit_distance
from fairseq.dataclass import FairseqDataclass
import fastwer

class CER2Scorer(BaseScorer):
    def __init__(self):
        self.refs = []
        self.preds = []

    def add_string(self, ref, pred):
        self.refs.append(ref)
        self.preds.append(pred)
    
    def score(self):
        return fastwer.score(self.preds, self.refs, char_level=True)

    def result_string(self) -> str:
        return f"CER2: {self.score():.2f}"
        
        
def init(model_path, beam=5):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={"beam": beam, "task": "text_recognition", "data": "", "fp16": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    generator = task.build_generator(
        model, cfg.generation, extra_gen_cls_kwargs={'lm_model': None, 'lm_weight': None}
    )

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):
    im = Image.open(img_path).convert('RGB').resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        'net_input': {"imgs": im},
    }

    return sample


def get_text(cfg, generator, model, sample, bpe):
    decoder_output = task.inference_step(generator, model, sample, prefix_tokens=None, constraints=None)
    decoder_output = decoder_output[0][0]       #top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )

    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str


if __name__ == '__main__':
    model_path = '/home/user/unilm/trocr/path/model/trocr-base-handwritten.pt'
    jpg_path = "/home/user/unilm/trocr/path/data/IAM/image/"
    data_gt_path = "/home/user/unilm/trocr/path/data/IAM/gt_test.txt"
    data_gt = open(data_gt_path,'r').readlines()
    beam = 5

    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)
    CER = CER2Scorer()
    for dg in data_gt:
        name_text = dg.split('	')
        jpg_name = name_text[0]
        text_label = name_text[1].split('\n')[0]
        sample = preprocess(jpg_path+jpg_name, img_transform)
        text_pred = get_text(cfg, generator, model, sample, bpe)
        CER.add_string(text_label, text_pred)
        
    print(CER.score())


