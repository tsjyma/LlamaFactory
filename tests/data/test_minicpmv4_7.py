# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX, SUPPORTED_MODELS
from llamafactory.hparams import DataArguments, FinetuningArguments
from llamafactory.model.model_utils.visual import get_forbidden_modules


@pytest.fixture
def processor():
    processing = pytest.importorskip("transformers.models.minicpmv4_7.processing_minicpmv4_7")
    from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor
    from transformers.models.minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor

    extra = {
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
        "image_start_token": "<image>",
        "image_end_token": "</image>",
        "slice_start_token": "<slice>",
        "slice_end_token": "</slice>",
        "image_id_start_token": "<image_id>",
        "image_id_end_token": "</image_id>",
    }
    tokens = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", *extra.values()]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Read image. One cat. Two cats."],
        trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=tokens
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=tokens[2:],
        extra_special_tokens=extra,
    )
    return processing.MiniCPMV4_7Processor(
        image_processor=MiniCPMV4_6ImageProcessor(scale_resolution=56, max_slice_nums=1),
        video_processor=MiniCPMV4_6VideoProcessor(scale_resolution=56, max_slice_nums=1, do_sample_frames=False),
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_native_features_and_padding(processor, monkeypatch, mode, padding_side):
    monkeypatch.setenv("DOWNSAMPLE_MODE", mode)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    template = get_template_and_fix_tokenizer(
        tokenizer, DataArguments(template="minicpm_v_4_7", enable_thinking=False)
    )
    images = [Image.new("RGB", (56, 56))]
    features = []
    for answer in ["One cat.", "Two cats. Two cats."]:
        messages = [{"role": "user", "content": "<image>Read image."}, {"role": "assistant", "content": answer}]
        processed = template.mm_plugin.process_messages(messages, images, [], [], processor)
        prompt, target = template.encode_oneturn(tokenizer, processed)
        assert prompt[-len(template.get_thought_word_ids(tokenizer)) :] == template.get_thought_word_ids(tokenizer)
        assert tokenizer.decode(target) == answer + "<|im_end|>\n"
        native = processor(text="<|image_pad|>\nRead image.", images=images, downsample_mode=mode)
        native_prompt = tokenizer.decode(native["input_ids"][0])
        assert processed[0]["content"] == native_prompt
        features.append(
            dict(
                input_ids=prompt + target,
                attention_mask=[1] * (len(prompt) + len(target)),
                labels=[IGNORE_INDEX] * len(prompt) + target,
                images=images,
            )
        )

    model = SimpleNamespace(config=SimpleNamespace(model_type="minicpmv4_7"), get_rope_index=Mock())
    batch = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer, template=template, processor=processor, model=model, pad_to_multiple_of=8
    )(deepcopy(features))
    model.get_rope_index.assert_not_called()
    assert "position_ids" not in batch
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    assert batch["input_ids"].shape[-1] % 8 == 0
    assert torch.all(batch["mm_token_type_ids"][batch["attention_mask"] == 0] == 0)
    expected_types = processor.create_mm_token_type_ids(batch["input_ids"].tolist())
    assert batch["mm_token_type_ids"].tolist() == expected_types


def test_video_and_invalid_media(processor, monkeypatch):
    monkeypatch.setenv("DOWNSAMPLE_MODE", "4x")
    template = get_template_and_fix_tokenizer(processor.tokenizer, DataArguments(template="minicpm_v_4_7"))
    videos = [[Image.new("RGB", (56, 56)) for _ in range(2)]]
    messages = [{"role": "user", "content": "<video>Read image."}]
    result = template.mm_plugin.process_messages(messages, [], videos, [], processor)
    assert result[0]["content"].count("<|video_pad|>") > 0
    assert messages[0]["content"] == "<video>Read image."
    with pytest.raises(ValueError):
        template.mm_plugin.process_messages(messages, [], [], [], processor)
    with pytest.raises(ValueError):
        template.mm_plugin.get_mm_inputs([], [], [torch.zeros(1600)], [0], [0], [1], [[1]], processor)


def test_registered_models_and_freezing():
    assert "MiniCPM-V-4.7-1B" in SUPPORTED_MODELS
    assert "MiniCPM-V-4.7-35B-A3B" in SUPPORTED_MODELS
    args = FinetuningArguments(freeze_vision_tower=False, freeze_multi_modal_projector=True)
    assert get_forbidden_modules(SimpleNamespace(model_type="minicpmv4_7"), args) == {"model.merger"}


@pytest.mark.parametrize("text_type", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_zero3_merger_and_expert_leaves(monkeypatch, text_type):
    modeling = pytest.importorskip("transformers.models.minicpmv4_7.modeling_minicpmv4_7")
    from llamafactory.model.model_utils import moe

    monkeypatch.setattr(moe, "is_deepspeed_zero3_enabled", lambda: True)
    register = Mock()
    monkeypatch.setattr(moe, "_set_z3_leaf_modules", register)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="minicpmv4_7", text_config=SimpleNamespace(model_type=text_type)),
        model=SimpleNamespace(merger=torch.nn.Identity()),
    )
    moe.add_z3_leaf_module(model)
    assert register.call_args_list[0].args == (model, [modeling.MiniCPMV4_7Merger])
    assert register.call_count == (2 if text_type == "qwen3_5_moe_text" else 1)
    output = model.model.merger([torch.ones(2, 3), torch.zeros(1, 3)])
    assert len(output) == 1 and output[0].shape == (3, 3)
