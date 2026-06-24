"""
BERT-BASE-CHINESE 保险意图分类器 (LoRA 微调)

架构: bert-base-chinese → LoRA (r=8) → Dropout → Linear(768 → 9)
9分类: 闲聊寒暄/条款解读/保单查询/理赔咨询/产品对比/保费试算/退保咨询/投诉建议/out_of_domain

功能:
  - LoRA 微调: 仅训练 ~0.3M 参数 (全量 110M 的 0.3%)，单卡秒级训练
  - 单例模式: get_bert_classifier()
  - 延迟加载: 首次 predict() 时才加载模型
  - 温度校准: 训练后用验证集搜索最优温度 T
  - 批量预测: predict_batch() 10x 快于串行
  - LLM 扩增数据: generate_training_data() 用 DeepSeek 生成变体
  - merge_and_save: 合并 LoRA 到基座导出完整模型 (推理部署)

使用:
  python -m rag_qa.core.bert_intent_classifier --predict "肺炎住院能赔吗"
  python -m rag_qa.core.bert_intent_classifier --train --data rag_qa/data/seed_data.json --output models/bert_intent
"""

import json
import os
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from base.logger import logger
from config.settings import settings


INTENT_LABELS = [
    "闲聊寒暄",
    "条款解读",
    "保单查询",
    "理赔咨询",
    "产品对比",
    "保费试算",
    "退保咨询",
    "投诉建议",
    "out_of_domain",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(INTENT_LABELS)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BERTIntentClassifier:
    """BERT-BASE-CHINESE 意图分类器"""

    # ── LoRA 超参数 ─────────────────────────
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["query", "value"]

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.models.bert_classifier
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._temperature = 1.0
        self._lock = threading.Lock()
        self._is_lora = False  # True = PeftModel wrapper, False = merged/raw model

    # ── 懒加载 ──────────────────────────────

    def load(self, model_dir: str = None):
        """延迟加载模型，首次 predict/train 时自动触发

        加载优先级:
          1. 已合并的完整模型 (config.json 存在，无 adapter_config.json)
          2. LoRA adapter (adapter_config.json 存在) → 加载基座+adapter
          3. 预训练基座 (从 HuggingFace 或本地) → 创建新 LoRA wrapper
        """
        if self._loaded:
            return

        path = model_dir or self.model_path

        try:
            adapter_config = os.path.join(path, "adapter_config.json") if os.path.isdir(path) else None
            full_config = os.path.join(path, "config.json") if os.path.isdir(path) else None

            # ── Case 1: 已合并的完整微调模型 ──
            if full_config and os.path.exists(full_config) and not (adapter_config and os.path.exists(adapter_config)):
                logger.info(f"[BERT] 从完整模型加载: {path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    path, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self._is_lora = False

            # ── Case 2: LoRA adapter (保存的 adapter) ──
            elif adapter_config and os.path.exists(adapter_config):
                logger.info(f"[BERT] 从 LoRA adapter 加载: {path}")
                base_path = settings.models.bert_classifier  # 基座路径
                # 尝试从同目录找基座，否则用默认路径
                base_config = os.path.join(path, "base_model_config.json")
                if os.path.exists(base_config):
                    with open(base_config) as f:
                        base_path = json.load(f).get("base_model", base_path)

                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_path, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
                )
                self.model = PeftModel.from_pretrained(base_model, path)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self._is_lora = True

            # ── Case 3: 预训练基座 → 创建 LoRA wrapper ──
            else:
                logger.info(f"[BERT] 从基座加载并创建 LoRA: {path}")
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    path, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
                )
                lora_config = LoraConfig(
                    r=self.LORA_R,
                    lora_alpha=self.LORA_ALPHA,
                    lora_dropout=self.LORA_DROPOUT,
                    target_modules=self.LORA_TARGET_MODULES,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                )
                self.model = get_peft_model(base_model, lora_config)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self._is_lora = True

            # ── 加载温度参数 ──
            temp_path = os.path.join(path, "temperature.json") if os.path.isdir(path) else None
            if temp_path and os.path.exists(temp_path):
                with open(temp_path) as f:
                    self._temperature = json.load(f).get("temperature", 1.0)

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"[BERT] 加载完成 device={self.device} temp={self._temperature:.2f} "
                f"lora={self._is_lora} trainable={trainable}/{total} ({100*trainable/total:.1f}%)"
            )
        except Exception as e:
            logger.error(f"[BERT] 模型加载失败: {e}")
            raise

    # ── 预测 ────────────────────────────────

    def predict(self, query: str) -> dict:
        """单条预测 → {"intent": "理赔咨询", "confidence": 0.92}"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.load()

        inputs = self.tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / self._temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "intent": ID2LABEL[pred_idx],
            "confidence": round(confidence, 4),
            "all_probs": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }

    def predict_batch(self, queries: list[str], batch_size: int = 32) -> list[dict]:
        """批量预测，比逐条调用快 10x"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.load()

        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits / self._temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, prob in enumerate(probs):
                pred_idx = int(np.argmax(prob))
                results.append(
                    {
                        "intent": ID2LABEL[pred_idx],
                        "confidence": round(float(prob[pred_idx]), 4),
                        "all_probs": {
                            ID2LABEL[k]: round(float(p), 4) for k, p in enumerate(prob)
                        },
                    }
                )

        return results

    # ── 训练 ────────────────────────────────

    def train(
        self,
        texts: list[str],
        labels: list[str],
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        eval_split: float = 0.1,
        save_path: str = None,
    ):
        """LoRA 微调 BERT 分类器（仅训练 adapter 参数）

        Args:
            learning_rate: LoRA 推荐 1e-4 ~ 5e-4 (高于全量微调的 2e-5)
        """
        if not self._loaded:
            self.load()

        # 确保模型在 LoRA 模式下
        if not self._is_lora:
            logger.info("[BERT] 模型非 LoRA，重新包装...")
            lora_config = LoraConfig(
                r=self.LORA_R, lora_alpha=self.LORA_ALPHA,
                lora_dropout=self.LORA_DROPOUT,
                target_modules=self.LORA_TARGET_MODULES,
                bias="none", task_type=TaskType.SEQ_CLS,
            )
            self.model = get_peft_model(self.model, lora_config)
            self._is_lora = True

        # 数据划分
        from sklearn.model_selection import train_test_split

        if eval_split > 0:
            train_texts, eval_texts, train_labels, eval_labels = train_test_split(
                texts, labels, test_size=eval_split, stratify=labels, random_state=42
            )
        else:
            train_texts, train_labels = texts, labels
            eval_texts, eval_labels = [], []

        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        eval_dataset = (
            IntentDataset(eval_texts, eval_labels, self.tokenizer)
            if eval_texts else None
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # LoRA: 仅优化可训练参数 (~0.3M)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        self.model.train()
        self.model.to(self.device)

        trainable_count = sum(p.numel() for p in trainable_params)
        logger.info(
            f"[BERT] LoRA训练 epochs={epochs} samples={len(train_texts)} "
            f"batch={batch_size} lr={learning_rate} trainable={trainable_count}"
        )

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            eval_acc = 0
            if eval_dataset:
                eval_acc = self._evaluate_dataset(eval_dataset)
                logger.info(
                    f"[BERT] Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} eval_acc={eval_acc:.3f}"
                )
            else:
                logger.info(f"[BERT] Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

        self.model.eval()

        # 温度校准
        if eval_dataset:
            self._temperature = self._calibrate_temperature(eval_dataset)
            logger.info(f"[BERT] 温度校准: T={self._temperature:.3f}")

        # 保存 LoRA adapter
        if save_path:
            self._save(save_path)
            logger.info(f"[BERT] LoRA adapter 已保存到 {save_path}")

    def _evaluate_dataset(self, dataset: IntentDataset) -> float:
        loader = DataLoader(dataset, batch_size=32)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0

    def _calibrate_temperature(self, dataset: IntentDataset) -> float:
        loader = DataLoader(dataset, batch_size=32)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels)
        logits = torch.cat(all_logits, dim=0).cpu()
        labels = torch.cat(all_labels, dim=0).cpu()

        best_temp, best_nll = 1.0, float("inf")
        for t in np.logspace(-0.5, 0.5, 20):
            probs = F.softmax(logits / t, dim=1)
            nll = F.nll_loss(torch.log(probs + 1e-8), labels).item()
            if nll < best_nll:
                best_nll, best_temp = nll, t
        return best_temp

    def _save(self, save_path: str):
        """保存 LoRA adapter + tokenizer + 温度参数"""
        os.makedirs(save_path, exist_ok=True)
        if self._is_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # 记录基座路径
        with open(os.path.join(save_path, "base_model_config.json"), "w") as f:
            json.dump({"base_model": settings.models.bert_classifier}, f)
        with open(os.path.join(save_path, "temperature.json"), "w") as f:
            json.dump({"temperature": self._temperature}, f)

    def merge_and_save(self, save_path: str):
        """合并 LoRA adapter 到基座模型并保存为完整模型

        用于推理部署: 合并后无需 peft 依赖，直接 AutoModel 加载。
        """
        if not self._is_lora:
            logger.info("[BERT] 模型已是完整模型，直接保存")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            return

        logger.info("[BERT] 合并 LoRA → 完整模型...")
        merged = self.model.merge_and_unload()
        merged.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # 删除旧 adapter 文件，避免 load() 误判为 LoRA adapter
        for f in ["adapter_config.json", "adapter_model.safetensors",
                   "adapter_model.bin", "base_model_config.json"]:
            fp = os.path.join(save_path, f)
            if os.path.exists(fp):
                os.remove(fp)
        with open(os.path.join(save_path, "temperature.json"), "w") as f:
            json.dump({"temperature": self._temperature}, f)
        logger.info(f"[BERT] 合并模型已保存到 {save_path}")

    @classmethod
    def from_pretrained(cls, path: str) -> "BERTIntentClassifier":
        """从本地模型加载（支持完整模型或 LoRA adapter）"""
        instance = cls(model_path=path)
        instance.load(model_dir=path)
        return instance

    # ── LLM 数据扩增 ────────────────────────

    @staticmethod
    def generate_training_data(seed_path: str, output_path: str, num_variants: int = 5):
        """用 LLM 基于种子数据生成变体训练数据"""
        from base.llm_client import get_llm_client

        with open(seed_path) as f:
            seed_data = json.load(f)

        llm = get_llm_client()
        expanded = list(seed_data)  # 保留原始种子

        for item in seed_data:
            prompt = f"""你是一个数据增强助手。请为以下保险客服查询生成 {num_variants} 个语义相似但表述不同的变体。

原始查询: {item["text"]}
意图标签: {item["label"]}

要求:
1. 保持相同的意图
2. 变换句式、用词、语气
3. 模拟真实用户的口语化提问
4. 只返回 JSON 数组，格式: ["变体1", "变体2", ...]
"""
            try:
                response = llm.chat_json(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1024,
                )
                variants = (
                    response
                    if isinstance(response, list)
                    else response.get("variants", [])
                )
                for v in variants[:num_variants]:
                    expanded.append({"text": v, "label": item["label"]})
                logger.info(
                    f"[BERT] 数据扩增: '{item['text'][:30]}' → +{len(variants)} 变体"
                )
            except Exception as e:
                logger.warning(f"[BERT] LLM 扩增失败: {e}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[BERT] 数据扩增完成: {len(seed_data)} → {len(expanded)} 条 → {output_path}"
        )


# ── 单例工厂 ──────────────────────────────

_bert_classifier: Optional[BERTIntentClassifier] = None


def get_bert_classifier() -> Optional[BERTIntentClassifier]:
    """获取 BERT 意图分类器单例（首次调用时加载模型）"""
    global _bert_classifier
    if _bert_classifier is None:
        try:
            _bert_classifier = BERTIntentClassifier()
            _bert_classifier.load()
        except Exception as e:
            logger.warning(f"[BERT] 分类器初始化失败，BERT 层不可用: {e}")
            _bert_classifier = None
    return _bert_classifier


# ── CLI 入口 ───────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT 保险意图分类器")
    sub = parser.add_subparsers(dest="command")

    # predict
    p = sub.add_parser("--predict", help="单条预测")
    p.add_argument("--query", required=True, help="查询文本")
    p.add_argument("--model", default=None, help="微调模型路径")

    # train
    t = sub.add_parser("--train", help="训练模型")
    t.add_argument("--data", required=True, help="训练数据 JSON 路径")
    t.add_argument("--output", required=True, help="输出模型目录")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=16)

    # evaluate
    e = sub.add_parser("--evaluate", help="评估模型")
    e.add_argument("--model", required=True, help="模型路径")
    e.add_argument("--data", required=True, help="评估数据 JSON 路径")

    # generate-data
    g = sub.add_parser("--generate-data", help="LLM 扩增训练数据")
    g.add_argument("--seed", required=True, help="种子数据 JSON 路径")
    g.add_argument("--output", required=True, help="输出路径")
    g.add_argument("--variants", type=int, default=5, help="每条种子生成的变体数")

    args = parser.parse_args()

    if args.command == "--predict":
        cls = _bert_classifier
        if not cls:
            cls = BERTIntentClassifier()
            cls.load()
        result = cls.predict(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "--train":
        with open(args.data) as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]
        cls = BERTIntentClassifier()
        cls.load()
        cls.train(
            texts,
            labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.output,
        )

    elif args.command == "--evaluate":
        cls = BERTIntentClassifier.from_pretrained(args.model)
        with open(args.data) as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        true_labels = [d["label"] for d in data]
        results = cls.predict_batch(texts)
        correct = sum(1 for r, t in zip(results, true_labels) if r["intent"] == t)
        acc = correct / len(results) if results else 0

        # 混淆矩阵
        from collections import defaultdict

        matrix = defaultdict(lambda: defaultdict(int))
        for r, t in zip(results, true_labels):
            matrix[t][r["intent"]] += 1

        print(f"Accuracy: {acc:.4f} ({correct}/{len(results)})")
        print("\nPer-class accuracy:")
        for label in INTENT_LABELS:
            total = sum(matrix[label].values())
            correct_c = matrix[label][label]
            print(
                f"  {label}: {correct_c}/{total} = {correct_c / total:.3f}"
                if total > 0
                else f"  {label}: N/A"
            )

    elif args.command == "--generate-data":
        BERTIntentClassifier.generate_training_data(
            args.seed, args.output, num_variants=args.variants
        )
